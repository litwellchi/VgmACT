"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast

# from prismatic.models.backbones.llm import LLMBackboneshared 
# from prismatic.models.backbones.llm.prompting import PromptBuilder
# from prismatic.models.backbones.vision import VisionBackbone
# from prismatic.models.vlms.base_vlm import VLM
# from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT
from vasim_model.cvap import ContrastiveModel
from vasim_model.conditional_project import ModalityCompressor, TemporalTransformerCondition, VideoTimeStepScheduler

import sys 

# sys.path.insert(1,'/aifs4su/mmcode/worldm/RoboCrafter')
sys.path.insert(1,'/aifs4su/mmcode/worldm/videoact/VgmACT/DynamiCrafter')
from scripts.evaluation.inference import instantiate_from_config,load_model_checkpoint
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
import torchvision.transforms as transforms
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class VGM(nn.Module):
    def __init__(self,
                 config_yaml: str,
                 ckpt_path: str,
                 sys_path='/aifs4su/mmcode/worldm/videoact/VgmACT/DynamiCrafter',
                 proj_dim: int = 4096,
                 fake_ddpm_step=918,
                 mode='fix',
                 mask_video_prob = 0.2,
                 load_concate_frame=True,
                 use_vgm_prob = 0.8):
        super().__init__()
        from omegaconf import OmegaConf
        import sys 
        sys.path.insert(1,sys_path)
        from scripts.evaluation.inference import instantiate_from_config,load_model_checkpoint

        self.config_yaml = config_yaml
        self.ckpt_path = ckpt_path
        self.proj_dim = proj_dim
        self.fake_ddpm_step = fake_ddpm_step
        perframe_ae = True
        self.mask_video_prob = mask_video_prob
        
        config = OmegaConf.load(config_yaml)
        self.model_config = config.pop("model", OmegaConf.create())
        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        self.model_config['params']['unet_config']['params']['use_checkpoint'] = False
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        h,w = self.model_config['params']['image_size']
        self.video_length = self.model_config['params']['unet_config']['params']['temporal_length']
        self.default_image_resolution=(h,w)


        # state_dict.name
        vgm = instantiate_from_config(self.model_config)
        vgm.perframe_ae = perframe_ae
        # print("checkpoint", self.ckpt_path)
        assert os.path.exists(self.ckpt_path), "Error: checkpoint Not Found!"
        self.vgm = load_model_checkpoint(vgm, ckpt_path)
        self.init_projection(proj_dim)
        self.all_module_keys=['projection','image_compressor','lang_compressor']
        for module_keys, _ in self.vgm.named_children():
            self.all_module_keys.append("vgm." + module_keys)

        # 确保 logit_scale 是一个 1D 张量
        if isinstance(vgm.cond_stage_model.model.logit_scale, torch.Tensor):
            vgm.cond_stage_model.model.logit_scale = nn.Parameter(
                vgm.cond_stage_model.model.logit_scale.view(1)
            )
        # 确保 logit_scale 是一个 1D 张量
        if isinstance(vgm.embedder.model.logit_scale, torch.Tensor):
            vgm.embedder.model.logit_scale = nn.Parameter(
                vgm.embedder.model.logit_scale.view(1)
            )

        self.scheduler = VideoTimeStepScheduler(
            total_ddpm_steps=1000,
            ddim_steps_list=[5, 10, 50], # ususally video generation ddim step is 5,10,or50
            prefer_late_steps=True,
            device=torch.device("cuda")
        )

        self.load_concate_frame=load_concate_frame
        self.use_vgm_prob = use_vgm_prob

        if mode=='freeze':
            self.vgm.eval()
        elif mode=='lora':
            self.set_trainable_param(use_lora=True)
        elif mode=='full':
            self.set_trainable_param(use_lora=False)
    
        self.vgm.embedder.eval()
        self.vgm.image_proj_model.eval()

        overwatch.info(
        f" ===== Loading [bold blue] Video Generation Model [/] ====\n"
        f"Found Config =>> Loading Video Generation Model config from [bold]{self.config_yaml}[/] with:\n"
        f"             Checkpoint Path =>> [underline]`{self.ckpt_path}`[/]"
        f" Running VGM backbone (if training) in =>> [underline]`{mode}`[/]"
        )


    def init_projection(self, proj_dim):
        # Version1.0  hidden_size = h*w*c
        # Version2.0  hidden_size = h*w*c*t
        h,w = self.model_config['params']['image_size']
        c = self.model_config['params']['channels']
        t = self.model_config['params']['unet_config']['params']['temporal_length']
        hidden_size = h*w*c*t
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.projection = Mlp(in_features=hidden_size,
            hidden_features=int(hidden_size/8),
            out_features=self.proj_dim, 
            act_layer=approx_gelu, 
            drop=0)
        
        self.projection = TemporalTransformerCondition.from_input_shape(
            input_shape=(c, t, h, w),  # (C, T, H, W)
            proj_dim=self.proj_dim,               # 输出给 diffusion 的 cond 向量维度
            target_model_size_mb=100    # 目标模型大小（可选）
        )

        self.image_compressor = ModalityCompressor(input_dim=1024, output_dim=self.proj_dim, method='mlp')
        self.lang_compressor  = ModalityCompressor(input_dim=1024, output_dim=self.proj_dim, method='mlp')

    def set_trainable_param(self, use_lora=False):
        # 冻结所有参数
        for param in self.vgm.parameters():
            param.requires_grad = False

        # 仅解冻 `vgm.model` 和 `projection`
        for param in self.vgm.model.parameters():
            param.requires_grad = True

        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.image_compressor.parameters():
            param.requires_grad = True
        for param in self.lang_compressor.parameters():
            param.requires_grad = True
        
        self._init_unet_lora(use_lora=use_lora)

    def _init_unet_lora(self, use_lora):
        import peft
        if use_lora:
            lora_config = peft.LoraConfig(
                r=4,
                lora_alpha=1,
                # only diffusion_model has these modules
                target_modules=["to_k", "to_v", "to_q"],
                lora_dropout=0.0,
            )
            self.vgm.model.diffusion_model = peft.get_peft_model(
                self.vgm.model.diffusion_model, lora_config)
            self.vgm.model.diffusion_model.print_trainable_parameters()


    def get_image_transform(self):
        if self.load_concate_frame:
            video_size = (self.default_image_resolution[0]*8,self.default_image_resolution[1]*8*4) #version 2.9: load 连续四帧，包括future frame
        else:
            video_size = (self.default_image_resolution[0]*8,self.default_image_resolution[1]*8)
        transform = transforms.Compose([
            transforms.Resize(video_size),
            # transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        return transform

    def forward(self):
        pass

    def create_time_varying_mask(self, shape, rate=10, device='cuda'):
        # TODO mask 还没加上，不完全是first frame condition
        b, c, t, h, w = shape
        time_mask = torch.linspace(0, 1, t, device=device)  
        time_mask[1:]=0
        time_mask[0]=1
        time_mask = time_mask.view(1, 1, t, 1, 1).expand(b, c, t, h, w)
        return time_mask
    
    def get_video_loss(self, x_start, model_output, noise, t):
        if self.vgm.parameterization == "x0":
            target = x_start
        elif self.vgm.parameterization == "eps":
            target = noise
        elif self.vgm.parameterization == "v":
            target = self.vgm.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.vgm.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        return loss_simple

    def one_ddim_step_forward(
        self, prompts, images, n_samples=1, ddim_steps=50, ddim_eta=1.0,
        unconditional_guidance_scale=1.0, cfg_img=None, fs=16, text_input=False,
        multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform',
        guidance_rescale=0.0, train=False, fix_video_timesteps=None, 
        pre_gen_video=None, 
        **kwargs
    ):
        """
        Forward pass for one DDIM step. Extracts cognitive features.

        Args:
            prompts: List of text prompts.
            images: List of images or video frames.
            train (bool): Whether in training mode.

        Returns:
            cognition_features: Tensor of shape [B, T, D]
            video_loss (optional): Only during training
        """
        # === Setup ===
        device = self.vgm.model.diffusion_model.out[2].weight.device
        images = torch.stack(images, dim=0).to(device)  # [B, C, H, W]
        batch_size, c, h, w = images.shape
        t = 4  # temporal length
        fs = torch.tensor([self.video_length] * batch_size, dtype=torch.long, device=device)
        kwargs.update({"fs": fs})

        # === Handle video format ===
        if images.ndim == 4:  # [B, C, H, W]
            if images.shape[3] / images.shape[2] == t: # video b c h w
                videos = images.view(batch_size, c, h, t, h).permute(0, 1, 3, 2, 4)  # [B, C, T, H, W]
            else:
                videos = images.unsqueeze(2)  # -> [B, C, 1, H, W]
                videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=self.video_length)
            img = images
        else:
            raise ValueError("Expecting images of shape [B, C, H, W]")

        # === Text and Image Embeddings ===
        if not text_input:
            prompts = [""] * batch_size
        cond_emb = self.vgm.get_learned_conditioning(prompts).detach().clone()  # [B, L, D]
        img_emb = self.vgm.image_proj_model(self.vgm.embedder(img)).detach().clone()
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

        # === Hybrid Conditioning ===
        if self.vgm.model.conditioning_key == 'hybrid':
            b, c, t, h, w = videos.shape
            x = rearrange(videos, 'b c t h w -> (b t) c h w')
            z = self.vgm.encode_first_stage(x)
            z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)

            if loop or interp:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                img_cat_cond = z[:,:,:1,:,:].detach().clone()
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=t)
            cond["c_concat"] = [img_cat_cond]

        # === Unconditional Guidance ===
        if unconditional_guidance_scale != 1.0:
            if self.vgm.uncond_type == "empty_seq":
                uc_emb = self.vgm.get_learned_conditioning([""] * batch_size)
            elif self.vgm.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            else:
                raise ValueError(f"Unsupported uncond_type: {self.vgm.uncond_type}")

            uc_img_emb = self.vgm.image_proj_model(
                self.vgm.embedder(torch.zeros_like(img))
            )
            uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
            if self.vgm.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        # === Optional Secondary Unconditional Condition ===
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
            if self.vgm.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        # === Sample Noise and Time Step ===
        x_start = z.detach().clone()
        noise = torch.randn_like(x_start)
        if train:
            t_vid = self.scheduler.sample_train_timestep(batch_size=x_start.shape[0])
        elif not train and fix_video_timesteps is not None:
            t_vid = self.scheduler.get_infer_timestep(batch_size=x_start.shape[0], t_value=fix_video_timesteps)
        x_noisy = self.vgm.q_sample(x_start=x_start, t=t_vid, noise=noise)

        # === Apply Conditional Mask ===
        cond_mask = self.create_time_varying_mask(z.shape)
        if cond_mask is not None and not train:
            x_noisy = x_start * cond_mask + (1. - cond_mask) * x_noisy

        # === Cognition Features ===
        img_cognition_features = self.image_compressor(img_emb)            # [B, 1, D]
        cond_cognition_features = self.lang_compressor(cond_emb)          # [B, 1, D]

        # === Video Feature Masking/Projection ===
        # # only Masking when training
        # if train and torch.rand(1).item() < self.mask_video_prob:
        #     video_features = torch.zeros_like(cond_cognition_features)    # [B, 1, D]
        #     cognition_features = torch.cat([video_features, img_cognition_features, cond_cognition_features], dim=1)
        #     return (cognition_features, 0, x_start) if train else cognition_features

        # # === Full Forward or Skip Training ===
        # # only skil when training
        # use_vgm = True if not train else (torch.rand(1).item() < self.use_vgm_prob)
        # if use_vgm:
        #     video_samples = self.vgm.apply_model(x_noisy, t_vid, cond, **kwargs)
        #     x0_hat_target = self.vgm.predict_start_from_z_and_v(x_noisy, t_vid, video_samples)
        # else:
        #     x0_hat_target = x_noisy

        # # === Final Cognition Feature Assembly ===
        # video_features = self.projection(x0_hat_target,t_vid)
        cognition_features = torch.cat([img_cognition_features, cond_cognition_features], dim=1)

        # === Compute Loss (if Training) ===
        # if train:
            # video_loss = self.get_video_loss(x_start, x0_hat_target, noise, t_vid)
            # return cognition_features, video_loss, x0_hat_target

        return cognition_features


class VgmACT(nn.Module):
    def __init__(
        self,
        vgm: VGM,
        action_model_type: str = 'DiT-B',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.action_model = ActionModel(model_type = action_model_type, 
                                            token_size = token_size, 
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size,
                                            condition_token_len=1+1) # img+lang
        self.vgm = vgm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema

        h,w = self.vgm.model_config['params']['image_size']
        c = self.vgm.model_config['params']['channels']
        t = self.vgm.model_config['params']['unet_config']['params']['temporal_length']

        self.cvap = ContrastiveModel(
            image_channels=c,
            pose_dim=action_dim,
            embedding_dim=128,
        )

        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion','cvap']
        else:
            self.all_module_keys = ['action_model']

        for module_keys in self.vgm.all_module_keys:
            # [name for name, _ in self.vlm.vgm.named_children()]
            # [name for name, _ in self.vlm.vgm.vgm.named_children()]
            self.all_module_keys.append("vgm." + module_keys)

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model']
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vgm.trainable_module_keys:
            keys.append("vgm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks = None,
        lang = None,
        ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # # extract the last hidden state and the learnable EOS token feature
        num_params = sum(p.numel() for p in self.vgm.parameters())
        num_trainable_params = sum(p.numel() for p in self.vgm.parameters() if p.requires_grad)
        # overwatch.info(
        #     f"# Parameters after re-set (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        # )

        cognition_features = self.vgm.one_ddim_step_forward(prompts=lang,images=pixel_values,train=True) # B, 1, D

        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

        # Action model forward and compute loss
        act_loss = self.action_model.loss(actions_repeated, cognition_features_repeated)


        # === 5. CVAP contrastive loss ===
        # Flatten for contrastive model
        # 只计算生成的第一个关键帧和第一个action应该是对应的
        # x0_hat_flat = x0_hat[:,1,:,:,:].detach()  # [B, C, H, W] if input is [B, T, C, H, W]
        # pose_flat = actions_future[:, 0, :].detach()  # use first future action as anchor [B, D]

        # z_img, z_pose = self.cvap(x0_hat_flat, pose_flat)
        # contrastive_loss = self.cvap.nt_xent_loss(z_img, z_pose)


        # loss = act_loss + video_loss.mean()
        loss = act_loss 
        return loss,  cognition_features, 0

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP wrapping policy that applies to `vgm.model`, `vgm.projection`, and specific prismatic modules."""

        # 1️⃣ 获取 vgm.model 的 FSDP wrapping policy
        vgm_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={type(self.vgm.vgm.model)},  # 确保正确获取类型
        )

        # 2️⃣ 定义 `projection` 的 wrapping policy
        projection_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={type(self.vgm.projection),type(self.vgm.image_compressor),type(self.vgm.lang_compressor)},  # 确保正确获取类型
        )

        # 3️⃣ 定义 `prismatic` 相关模块的 wrapping policy
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT, ContrastiveModel},
        )

        # 4️⃣ 合并所有 Policy，使得 FSDP 仅包裹 `vgm.model`、`projection` 和 `prismatic` 相关模块
        return partial(
            _or_policy,
            policies=[
                vgm_fsdp_wrapping_policy,
                projection_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        self,
        vgm:VGM,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        full_ckpt=None,
        pretrain_action_model=None,
        vgm_param_mode="freeze",
        **kwargs,
    ) -> VgmACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        # Initialize CogACT
        vgmact = VgmACT(vgm,
                        token_size = 4096,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        )
        if full_ckpt is not None:
            model_state_dict=torch.load(full_ckpt, map_location="cpu")["model"]
         # Load ActionModel from Checkpoint
            overwatch.info(
                f" ===== Loading [bold blue] Pretrained Weight [/] ====\n"
                f"=>> Loading Action model and VGM from [bold]{full_ckpt}[/] with:\n"
                )
            if "action_model" in model_state_dict:
                vgmact.action_model.load_state_dict(model_state_dict["action_model"])
                if "ema_diffusion" in model_state_dict and use_ema:
                    vgmact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
                elif use_ema:
                    vgmact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
            else:
                overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
           

            try:
                vgmact.vgm.projection.load_state_dict(model_state_dict["vgm.projection"])
                vgmact.vgm.image_compressor.load_state_dict(model_state_dict["vgm.image_compressor"])
                vgmact.vgm.lang_compressor.load_state_dict(model_state_dict["vgm.lang_compressor"])
                overwatch.info("Loading vgm.projection,vgm.lang_compressor,vgm.image_compressor successfully.")
            except:
                overwatch.warning("No vgm.lang_compressor,vgm.image_compressor found in the pretrained checkpoint. Initializing a new one.")

            missing_keys_info = vgmact.vgm.vgm.model.load_state_dict(model_state_dict["vgm.vgm.model"], strict=False)
            if not missing_keys_info.missing_keys and not missing_keys_info.unexpected_keys:
                overwatch.info("Loading vgm.model successfully.")
            else:
                overwatch.warning("Loading vgm.model warning: Some missing keys, or  unexpected keys.")

        elif pretrain_action_model is not None:
            overwatch.info(f"Using pretrained action model from {pretrain_action_model}")
            action_model_state_dict = torch.load(pretrain_action_model, map_location="cpu")["model"]
            if "action_model" in action_model_state_dict:
                try:
                    vgmact.action_model.load_state_dict(action_model_state_dict["action_model"], strict=False)
                except:
                    model_dict = vgmact.action_model.state_dict()
                    pretrained_dict = action_model_state_dict["action_model"]

                    # 只保留形状匹配的参数
                    compatible_dict = {
                        k: v for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape
                    }
                    # 更新模型参数
                    model_dict.update(compatible_dict)
                    vgmact.action_model.load_state_dict(model_dict)

        if vgm_param_mode =='freeze':
            for name, param in vgmact.vgm.vgm.named_parameters():
                param.requires_grad = False
            # for name, param in vgmact.action_model.named_parameters():
            #     param.requires_grad = False
        
        # vgmact.check_trainable_param()
        return vgmact        
    def check_trainable_param(self):
        seen = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                short_name = ".".join(name.split(".")[:2])
                if short_name not in seen:
                    seen.add(short_name)
        overwatch.info(f"=== [bold] trainable paramater {seen} [/] ===")                


    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        dual_system: bool = False,
        dual_system_video: list = None,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
        was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """

        # TODO Image transform
        device = self.vgm.vgm.model.diffusion_model.out[2].weight.device
        image_transform = self.vgm.get_image_transform()
        pixel_values = image_transform(image).to(device)

        # print(pixel_values.shape)
        cognition_features = self.vgm.one_ddim_step_forward(prompts=[instruction],
                                                            images=[pixel_values],
                                                            train=False,
                                                            fix_video_timesteps=918) # [B, 1, D]
        # print(cognition_features.shape)
        B = cognition_features.shape[0]
        model_dtype = next(self.action_model.net.parameters()).dtype
        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]

        using_cfg = cfg_scale > 1.0
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            # uncondition_video = self.action_model.net.z_embedder_video.uncondition
            uncondition_image = self.action_model.net.z_embedder_image.uncondition
            uncondition_text = self.action_model.net.z_embedder_text.uncondition
            # uncondition_video = uncondition_video.unsqueeze(0)  #[1, D]
            # uncondition_video = uncondition_video.expand(B, 1, -1) #[B, 1, D]
            uncondition_image = uncondition_image.unsqueeze(0)  #[1, D]
            uncondition_image = uncondition_image.expand(B, 1, -1) #[B, 1, D]
            uncondition_text = uncondition_text.unsqueeze(0)  #[1, D]
            uncondition_text = uncondition_text.expand(B, 1, -1) #[B, 1, D]
            uncondition = torch.cat([uncondition_image, uncondition_text], 1)
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    

if __name__ == "__main__":
    config_yaml = "/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_v1.0_rt1/configs/model_infer.yaml"
    ckpt_path = "/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_v1.0_rt1/checkpoints/epoch=13-step=9000.ckpt"
    vgm = VGM(config_yaml=config_yaml,
              ckpt_path=ckpt_path).cuda()
    image = [torch.zeros((3,128,128))*4]
    lang=['a','a','a','a']
    vgm.one_ddim_step_forward(lang, image)