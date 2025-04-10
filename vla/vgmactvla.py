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
                 proj_tklen: int = 16,
                 proj_dim: int = 4096,
                 fake_ddpm_step=900,
                 mode='fix'):
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

        config = OmegaConf.load(config_yaml)
        self.model_config = config.pop("model", OmegaConf.create())
        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        self.model_config['params']['unet_config']['params']['use_checkpoint'] = False
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        h,w = self.model_config['params']['image_size']
        self.default_image_resolution=(h,w)


        # state_dict.name
        vgm = instantiate_from_config(self.model_config)
        vgm.perframe_ae = perframe_ae
        # print("checkpoint", self.ckpt_path)
        assert os.path.exists(self.ckpt_path), "Error: checkpoint Not Found!"
        self.vgm = load_model_checkpoint(vgm, ckpt_path)
        self.init_projection(proj_dim)
        self.all_module_keys=['projection']
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


    def set_trainable_param(self, use_lora=False):
        # 冻结所有参数
        for param in self.vgm.parameters():
            param.requires_grad = False

        # 仅解冻 `vgm.model` 和 `projection`
        for param in self.vgm.model.parameters():
            param.requires_grad = True

        for param in self.projection.parameters():
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


    def get_image_transform(self,video_size=(320,512), video_frames=16, interp=False):
        video_size = (self.default_image_resolution[0]*8,self.default_image_resolution[1]*8)
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
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

    def one_ddim_step_forward(self, prompts, images, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=16, text_input=False, \
                        multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, train=False, **kwargs):
        """
        model forward一次 ，取出一次forward的feature
        noise_shape = [args.bs, channels, n_frames, h, w]
        image [b,c,h,w] 
        video [b,c,t, h,w] 
        return:
            cognition_features: [B, T, D]

        """
        device = self.vgm.model.diffusion_model.out[2].weight.device
        images = torch.stack(images, dim=0).to(device)


        noise_shape = images.shape
        batch_size = images.shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=device)
        kwargs.update({"fs": fs.long()})

        if len(images.shape)==5: # video
            videos = images
            images = images[:,:,0,:,:]
        else:
            videos = images.unsqueeze(2)#bcthw #TODO
            videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=16)
            
        if not text_input:
            prompts = [""]*batch_size
        img = videos[:,:,0] #bchw
        img_emb = self.vgm.embedder(img) ## blc
        img_emb = self.vgm.image_proj_model(img_emb).detach().clone()
        cond_emb = self.vgm.get_learned_conditioning(prompts).detach().clone()
        cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
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
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            cond["c_concat"] = [img_cat_cond] # b c 1 h w
        
        if unconditional_guidance_scale != 1.0:
            if self.vgm.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = self.vgm.get_learned_conditioning(prompts)
            elif self.vgm.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            uc_img_emb = self.vgm.embedder(torch.zeros_like(img)) ## b l c
            uc_img_emb = self.vgm.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
            if self.vgm.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        ## we need one more unconditioning image=yes, text=""
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
            if self.vgm.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        
        
        x_start = z.detach().clone()
        noise = torch.randn_like(x_start)
        fix_video_timesteps=True
        if fix_video_timesteps==True:
            t_vid = torch.randint(self.fake_ddpm_step, self.fake_ddpm_step+1,(x_start.shape[0],), device=device).long()
        

        # x_noisy = self.vgm.q_sample(x_start=x_start, t=t_vid, noise=noise)
        cond_mask = self.create_time_varying_mask(z.shape)
        if cond_mask is not None:
            x_noisy = self.vgm.q_sample(x_start=x_start, t=t_vid, noise=noise)
            x_noisy = x_start * cond_mask + (1. - cond_mask) * x_noisy
        
        video_samples = self.vgm.apply_model(x_noisy, t_vid, cond, **kwargs) # b c t h w
        


        video_features = rearrange(video_samples, 'b c t h w -> b (t c h w)')# Version1.0 is B T D, Version2.0 is B 1 D
        cognition_features = self.projection(video_features).unsqueeze(1) # B 1 D
        # cognition_features = self.projection(video_features) # B T D

        if train:
            video_loss = self.get_video_loss(x_start,video_samples,noise,t_vid)
            return cognition_features, video_loss
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
                                            past_action_window_size = past_action_window_size)
        self.vgm = vgm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
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

        cognition_features, video_loss = self.vgm.one_ddim_step_forward(prompts=lang,images=pixel_values,train=True) # B, 1, D

        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

        # Action model forward and compute loss
        act_loss = self.action_model.loss(actions_repeated, cognition_features_repeated)
        loss = act_loss + video_loss.mean()
        return loss,  cognition_features

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
            module_classes={type(self.vgm.projection)},  # 确保正确获取类型
        )

        # 3️⃣ 定义 `prismatic` 相关模块的 wrapping policy
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
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
            vgmact.vgm.projection.load_state_dict(model_state_dict["vgm.projection"])
            vgmact.vgm.vgm.model.load_state_dict(model_state_dict["vgm.vgm.model"])

        return vgmact        

    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
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
        cognition_features = self.vgm.one_ddim_step_forward(prompts=[instruction],images=[pixel_values]) # B, 1, D]
        # print(cognition_features.shape)
        # import pdb;pdb.set_trace()
        B = cognition_features.shape[0]
        model_dtype = next(self.action_model.net.parameters()).dtype
        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]

        using_cfg = cfg_scale > 1.0
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            # uncondition = uncondition.repeat(1,16,1) # only use in V1
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

    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]
        
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
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
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
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