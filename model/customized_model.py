import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from .segmenter import BaseSegmenter
from timm.models.registry import register_model
from .utils import  update_mamba_config
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops

from vmamba_model.vmamba import SS2D, LayerNorm2d, Linear2d
from .swin_image_encoder import MultiModalSwinTransformer


class FuseLayer(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, out_dim, bias=False) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim_1+in_dim_2, out_dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, in_1, in_2):
        if in_1.shape[-1] < in_2.shape[-1]:
            in_1 = F.interpolate(in_1, size=in_2.shape[-2:], mode='bilinear', align_corners=True)
        elif in_1.shape[-1] > in_2.shape[-1]:
            in_2 = F.interpolate(in_2, size=in_1.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat((in_1, in_2), dim=1)
        x = self.fusion(x)
        return x

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class MorphologyAwareBoundaryDelineation(nn.Module):
    def __init__(
            self,         
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            input_res=32,
            window_size=4,
            **kwargs,
        ):
        super().__init__()

        self.input_res = input_res
        self.window_size = window_size

        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.ss2d.in_proj = Linear2d(d_model, self.ss2d.in_proj.weight.shape[0], bias=bias)
        self.input_res = min(input_res, 16)

        self.text_avgsum = True
        if self.text_avgsum:
            self.text_weights = nn.Parameter(torch.ones(16) / 16, requires_grad=True)
    
    def forward(self, x: torch.Tensor,  **kwargs):
        img, text = x
        B, C, H, W = img.shape
        _, _, _, L = text.shape
        img_o = img

        img = img.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        img = img.permute(0, 2, 4, 1, 3, 5)
        img = img.contiguous().view(B, H // self.window_size, W // self.window_size, C, self.window_size*self.window_size)
        img = einops.rearrange(img, 'b h w c s -> b (h w) s c')

        text = text.mean(dim=3)
        text = einops.repeat(text, 'b c l -> b P l c', P = img.shape[1])
        mix_pre = torch.cat([img, text], dim=2)

        mix_pre = mix_pre.permute(0,3,1,2)

        _, _, Hm, Wm = mix_pre.shape

        mix = mix_pre
        out = self.ss2d(mix)

        out = out.permute(0,2,3,1)

        img_f, text_f = out[:,:,:-L,:], out[:,:,-L:,:]
        img_f = img_f.view(B, H // self.window_size, W // self.window_size, C, self.window_size*self.window_size) 
        img_f = einops.rearrange(img_f, 'b h w c s -> b c h w s')
        img_f = img_f.permute(0, 3, 1, 4, 2)
        img_f = img_f.contiguous().view(B, C, H, W)

        if self.text_avgsum:
            weights = self.text_weights
            if weights.numel() != text_f.shape[1]:
                weights = F.interpolate(
                    weights.view(1, 1, -1),
                    size=text_f.shape[1],
                    mode="linear",
                    align_corners=False,
                ).view(-1)
            normalized_weights = F.softmax(weights, dim=0)
            text_f = (text_f * normalized_weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            text_f = text_f.mean(dim=1)
    
        text_f = einops.repeat(text_f, 'b l c -> b c l L', L=L)
        

        out = [img_f, text_f]
        return out



class VSSBlock(nn.Module):
    def __init__(
        self,
        forward_coremm='SS2D',
        window_size=4,
        **kwargs,
    ):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        dim = kwargs['dim']
        drop_path = kwargs['drop_path']
        self.ln_1 = norm_layer(dim)
        self.forward_coremm = forward_coremm
        if not forward_coremm:
            raise

        elif forward_coremm == 'scm' or forward_coremm == 'SS2D':
            self.self_attention = SS2D(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                **kwargs,
            )
        elif forward_coremm == 'bdm':
            self.self_attention = MorphologyAwareBoundaryDelineation(
                d_model=dim,
                window_size=window_size,
                d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                channel_first=kwargs['channel_first'],
            )
        else:
            raise
        self.drop_path = DropPath(drop_path)

        gmlp = kwargs['gmlp']
        mlp_ratio = kwargs['mlp_ratio']
        mlp_act_layer = kwargs['mlp_act_layer']
        mlp_drop_rate = kwargs['mlp_drop_rate']
        channel_first = kwargs['channel_first']
        _MLP = Mlp if not gmlp else gMlp
        self.ln_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = _MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, input: torch.Tensor):
        
        if self.forward_coremm == 'scm':
            text = input[1]
            input = input[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(text).chunk(6, dim=1)
            
            if isinstance(input, torch.Tensor):
                #ss2d
                out = self.ln_1(input)
                out = modulate(out, shift_msa, scale_msa)
                out = self.self_attention(out)
                out = input + gate_msa.unsqueeze(-1).unsqueeze(-1)*self.drop_path(out)

                #ffn
                out2 = self.ln_2(out)
                out2 = modulate(out2, shift_mlp, scale_mlp)
                out2 = self.mlp(out2)
                out2 = out + gate_mlp.unsqueeze(-1).unsqueeze(-1)*self.drop_path(out2)

                x = (out2, text)
            else:
                # input should be a list (img and global / local conditions)
                #ss2d
                out = [modulate(self.ln_1(i), shift_msa, scale_msa) if i is not None else None for i in input]
                out = self.self_attention(out)
                out = [i + gate_msa.unsqueeze(1)*self.drop_path(o) if i is not None else None for i, o in zip(input, out)]

                #ffn
                out2 = [modulate(self.ln_2(i), shift_mlp, scale_mlp) if i is not None else None for i in out]
                out2 = self.self_attention(out2)
                out2 = [i + gate_mlp.unsqueeze(1)*self.drop_path(o) if i is not None else None for i, o in zip(out, out2)]

                x = (out2, text)

        elif self.forward_coremm == 'bdm' or self.forward_coremm == 'SS2D':
            if isinstance(input, torch.Tensor):
                #ss2d
                out = self.ln_1(input)
                out = self.self_attention(out)
                out = input + self.drop_path(out)

                #ffn
                out2 = self.ln_2(out)
                out2 = self.mlp(out2)
                out2 = out + self.drop_path(out2)

                x = out2
            else:
                # input should be a list (img and global / local conditions)
                #ss2d
                out = [self.ln_1(i) if i is not None else None for i in input]
                out = self.self_attention(out)
                out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]

                #ffn
                out2 = [self.ln_2(i) if i is not None else None for i in out]
                out2 = self.self_attention(out2)
                out2 = [i + self.drop_path(o) if i is not None else None for i, o in zip(out, out2)]

                x = out2

        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        depth,
        window_size,
        downsample=None,
        forward_coremm="scm",
        **kwargs,
    ):
        super().__init__()
        drop_path = 0.
        dim = kwargs['dim']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']


        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                forward_coremm=forward_coremm,
                window_size=window_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                **kwargs,
            )
            for i in range(depth)])
        
        if True: 
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, channel_first=kwargs['channel_first'])
        else:
            self.downsample = None

    def forward(self, x, l_feat, l_mask):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        # x: b w h c
        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner




class AttrMambaDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.channel_first = True
        depths = [2,2,2,2]
        window_size = [4, 4, 4, 4]
        self.num_layers = len(depths)
        dims = kwargs['dims'][0]
        dims = [dims*8, dims*4, dims*2, dims]
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']
        self.text_guidance = nn.ModuleList()
        self.image_guidance = nn.ModuleList()
        self.stage_fusion = nn.ModuleList()
        self.multimodal_blocks = nn.ModuleList()
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)


        for i_layer in range(self.num_layers):
            layer1 = VSSLayer(
                dim = dims[i_layer],
                depth = depths[i_layer],
                window_size=window_size[i_layer],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=None,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # =================
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],

                forward_coremm='scm',
            )

            layer2 = VSSLayer(
                dim = dims[i_layer],
                depth = depths[i_layer],
                window_size=window_size[i_layer],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                downsample=None,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=kwargs['ssm_d_state'],
                ssm_ratio=kwargs['ssm_ratio'],
                ssm_dt_rank=kwargs['ssm_dt_rank'],
                ssm_conv=kwargs['ssm_conv'],
                ssm_conv_bias=kwargs['ssm_conv_bias'],
                ssm_drop_rate=kwargs['ssm_drop_rate'],
                ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'],
                # =================
                mlp_ratio=kwargs['mlp_ratio'],
                mlp_act_layer=kwargs['mlp_act_layer'],
                mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'],

                forward_coremm='bdm',
            )

            self.multimodal_blocks.append(layer1)
            self.multimodal_blocks.append(layer2)

            if i_layer==0:
                text_guidance = nn.Sequential(
                    nn.Linear(768, dims[i_layer]),
                    nn.ReLU(),
                )

                image_guidance = nn.Sequential(
                    nn.Conv2d(in_channels=dims[i_layer], out_channels=dims[i_layer], kernel_size=1),
                    nn.ReLU(),
                )
            else:
                text_guidance = nn.Sequential(
                    nn.Linear(dims[i_layer-1], dims[i_layer]),
                    nn.ReLU(),
                )
                image_guidance = nn.Sequential(
                    nn.Conv2d(in_channels=dims[i_layer-1], out_channels=dims[i_layer], kernel_size=1),
                    nn.ReLU(),
                )
            self.text_guidance.append(text_guidance)
            self.image_guidance.append(image_guidance)


            if i_layer != self.num_layers - 1:
                self.stage_fusion.append(
                    FuseLayer(
                        dims[i_layer], dims[i_layer]//2, dims[i_layer]//2
                    )
                )


        self.proj_out = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv_layer(dims[3], dims[3]//2, 3, padding=1),
        nn.Conv2d(dims[3]//2, 1, 3, padding=1),
    )

    
    def forward(self, x, l_feat, l_mask, pooler_out=None):
        img_outs = []
        txt_outs = []
        
        for i in range(self.num_layers):
            pooling_text = l_feat[..., 0]
            l_feat = l_feat.permute(0,2,1)
            _, c, h, w = x.shape
            _, l, ct = l_feat.shape

            
            text_guidance = self.text_guidance[i](pooling_text)
            l_feat_guidance = self.text_guidance[i](l_feat)
            l_feat_guidance_repeat = einops.repeat(l_feat_guidance, "b l c -> b c l L", L=l)
            x = self.image_guidance[i](x)

            layer1 = self.multimodal_blocks[2*i]
            layer2 = self.multimodal_blocks[2*i+1]

            mm_input = (x, text_guidance)
            out1 = layer1(mm_input, None, None)
            
            img_out1 = out1[0][0]

            mm_input_2 = (img_out1, l_feat_guidance_repeat)
            out2 = layer2(mm_input_2, None, None)

            img_feat = out2[0][0]
            txt_feat = out2[0][1]

            x = img_feat

            l_feat = txt_feat.mean(dim=3)

            img_outs.append(img_feat)
            txt_outs.append(txt_feat)
        
        feat = img_outs[0]
        for i in range(len(img_outs)-1):
            feat = self.stage_fusion[i](feat, img_outs[i+1])
            feat = feat + img_outs[i+1]

        out = self.proj_out(feat)

        return out




def _build_attr_mamba(model_size="tiny", **kwargs):
    config_dict = update_mamba_config(model_size)
    decoder = AttrMambaDecoder(**config_dict)

    if model_size == "base":
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
        window_size = 12
        mha = [8, 8, 8, 8]
        drop_path_rate = 0.3
    else:
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        window_size = 7
        mha = [4, 4, 4, 4]
        drop_path_rate = 0.2

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=drop_path_rate, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=True, num_heads_fusion=mha,
                                         fusion_drop=0
                                         )
    pretrained = kwargs.get("swin_pretrained", "")
    if pretrained and os.path.exists(pretrained):
        backbone.init_weights(pretrained=pretrained)
    else:
        if pretrained:
            print(f"Warning: Swin pretrained weights not found: {pretrained}. Initializing backbone weights.")
        backbone.init_weights(pretrained=None)
    
    segmenter_kwargs = {**config_dict, **kwargs}
    return BaseSegmenter(backbone, decoder, **segmenter_kwargs)


@register_model
def AttrMamba(model_size="tiny", **kwargs):
    return _build_attr_mamba(model_size=model_size, **kwargs)
