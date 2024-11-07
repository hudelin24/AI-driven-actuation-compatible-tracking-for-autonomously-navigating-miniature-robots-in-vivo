import torch
import torch.nn as nn
from models.model_registry import MODEL_REGISTRY
from models.model_utils import DropPath, trunc_normal_
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dim_out=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or dim_in
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, x):
        """
        Args:
            x: sequence                     [bsz, l, dim]
        """
        bsz, l, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, l, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, l, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Cblock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Tblock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        """
        Args:
            x           [bsz, num_temporal_tokens * num_spatial_tokens + 1, embed_dim]
            B            bsz
            T            num_temporal_tokens
            W            out_W, out_W * out_H = num_spatial_tokens
        """
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_pre_token = x[:,0,:].unsqueeze(1)
            pre_token = init_pre_token.repeat(1, T, 1)
            pre_token = rearrange(pre_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((pre_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of PRE token
            pre_token = res_spatial[:,0,:]
            pre_token = rearrange(pre_token, '(b t) m -> b t m',b=B,t=T)
            pre_token = torch.mean(pre_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_pre_token, x), 1) + torch.cat((pre_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class MagClipEmbed(nn.Module):
    def __init__(self, mag_size, patch_size, stride_size, embed_dim=128):
        """
        Args:
            mag_size: list                     [in_chans, T, H, W]
            patch_size: list                   [t, h, w]
            stride_size: list                  [t, h, w]
            embed_dim: int
        """
        super().__init__()
        in_chans = mag_size[0]
        self.mag_size = mag_size
        self.patch_size = patch_size
        #self.num_spatial_tokens = (mag_size[-1] // patch_size[-1]) * (mag_size[-2] // patch_size[-2])
        #self.num_temporal_tokens = mag_size[-3] // patch_size[-3]
        self.num_spatial_tokens = 2 * 3
        self.num_temporal_tokens = mag_size[-3] // patch_size[-3]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)                    #[bsz, embed_dim, out_t, out_H, out_W]
        W = x.size(-1)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = x.flatten(2).transpose(1, 2)    #[bsz * t, num_space_patches, embed_dim]
        return x, W


class MagEmbed(nn.Module):
    def __init__(self, mag_size, patch_size, embed_dim=128):
        """
        Args:
            mag_size: list                     [in_chans, r_len, H, W]
            patch_size: list                   [1, H, W]
            embed_dim: int
        """
        super().__init__()
        in_chans = mag_size[0]
        self.mag_size = mag_size
        self.patch_size = patch_size
        self.num_patches = (mag_size[-1] // patch_size[-1]) * (mag_size[-2] // patch_size[-2]) * (mag_size[-3] // patch_size[-3])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    #[bsz, embed_dim, out_t, out_H, out_W]
        x = x.flatten(2).transpose(1, 2)    #[bsz, num_patches, embed_dim]
        return x
    

class MagCalibTransformer(nn.Module):
    """ Magnetic Calibration Transformer
    """
    def __init__(self, mag_size=(3,4,7), num_calib=12, embed_dim=128, depth=6, 
                 num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.mag_embed = nn.Linear(mag_size[0], embed_dim)
        self.mag_size = mag_size

        #Positional Embeddings
        self.calib_pos_embed = nn.Parameter(torch.zeros(1, num_calib, embed_dim))
        self.pre_pos_embed = nn.Parameter(torch.zeros(1, mag_size[-1] * mag_size[-2], embed_dim))
        #self.pre_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        #Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Cblock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                   act_layer=act_layer, norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        
        #Prediction head
        self.head = nn.Linear(embed_dim, mag_size[0])

        #Initialization 
        #trunc_normal_(self.pre_token, std=.02)
        trunc_normal_(self.pre_pos_embed, std=.02)
        trunc_normal_(self.calib_pos_embed, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'calib_pos_embed', 'pre_pos_embed'}
    
    def forward(self, x):
        """
        Args:
            x: magnetic map         [bsz, in_chans, r_len (1), num_calib]
        """
        #print(x.shape)
        bsz, num_calib = x.shape[0], x.shape[-1]
        mag_embeddings = self.mag_embed(x.permute(0,3,2,1)).squeeze(2)         #[bsz, num_calib, embed_dim]
        mag_embeddings = mag_embeddings + self.calib_pos_embed.expand(bsz, -1, -1)
        input_token = self.pos_drop(torch.cat((mag_embeddings, self.pre_pos_embed.expand(bsz, -1, -1)), 1))   #[bsz, num_calib + num_pre, embed_dim]
        for blk in self.blocks:
            input_token = blk(input_token)
        input_token = self.norm(input_token)

        return self.head(input_token[:,num_calib:,:]).reshape(bsz, self.mag_size[1], self.mag_size[2], self.mag_size[0])

    
class MagTrackingTransformer(nn.Module):
    """ Magnetic Tracking Transformer
    """
    def __init__(self, mag_size, patch_size, stride_size, out_dim=3, embed_dim=128, depth=6, 
                 num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 attention_type='divided_space_time'):

        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.embed_dim = embed_dim
        self.mag_clip_embed = MagClipEmbed(mag_size=mag_size, patch_size=patch_size, stride_size=stride_size, embed_dim=embed_dim)
        num_spatial_tokens = self.mag_clip_embed.num_spatial_tokens
        num_temporal_tokens = self.mag_clip_embed.num_temporal_tokens

        ## Positional Embeddings
        self.pre_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_spatial_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_temporal_tokens, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                Tblock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                act_layer=act_layer, norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Prediction head
        self.head = nn.Linear(embed_dim, out_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pre_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Tblock' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'pre_token', 'time_embed'}

    def forward_features(self, x):
        """
        Args:
            x               [bsz, in_chans, in_T, in_H, in_W]
        """
        B, T = x.shape[0], self.mag_clip_embed.num_temporal_tokens
        x, W = self.mag_clip_embed(x)                      #x -> [(bsz out_T), (out_H out_W), embed_dim]
        pre_tokens = self.pre_token.expand(x.size(0), -1, -1)   #[(bsz out_T), 1, embed_dim]
        x = torch.cat((pre_tokens, x), dim=1)                   #[(bsz out_T), 1 + (out_H out_W), embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        pre_tokens = x[:B, 0, :].unsqueeze(1)        #[bsz, 1, embed_dim]
        x = x[:,1:]             #[(bsz out_T), (out_H out_W), embed_dim]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((pre_tokens, x), dim=1)   #[bsz, 1 + (out_H out_W out_T), embed_dim]

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)  

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)        #[bsz, embed_dim]
        x = self.head(x)                    
        return x

@MODEL_REGISTRY.register()
class mct_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = MagCalibTransformer(mag_size=cfg.MODEL_MCT.MAG_SIZE, num_calib=cfg.MODEL_MCT.NUM_CALIB, 
                                         embed_dim=cfg.MODEL_MCT.EMBED_DIM, depth=cfg.MODEL_MCT.DEPTH, 
                                         num_heads=cfg.MODEL_MCT.NUM_HEADS, mlp_ratio=cfg.MODEL_MCT.MLP_RATIO, 
                                         qkv_bias=cfg.MODEL_MCT.QKV_BIAS, qk_scale=cfg.MODEL_MCT.QKV_SCALE, 
                                         drop_rate=cfg.MODEL_MCT.DROP_RATE, attn_drop_rate=cfg.MODEL_MCT.ATTN_DROP_RATE, 
                                         drop_path_rate=cfg.MODEL_MCT.DROP_PATH_RATE, act_layer=nn.GELU, 
                                         norm_layer=nn.LayerNorm)
                    
    def forward(self, x):
        x = self.model(x)
        return x
        
@MODEL_REGISTRY.register()
class mtt_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = MagTrackingTransformer(mag_size=cfg.MODEL_MTT.MAG_SIZE, patch_size=cfg.MODEL_MTT.PATCH_SIZE, 
                                            stride_size = cfg.MODEL_MTT.STRIDE_SIZE, out_dim=cfg.MODEL_MTT.OUT_DIM, 
                                            embed_dim=cfg.MODEL_MTT.EMBED_DIM, depth=cfg.MODEL_MTT.DEPTH, 
                                            num_heads=cfg.MODEL_MTT.NUM_HEADS, mlp_ratio=cfg.MODEL_MTT.MLP_RATIO, 
                                            qkv_bias=cfg.MODEL_MTT.QKV_BIAS, qk_scale=cfg.MODEL_MTT.QKV_SCALE, 
                                            drop_rate=cfg.MODEL_MTT.DROP_RATE, attn_drop_rate=cfg.MODEL_MTT.ATTN_DROP_RATE, 
                                            drop_path_rate=cfg.MODEL_MTT.DROP_PATH_RATE, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                            attention_type=cfg.MODEL_MTT.ATTN_TYPE)
            
    def forward(self, x):
        x = self.model(x)
        return x

