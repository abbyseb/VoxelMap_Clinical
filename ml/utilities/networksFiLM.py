import torch
import torch.nn as nn
import numpy as np
from ml.utilities import layers
from ml.utilities.modelio import LoadableModel, store_config_args


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer for gantry angles.
    Ensures 0 and 360 are identical using Sine/Cosine encoding.
    """

    def __init__(self, num_features):
        super().__init__()
        self.gamma_fc = nn.Linear(2, num_features)
        self.beta_fc = nn.Linear(2, num_features)

    def forward(self, x, angle):
        if angle is None:
            return x

        if angle.dim() == 1:
            angle = angle.unsqueeze(-1)

        # Map angle to circular coordinates (Periodic)
        angle_rad = angle * (np.pi / 180.0)
        angle_sin = torch.sin(angle_rad)
        angle_cos = torch.cos(angle_rad)
        angle_encoded = torch.cat([angle_sin, angle_cos], dim=-1)

        gamma = self.gamma_fc(angle_encoded)
        beta = self.beta_fc(angle_encoded)

        # Reshape for broadcasting (B, C, 1, 1, ...)
        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * x + beta


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_film=False):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if use_film:
            self.film = FiLMLayer(out_channels)

    def forward(self, x, angle=None):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        if self.use_film and angle is not None:
            conv2 = self.film(conv2, angle)
        return self.activation(conv1 + conv2)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_film=False):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if use_film:
            self.film = FiLMLayer(out_channels)

    def forward(self, x, angle=None):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        if self.use_film and angle is not None:
            conv2 = self.film(conv2, angle)
        return self.activation(conv1 + conv2)


class TransBlock2Dto3D(nn.Module):
    """
    Corrected: Uses Adaptive Pooling to ensure spatial dimensions (H, W)
    are reduced to (1, 1) before reshaping to 3D.
    """

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.gap(x)  # (B, C, H, W) -> (B, C, 1, 1)
        return x.view(x.shape[0], x.shape[1], 1, 1, 1)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        return self.activation(conv1 + conv2)


class ExtraBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.activation(self.conv1(x)))


class BaseNetwork(nn.Module):
    def __init__(self, im_size, use_film=False):
        super().__init__()
        self.im_size = im_size
        self.use_film = use_film

    def build_feature_list(self, im_size, start_level=2):
        enc_nf = [2 ** nb for nb in range(start_level, int(np.log2(im_size)) + 2)]
        dec_nf = enc_nf[::-1] + [3]
        return enc_nf, dec_nf


class ConcatenatedEncoder(BaseNetwork):
    def __init__(self, im_size, use_film=False, in_channels=2):
        super().__init__(im_size, use_film)
        enc_nf, dec_nf = self.build_feature_list(im_size)
        self.downarm = nn.ModuleList()
        curr_nf = in_channels
        for nf in enc_nf:
            self.downarm.append(DownBlock2D(curr_nf, nf, use_film))
            curr_nf = nf

        self.transform = TransBlock2Dto3D()
        self.uparm = nn.ModuleList()
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock(curr_nf, nf))
            curr_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(curr_nf, nf))
            curr_nf = nf

    def forward(self, source_proj, target_proj, angle=None):
        x = torch.cat([source_proj, target_proj], dim=1)
        for layer in self.downarm:
            x = layer(x, angle) if self.use_film else layer(x)
        x = self.transform(x)
        for layer in self.uparm: x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


class DualEncoder(BaseNetwork):
    def __init__(self, im_size, use_film=False, in_channels=1):
        super().__init__(im_size, use_film)
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)

        self.source_encoder = nn.ModuleList()
        self.target_encoder = nn.ModuleList()
        curr_nf = in_channels
        for nf in enc_nf:
            self.source_encoder.append(DownBlock2D(curr_nf, nf, use_film))
            self.target_encoder.append(DownBlock2D(curr_nf, nf, use_film))
            curr_nf = nf

        self.transform = TransBlock2Dto3D()
        curr_nf = enc_nf[-1] * 2  # Concatenated output of two encoders

        self.uparm = nn.ModuleList()
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock(curr_nf, nf))
            curr_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(curr_nf, nf))
            curr_nf = nf

    def forward(self, source_proj, target_proj, angle=None):
        xs, xt = source_proj, target_proj
        for s_l, t_l in zip(self.source_encoder, self.target_encoder):
            xs = s_l(xs, angle) if self.use_film else s_l(xs)
            xt = t_l(xt, angle) if self.use_film else t_l(xt)

        # Concatenate on channel dim BEFORE 2D->3D transform
        x = torch.cat([xs, xt], dim=1)
        x = self.transform(x)

        for layer in self.uparm: x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


class SeparateProjectionVolumeEncoder(BaseNetwork):
    def __init__(self, im_size, use_film=False, proj_channels=1, vol_channels=1):
        super().__init__(im_size, use_film)
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)

        self.proj_encoder = nn.ModuleList()
        curr_p = proj_channels
        for nf in enc_nf:
            self.proj_encoder.append(DownBlock2D(curr_p, nf, use_film))
            curr_p = nf

        self.vol_encoder = nn.ModuleList()
        curr_v = vol_channels
        for nf in enc_nf:
            self.vol_encoder.append(DownBlock3D(curr_v, nf, use_film))
            curr_v = nf

        self.transform = TransBlock2Dto3D()
        curr_nf = enc_nf[-1] * 2

        self.uparm = nn.ModuleList()
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock(curr_nf, nf))
            curr_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(curr_nf, nf))
            curr_nf = nf

    def forward(self, target_proj, source_vol, angle=None):
        xp, xv = target_proj, source_vol
        for p_l in self.proj_encoder: xp = p_l(xp, angle) if self.use_film else p_l(xp)
        for v_l in self.vol_encoder: xv = v_l(xv, angle) if self.use_film else v_l(xv)

        xp = self.transform(xp).expand(-1, -1, *xv.shape[2:])
        x = torch.cat([xp, xv], dim=1)
        for layer in self.uparm: x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


class BroadcastEncoder(BaseNetwork):
    def __init__(self, im_size, use_film=False, proj_channels=1, vol_channels=1):
        super().__init__(im_size, use_film)
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)
        self.proj_conv = DownBlock3D(proj_channels, enc_nf[0], use_film)

        self.encoder = nn.ModuleList()
        curr_nf = enc_nf[0] + vol_channels
        for nf in enc_nf:
            self.encoder.append(DownBlock3D(curr_nf, nf, use_film))
            curr_nf = nf

        self.uparm = nn.ModuleList()
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock(curr_nf, nf))
            curr_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(curr_nf, nf))
            curr_nf = nf

    def forward(self, target_proj, source_vol, angle=None):
        B = source_vol.shape[0]
        xp = target_proj
        if xp.shape[0] != B:
            xp = xp.view(B, -1, *xp.shape[1:]).mean(dim=1)

        xp_3d = xp.unsqueeze(2).repeat(1, 1, self.im_size, 1, 1)
        xp_3d = self.proj_conv(xp_3d, angle) if self.use_film else self.proj_conv(xp_3d)

        # Ensure spatial alignment
        xv_down = nn.functional.interpolate(source_vol, size=xp_3d.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([xp_3d, xv_down], dim=1)

        for layer in self.encoder: x = layer(x, angle) if self.use_film else layer(x)
        for layer in self.uparm: x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


class Model(LoadableModel):
    @store_config_args
    def __init__(self, im_size, architecture='concatenated', use_film=False, int_steps=10, in_channels=2):
        super().__init__()
        self.architecture = architecture
        self.use_film = use_film

        if architecture == 'concatenated':
            self.backbone = ConcatenatedEncoder(im_size, use_film, in_channels=in_channels * 2)
        elif architecture == 'dual':
            self.backbone = DualEncoder(im_size, use_film, in_channels=in_channels)
        elif architecture == 'separate':
            self.backbone = SeparateProjectionVolumeEncoder(im_size, use_film, proj_channels=in_channels)
        elif architecture == 'broadcast':
            self.backbone = BroadcastEncoder(im_size, use_film, proj_channels=in_channels)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        vol_shape = [im_size, im_size, im_size]
        self.integrate = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, source_proj, target_proj, source_vol, angle=None):
        if self.architecture in ['concatenated', 'dual']:
            pos_flow = self.backbone(source_proj, target_proj, angle)
        else:
            pos_flow = self.backbone(target_proj, source_vol, angle)

        B = source_vol.shape[0]
        if pos_flow.shape[0] != B:
            pos_flow = pos_flow.view(B, -1, *pos_flow.shape[1:]).mean(dim=1)

        # Interpolate flow to match source volume resolution if necessary
        if pos_flow.shape[2:] != tuple(source_vol.shape[2:]):
            pos_flow = torch.nn.functional.interpolate(
                pos_flow, size=source_vol.shape[2:], mode="trilinear", align_corners=True
            )

        if self.integrate is not None:
            pos_flow = self.integrate(pos_flow)

        y_source = self.transformer(source_vol, pos_flow)
        return y_source, pos_flow