import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from transformers import SegformerForSemanticSegmentation


DECODER_ARCHITECTURE = "fig1"


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x))
        return x * weight


class DFESA(nn.Module):
    def __init__(self, base_ratio: float = 0.1, fusion_lambda: float = 0.6, eps: float = 1e-5):
        super().__init__()
        self.base_ratio = base_ratio
        self.fusion_lambda = fusion_lambda
        self.eps = eps
        self.activation = nn.Sigmoid()

    def _edge_attention(self, x: torch.Tensor) -> torch.Tensor:
        centered_sq = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        return centered_sq / (variance + self.eps)

    def _structure_attention(self, x: torch.Tensor) -> torch.Tensor:
        energy = x.pow(2)
        structure_mean = x.mean(dim=(2, 3), keepdim=True)
        structure_var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        return self.activation((energy - structure_mean) / (structure_var + self.eps))

    def _create_low_freq_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask_ratio = self.base_ratio * min(h, w) / max(h, w)
        mask_ratio = max(mask_ratio, 1e-4)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        mask = torch.exp(-(yy.pow(2) + xx.pow(2)) / (2.0 * mask_ratio ** 2))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        low_mask = self._create_low_freq_mask(h, w, x.device, x.real.dtype)
        high_mask = 1.0 - low_mask

        low_freq = x_freq * low_mask
        high_freq = x_freq * high_mask

        low_spatial = torch.abs(fft.ifftn(fft.ifftshift(low_freq, dim=(-2, -1)), dim=(-2, -1)))
        high_spatial = torch.abs(fft.ifftn(fft.ifftshift(high_freq, dim=(-2, -1)), dim=(-2, -1)))

        a_s = self._structure_attention(low_spatial)
        a_e = self._edge_attention(high_spatial)
        attention = self.activation(self.fusion_lambda * a_e + (1.0 - self.fusion_lambda) * a_s)
        return x + attention * x


class DilatedBranch(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, dilation_rate: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            SEBlock(mid_channels),
            ConvBNReLU(mid_channels, mid_channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScaleAwareModule(nn.Module):
    def __init__(self, in_channels: int, inter_channels: int):
        super().__init__()
        self.branch_r1 = DilatedBranch(in_channels, inter_channels, dilation_rate=1)
        self.branch_r3 = DilatedBranch(in_channels, inter_channels, dilation_rate=3)
        self.branch_r5 = DilatedBranch(in_channels, inter_channels, dilation_rate=5)
        self.attention_gen = nn.Sequential(
            ConvBNReLU(inter_channels, inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(inter_channels, 3, kernel_size=3, padding=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_a = self.branch_r1(x)
        f_b = self.branch_r3(x)
        f_c = self.branch_r5(x)
        f_sum = f_a + f_b + f_c
        att = self.attention_gen(f_sum)
        return att[:, 0:1] * f_a + att[:, 1:2] * f_b + att[:, 2:3] * f_c


class SCPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inter_channels = max(out_channels // 4, 8)
        self.local_branch = ConvBNReLU(in_channels, inter_channels, kernel_size=1, padding=0)
        self.scale_aware = ScaleAwareModule(in_channels, inter_channels)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feat_local = self.local_branch(x)
        feat_scale = self.scale_aware(x)
        feat_global = self.global_branch(x)
        feat_global = F.interpolate(feat_global, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([feat_local, feat_scale, feat_global], dim=1)
        return self.final_conv(out)


class CustomSegformerDecodeHead(nn.Module):
    """Fig. 1: shallow DFESA -> per-level MLP/resize -> concat -> SCPP -> MLP.

    The four MiT features have strides 4, 8, 16 and 32. Each feature is
    projected to C channels, resized to the stride-4 grid, and concatenated
    in deepest-to-shallowest order. SCPP maps 4C channels back to C, and a
    point-wise affine classifier produces N_cls logits at stride 4.
    """

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.hidden_sizes = config.hidden_sizes
        if len(self.hidden_sizes) != 4:
            raise ValueError("The Fig. 1 decode head requires four encoder feature levels.")
        architecture = getattr(config, "custom_decoder_architecture", DECODER_ARCHITECTURE)
        if architecture != DECODER_ARCHITECTURE:
            raise ValueError(f"Unsupported decode-head architecture: {architecture}")
        config.custom_decoder_architecture = DECODER_ARCHITECTURE
        self.decoder_hidden_size = getattr(config, "custom_decoder_dim", getattr(config, "decoder_hidden_size", 256))
        self.dropout_p = getattr(config, "classifier_dropout_prob", 0.1)
        self.dfesa_base_ratio = getattr(config, "dfesa_base_ratio", 0.1)
        self.dfesa_fusion_lambda = getattr(config, "dfesa_fusion_lambda", 0.6)
        self.use_dfesa = getattr(config, "use_dfesa", True)
        self.use_scpp = getattr(config, "use_scpp", True)

        self.proj_layers = nn.ModuleList([
            nn.Linear(in_channels, self.decoder_hidden_size)
            for in_channels in self.hidden_sizes
        ])

        self.dfesa = DFESA(self.dfesa_base_ratio, self.dfesa_fusion_lambda) if self.use_dfesa else None
        fused_channels = self.decoder_hidden_size * len(self.hidden_sizes)
        self.scpp = SCPP(fused_channels, self.decoder_hidden_size) if self.use_scpp else None
        # With SCPP disabled for an ablation, apply the classifier to 4C directly.
        classifier_channels = self.decoder_hidden_size if self.use_scpp else fused_channels
        self.dropout = nn.Dropout(self.dropout_p)
        # A 1x1 convolution is the same affine mapping as a per-pixel linear MLP.
        self.classifier = nn.Conv2d(classifier_channels, self.num_labels, kernel_size=1)

    def _check_feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Feature level {idx} must have shape [B,C,H,W]. Got {tuple(x.shape)}")
        return x

    def forward(self, encoder_hidden_states):
        if len(encoder_hidden_states) != 4:
            raise ValueError(f"CustomSegformerDecodeHead expects 4 feature levels. Received {len(encoder_hidden_states)}.")

        features = [self._check_feature(x, i + 1) for i, x in enumerate(encoder_hidden_states)]
        if self.dfesa is not None:
            features[0] = self.dfesa(features[0])
        target_size = features[0].shape[-2:]
        projected = []
        for i, (feature, projection) in enumerate(zip(features, self.proj_layers)):
            batch, channels, height, width = feature.shape
            if channels != self.hidden_sizes[i]:
                raise ValueError(f"Feature level {i + 1} expects {self.hidden_sizes[i]} channels. Got {channels}.")
            feature = projection(feature.flatten(2).transpose(1, 2))
            feature = feature.transpose(1, 2).reshape(batch, self.decoder_hidden_size, height, width)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        fused = torch.cat(projected[::-1], dim=1)
        if self.scpp is not None:
            fused = self.scpp(fused)
        return self.classifier(self.dropout(fused))


def get_custom_segformer(
    model_checkpoint,
    id2label,
    label2id,
    custom_decoder_dim=None,
    dfesa_base_ratio: float = 0.1,
    dfesa_fusion_lambda: float = 0.6,
    use_dfesa: bool = True,
    use_scpp: bool = True,
):
    print("Loading the base model and replacing the decode head...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    config = model.config
    if custom_decoder_dim is not None:
        config.custom_decoder_dim = custom_decoder_dim
    config.dfesa_base_ratio = dfesa_base_ratio
    config.dfesa_fusion_lambda = dfesa_fusion_lambda
    config.use_dfesa = bool(use_dfesa)
    config.use_scpp = bool(use_scpp)
    config.custom_decoder_architecture = DECODER_ARCHITECTURE

    model.decode_head = CustomSegformerDecodeHead(config)

    return model
