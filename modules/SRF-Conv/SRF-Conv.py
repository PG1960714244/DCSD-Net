import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SRFConv(nn.Module):
    """
      Split-Reconstruction Fusion Convolution (SRF-Conv)

      SRFConv = (space-to-depth + depthwise conv + grouped 1×1 conv) × PConv (partial 3×3)

      - downsample=True:
          Perform 2× downsampling via space-to-depth (S2D),
          which is equivalent to stride=2 but preserves more spatial information.
      - downsample=False:
          Keep spatial resolution unchanged, equivalent to applying SRF-Conv
          at the original feature resolution (e.g., 256-channel PConv stage).

      Ablation mode description (controlled by class attribute ABLATION_MODE):
          1) "s2d_only":
              Only apply space-to-depth (optional mix), skip PConv and DWConv,
              and retain only the final grouped 1×1 fusion.
          2) "s2d_mix":
              Apply space-to-depth followed by quadrant mixing,
              still without PConv and DWConv.
          3) "s2d_pconv":
              Apply space-to-depth + quadrant mixing + PConv;
              only a subset of channels is processed by 3×3 convolution,
              while the remaining channels bypass convolution.
          4) "s2d_p_dw":
              Apply space-to-depth + PConv + DWConv;
              both partial 3×3 and depthwise 3×3 are enabled,
              but quadrant mixing is disabled to isolate their effects.
          5) "full":
              Full SRF-Conv structure:
              space-to-depth + quadrant mixing + PConv + DWConv.
      """
    default_act = nn.SiLU()

    # ===== Set ablation mode here when conducting ablation studies =====
    # Options: "s2d_only", "s2d_mix", "s2d_pconv", "s2d_p_dw", "full"
    ABLATION_MODE = "s2d_p_dw"
    # ========================================================

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True,
                 pw_groups: int = 8,         # Number of groups in 1×1 convolution (larger → fewer parameters)
                 n_div: int = 8,             # Channel division ratio for partial 3×3 convolution
                 downsample: bool = False,   # Enable internal downsampling via S2D (replacing SCDown)
                 mix_quadrant: bool = True,  # Enable lightweight cross-quadrant mixing after S2D
                 affine_bn: bool = False):   # Disable BN affine parameters to further reduce parameters
        super().__init__()
        assert s == 1, "SRFConv: set s=1; use downsample=True to enable internal S2D downsampling"

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Preserve original flags (used in full mode)
        self.downsample = downsample
        self.mix_quadrant = mix_quadrant

        # ------- Ablation mode configuration -------
        mode = SRF-Conv.ABLATION_MODE
        self.mode = mode

        # Enable or disable each substructure according to ablation mode
        if mode == "s2d_only":
            self.use_s2d = downsample
            self.use_mix = False
            self.use_p3 = False
            self.use_dw = False
        elif mode == "s2d_mix":
            self.use_s2d = downsample
            self.use_mix = True
            self.use_p3 = False
            self.use_dw = False
        elif mode == "s2d_pconv":
            self.use_s2d = downsample
            self.use_mix = True
            self.use_p3 = True
            self.use_dw = False
        elif mode == "s2d_p_dw":
            self.use_s2d = downsample
            self.use_mix = False  # Disable mixing to highlight the role of PConv + DWConv
            self.use_p3 = True
            self.use_dw = True
        else:  # "full" or any other string defaults to full configuration
            self.use_s2d = downsample
            self.use_mix = mix_quadrant
            self.use_p3 = True
            self.use_dw = True
        # ---------------------------

        in_ch = 4 * c1 if downsample else c1
        self.dim_conv3 = max(1, in_ch // n_div)      # Channels processed by standard 3×3 convolution
        self.dim_rest = in_ch - self.dim_conv3       # Remaining channels

        # PConv branch: apply standard 3×3 convolution to a small subset of channels
        self.p3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, autopad(k, p, d),
                            groups=1, dilation=d, bias=False)

        # DWConv branch: apply depthwise 3×3 convolution to remaining channels (very low cost)
        self.dw = nn.Conv2d(self.dim_rest, self.dim_rest, k, 1, autopad(k, p, d),
                            groups=self.dim_rest if self.dim_rest > 0 else 1, bias=False)

        # Normalization layers
        self.bn_p3 = nn.BatchNorm2d(self.dim_conv3, affine=affine_bn)
        self.bn_dw = nn.BatchNorm2d(self.dim_rest, affine=affine_bn)

        # Final fusion via grouped 1×1 convolution
        self.pw = nn.Conv2d(in_ch, c2, 1, 1, groups=max(1, pw_groups), bias=False)
        self.bn_pw = nn.BatchNorm2d(c2, affine=affine_bn)

    @staticmethod
    def s2d(x):
        # Space-to-depth operation for 2× downsampling.
        return torch.cat([x[..., ::2, ::2], x[..., ::2, 1::2],
                          x[..., 1::2, ::2], x[..., 1::2, 1::2]], 1)


    def forward(self, x):
        if self.use_s2d:
            x = self.s2d(x)
            if self.use_mix:
                B, C, H, W = x.shape
                x = x.view(B, 4, C // 4, H, W).transpose(1, 2).contiguous().view(B, C, H, W)

        if self.mode in ["s2d_only", "s2d_mix"]:
            x = self.act(self.bn_pw(self.pw(x)))
            return x

        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_rest], dim=1)

        if self.use_p3:
            x1 = self.act(self.bn_p3(self.p3(x1)))

        if self.dim_rest > 0:
            if self.use_dw:
                x2 = self.act(self.bn_dw(self.dw(x2)))
            x = torch.cat([x1, x2], 1)
        else:
            x = x1

        x = self.act(self.bn_pw(self.pw(x)))
        return x
