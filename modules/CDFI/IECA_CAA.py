import torch
import torch.nn as nn
import torch.nn.functional as F

from .ECA import ECA          # ch -> ECA(channel=k_size)
from .CAA import CAA          # ch -> CAA(ch, h_kernel_size=11, v_kernel_size=11)

class CDFI(nn.Module):
    """
       Cross-Dimensional Feature Interaction (CDFI)

       Interactive ECA–CAA Fusion Module.
       Bidirectional guidance mechanism:
           - Channel attention guides spatial attention
           - Spatial attention feeds back to channel attention
       The final output is obtained via dual-branch weighted fusion with optional residual connection.

       Args:
           ch (int):
               Number of input channels (automatically inferred by the parser;
               can be left empty [] in YAML configuration).
           k_size (int):
               Kernel size of the 1D convolution in ECA.
               It is recommended to use an odd value.
               If adaptive kernel sizing is required, it can be externally determined based on `ch`.
           h_kernel_size (int):
               Horizontal stripe convolution kernel size in CAA.
           v_kernel_size (int):
               Vertical stripe convolution kernel size in CAA.
           use_residual (bool):
               Whether to add residual connection from the input feature.
       """
    def __init__(self, ch, k_size=3, h_kernel_size=11, v_kernel_size=11, use_residual=True):
        super().__init__()
        self.ch = ch
        self.use_residual = use_residual

        # Sub-modules: constructed using the original implementations
        # to keep interfaces and behaviors consistent
        self.eca = ECA(channel=ch, k_size=k_size)
        self.caa = CAA(ch=ch, h_kernel_size=h_kernel_size, v_kernel_size=v_kernel_size)

        # To explicitly obtain attention weights ("gates"),
        # we replicate the internal weight-generation paths of ECA and CAA here.
        # This avoids modifying the original ECA/CAA files and preserves compatibility.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)             # for ECA weight
        self.sigmoid = nn.Sigmoid()

        # Learnable scalar coefficients for bidirectional fusion
        # Initialized to 0.5 to balance channel and spatial branches
        self.gamma_c = nn.Parameter(torch.tensor(0.5))
        self.gamma_s = nn.Parameter(torch.tensor(0.5))

        # Reuse internal layers of CAA to generate spatial attention A_s
        # (No new parameters are introduced; layers are directly accessed from the CAA module)
        # self.caa.avg_pool, self.caa.conv1, self.caa.h_conv,
        # self.caa.v_conv, self.caa.conv2, self.caa.act

    def channel_gate(self, x):
        """
                Channel attention generation (replicating ECA behavior):
                Global Average Pooling -> 1D Convolution -> Sigmoid
        """
        y = self.avg_pool(x)                                 # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)                  # B,C,1 -> B,1,C
        y = self.eca.conv(y)                                 # 1D conv on channel
        y = y.transpose(-1, -2).unsqueeze(-1)                # back to B,C,1,1
        w_c = self.sigmoid(y)                                # Channel gate
        return w_c

    def spatial_gate(self, x):
        """
               Spatial attention generation (replicating CAA behavior):
               Average Pooling -> 1×1 Conv -> (1×k & k×1) Stripe Convs -> 1×1 Conv -> Activation
        """
        a = self.caa.avg_pool(x)
        a = self.caa.conv1(a)
        a = self.caa.h_conv(a)
        a = self.caa.v_conv(a)
        a = self.caa.conv2(a)
        A_s = self.caa.act(a)                                 # Spatial gate (B, C, H, W)
        return A_s

    def forward(self, x):
        # Original channel and spatial attention gates
        w_c = self.channel_gate(x)                           # B,C,1,1
        A_s = self.spatial_gate(x)                           # B,C,H,W

        # Bidirectional interaction
        # Channel-to-spatial guidance
        A_hat = A_s * w_c                                    # # Broadcasted multiplication
        g = F.adaptive_avg_pool2d(A_hat, 1)
        w_hat = w_c * g

        # Dual-branch reweighting and fusion
        # Optional residual connection
        y_c = w_hat * x                                      # Channel-refined features
        y_s = A_hat * x                                       # Spatial-refined features
        y = self.gamma_c * y_c + self.gamma_s * y_s
        if self.use_residual:
            y = y + x
        return y
