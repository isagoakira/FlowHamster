"""
Mamba-2 State Space Model Implementation

Based on paper:
"Mamba-2: State Space Models with Structured State Spaces"
https://arxiv.org/abs/2405.xxxxx

Architecture:
1. Input projection: x → z, dt, B, C
2. Conv1D with SiLU activation
3. Selective SSM (parallel associative scan)
4. Gating: y * sigmoid(z)
5. Output projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MambaConfig:
    """Mamba configuration"""
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,        # N: state dimension
        d_conv: int = 4,          # Conv kernel size
        expand: int = 2,          # Expansion factor
        dt_rank: int = "auto",    # Rank of dt projection
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        if dt_rank == "auto":
            self.dt_rank = max(math.ceil(d_model / 16), 1)
        else:
            self.dt_rank = dt_rank
        self.bias = bias
        self.conv_bias = conv_bias
        self.dropout = dropout


class MambaBlock(nn.Module):
    """
    Single Mamba-2 block.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dt_rank: Rank for dt projection
        bias: Use bias in linear layers
        conv_bias: Use bias in conv
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: "int | str" = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            conv_bias=conv_bias,
            dropout=dropout,
        )

        d_inner = self.config.d_inner
        dt_rank = self.config.dt_rank

        # Input projection: splits into x and z
        # x goes through conv -> ssm -> gate
        # z goes through gate (sigmoid)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        # Convolutional layer (with causal padding)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # Causal padding
            bias=conv_bias
        )

        # Activation
        self.act = nn.SiLU()

        # SSM parameters
        # dt projection: d_inner -> dt_rank
        self.dt_proj = nn.Linear(d_inner, dt_rank, bias=True)

        # Initialize dt so that A has desired spectral radius
        dt_init_std = 1.0 / math.sqrt(dt_rank)
        nn.init.normal_(self.dt_proj.weight, mean=0, std=dt_init_std)
        nn.init.zeros_(self.dt_proj.bias)

        # State matrix A (in log space for stability)
        # Shape: (d_inner, d_state) - each channel has its own A row
        # We use a diagonal matrix where A[i,j] = lambda_j if i==j
        # For efficiency, we store as (d_inner, d_state) and use broadcasting
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, d_state).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D (skip connection) parameter
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Selectivity projection: d_inner -> dt_rank + 2*d_state
        # (dt and B, C are derived from this)
        self.select_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

    def selective_scan(self, x, dt, A, B, C, D):
        """
        Selective scan (SSM) - simplified recurrence version.

        Args:
            x: (batch, seq_len, d_inner)
            dt: (batch, seq_len, dt_rank) -> projected to (batch, seq_len, d_inner)
            A: (d_inner, d_state)
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)
            D: (d_inner,)

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Project dt to d_inner dimension
        # Simple: repeat or pad to match d_inner
        if d_inner % dt.shape[-1] == 0:
            dt = dt.repeat(1, 1, d_inner // dt.shape[-1])
        else:
            # Linear projection to d_inner
            dt_proj = torch.zeros(d_inner, dt.shape[-1], device=dt.device, dtype=dt.dtype)
            nn.init.xavier_uniform_(dt_proj)
            dt = torch.matmul(dt, dt_proj.transpose(0, 1))

        # Discretize: softplus ensures dt > 0
        dt = F.softplus(dt)

        # Discretize A: A_bar = exp(A * dt)
        # A: (d_inner, d_state), dt: (batch, seq_len, d_inner)
        # Compute element-wise: exp(A * dt_expanded)
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)  # (batch, seq_len, d_inner, d_state)
        A_bar = torch.exp(A_expanded * dt_expanded)  # (batch, seq_len, d_inner, d_state)

        # B_bar = B.unsqueeze(2) * A_bar  # (batch, seq_len, d_inner, d_state)
        B_unsqueezed = B.unsqueeze(2)  # (batch, seq_len, 1, d_state)
        B_bar = B_unsqueezed * A_bar  # (batch, seq_len, d_inner, d_state)

        # Selective scan via recurrence
        y = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            # h_t = A_bar[t] * h_{t-1} + B_bar[t] * x[t]
            A_t = A_bar[:, t]  # (batch, d_inner, d_state)
            B_t = B_bar[:, t]  # (batch, d_inner, d_state)
            x_t = x[:, t, :]  # (batch, d_inner)

            # Element-wise multiplication and sum over d_state
            # h_new = A_t * h + B_t * x_t (element-wise across d_state, then sum over d_state for each channel)
            h = A_t * h + B_t * x_t.unsqueeze(-1)  # (batch, d_inner, d_state)

            # y_t = C[t] @ h_t (sum over d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # (batch, d_inner)
            y[:, t] = y_t

        # Add skip connection: y = y + x * D
        y = y + x * D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, d_model)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        batch, seq_len, d_model = x.shape
        config = self.config

        # Input projection: x -> (z, dt, B, C)
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)

        # Split into x_proj and z
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Conv1D on x_proj (transpose for conv: batch, channels, length)
        x_conv = self.conv1d(x_proj.transpose(1, 2))  # (batch, d_inner, seq_len + d_conv - 1)
        x_conv = x_conv[:, :, :seq_len].transpose(1, 2)  # (batch, seq_len, d_inner)

        # Activation
        x_conv = self.act(x_conv)

        # Selective projection: x_conv -> (dt, B, C)
        select = self.select_proj(x_conv)  # (batch, seq_len, dt_rank + 2*d_state)

        # Split into dt, B, C
        dt = select[:, :, :config.dt_rank]  # (batch, seq_len, dt_rank)
        B_input = select[:, :, config.dt_rank:config.dt_rank + config.d_state]  # (batch, seq_len, d_state)
        C_input = select[:, :, config.dt_rank + config.d_state:]  # (batch, seq_len, d_state)

        # B and C projections
        B = B_input  # (batch, seq_len, d_state)
        C = C_input  # (batch, seq_len, d_state)

        # SSM forward
        y = self.selective_scan(x_conv, dt, self.A_log.exp(), B, C, self.D)

        # Gating
        y = y * torch.sigmoid(z)

        # Output projection
        output = self.out_proj(y)

        output = self.dropout(output)

        # If input was 2D, squeeze back
        if x.dim() == 2:
            output = output.squeeze(1)

        return output


class Mamba(nn.Module):
    """
    Mamba Model - stacks multiple MambaBlocks with residual connections

    Args:
        d_model: Model dimension
        n_layers: Number of Mamba layers
        d_state: SSM state dimension
        d_conv: Conv kernel size
        expand: Expansion factor
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: "int | str" = "auto",
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x) + x  # Residual connection

        x = self.norm(x)

        return x


def create_mamba(
    d_model: int = 512,
    n_layers: int = 1,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.0,
) -> Mamba:
    """Factory function to create a Mamba model."""
    return Mamba(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
    )
