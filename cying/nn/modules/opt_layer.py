import torch.nn as nn
from .opt_conv import OptConv1d, OptConv2d, OptConv3d
from .opt_spectral_conv import OptSpectralConv1d, OptSpectralConv2d, OptSpectralConv3d


class BaseOptLayer(nn.Module):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_width = hidden_width
        self.spe_opt_size = spe_opt_size
        self.spa_opt_size = spa_opt_size
        self.spectral_opts = self._create_spectral_opts()
        self.spatial_opts = self._create_spatial_opts()
        self.nonlinear_opts = self._create_nonlinear_opts()
        self.linear_skip = self._create_linear_skip()

    def forward(self, input):
        lin_opts = self.spectral_opts(input) + self.spatial_opts(input) + input
        return self.nonlinear_opts(lin_opts) + self.linear_skip(lin_opts)

    def _create_spectral_opts(self):
        raise NotImplementedError

    def _create_spatial_opts(self):
        raise NotImplementedError

    def _create_nonlinear_opts(self):
        raise NotImplementedError
    
    def _create_linear_skip(self):
        raise NotImplementedError


class OptLayer1d(BaseOptLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _create_spectral_opts(self):
        return OptSpectralConv1d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return OptConv1d(
            size=self.size,
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv1d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv1d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv1d(self.in_channels,self.out_channels,1)


class OptLayer2d(BaseOptLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _create_spectral_opts(self):
        return OptSpectralConv2d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return OptConv2d(
            size=self.size,
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv2d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv2d(self.in_channels,self.out_channels,1)


class OptLayer3d(BaseOptLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _create_spectral_opts(self):
        return OptSpectralConv3d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return OptConv3d(
            size=self.size,
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )
    
    def _create_nonlinear_opts(self):
        return nn.Sequential(
            nn.Conv3d(self.in_channels,self.hidden_width,1),
            nn.GELU(),
            nn.Conv3d(self.hidden_width,self.out_channels,1)
        )
    
    def _create_linear_skip(self):
        return nn.Conv3d(self.in_channels,self.out_channels,1)