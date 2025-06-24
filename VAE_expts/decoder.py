from pythae.models.base.base_architecture import BaseDecoder
import torch
import torch.nn as nn
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_architecture import BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class DatasetSpecificDecoder(BaseDecoder):
    """Decoder that adapts architecture based on dataset"""
    def __init__(self, model_config):
        BaseDecoder.__init__(self)
        
        self.dataset_name = getattr(model_config, 'dataset_name', 'dsprites')
        self.output_channels = getattr(model_config, 'input_channels', 1)
        self.output_size = getattr(model_config, 'input_size', 64)
        self.latent_dim = model_config.latent_dim
        self.bottleneck_dim = getattr(model_config, 'bottleneck_dim', 256)
        self.decoder_type = getattr(model_config, 'decoder_type', 'simple')
        
        # Build architecture based on dataset
        if self.dataset_name == "dsprites":
            self._build_dsprites_decoder()
        elif self.dataset_name == "xyobject":
            self._build_xyobject_decoder()
        elif self.dataset_name == "shapes3d":
            self._build_shapes3d_decoder()
        else:
            self._build_xyobject_decoder()  # default
            
    def _build_dsprites_decoder(self):
        """Simple decoder for dSprites"""
        self.initial_size = 4
        
        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, 256 * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        # Simple deconv layers
        self.decoder_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.output_activation = nn.Sigmoid()
    
    def _build_xyobject_decoder(self):
        """Standard decoder for XYObject"""
        self.initial_size = 4
        
        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        # Standard deconv layers with batch norm
        self.decoder_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.output_activation = nn.Sigmoid()
    
    def _build_shapes3d_decoder(self):
        """Advanced decoder for 3D Shapes with final conv layers"""
        self.initial_size = 2
        
        # Initial projection with heavy dropout
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512 * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        # Deconv layers
        self.decoder_layers = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Final conv layers for refinement
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, self.output_channels, kernel_size=3, stride=1, padding=1),
        )
        
        self.output_activation = nn.Sigmoid()
    
    def forward(self, z):
        # Project and reshape
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.initial_size, self.initial_size)
        
        # Decode
        h = self.decoder_layers(h)
        
        # Apply output activation
        reconstruction = self.output_activation(h)
        
        return ModelOutput(
            reconstruction=reconstruction
        )