from pythae.models.base.base_architecture import BaseEncoder
import torch
import torch.nn as nn
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_architecture import BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class DatasetSpecificEncoder(BaseEncoder):
    """Encoder that adapts architecture based on dataset"""
    def __init__(self, model_config):
        BaseEncoder.__init__(self)
        
        self.dataset_name = getattr(model_config, 'dataset_name', 'dsprites')
        self.input_channels = getattr(model_config, 'input_channels', 1)
        self.input_size = getattr(model_config, 'input_size', 64)
        self.semantic_dim = getattr(model_config, 'semantic_dim', 10)
        self.attribute_dim = getattr(model_config, 'attribute_dim', 6)
        self.bottleneck_dim = getattr(model_config, 'bottleneck_dim', 256)
        self.encoder_dropout = getattr(model_config, 'encoder_dropout', 0.0)
        
        # Build architecture based on dataset
        if self.dataset_name == "dsprites":
            self._build_dsprites_encoder()
        elif self.dataset_name == "xyobject":
            self._build_xyobject_encoder()
        elif self.dataset_name == "shapes3d":
            self._build_shapes3d_encoder()
        else:
            self._build_xyobject_encoder()  # default
            
        # Calculate flattened size
        with torch.no_grad():
            test_input = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            test_output = self.shared_encoder(test_input)
            self.flatten_size = test_output.view(1, -1).shape[1]
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flatten_size, self.bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(self.encoder_dropout) if self.encoder_dropout > 0 else nn.Identity()
        )
        
        # Separate heads for semantic and attribute factors
        self.semantic_head = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.semantic_dim * 2),  # mu and logvar
        )
        
        self.attribute_head = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.attribute_dim * 2),  # mu and logvar
        )
    
    def _build_dsprites_encoder(self):
        """Simple encoder for dSprites (4 conv layers, no dropout)"""
        self.shared_encoder = nn.Sequential(
            # Conv1: 64x64 -> 32x32
            nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
    
    def _build_xyobject_encoder(self):
        """Standard encoder for XYObject (4 conv layers + standard dropout)"""
        self.shared_encoder = nn.Sequential(
            # Conv1: 64x64 -> 32x32
            nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Conv2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Conv3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Conv4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten()
        )
    
    def _build_shapes3d_encoder(self):
        """Advanced encoder for 3D Shapes (6 conv layers, large initial kernel, heavy dropout)"""
        self.shared_encoder = nn.Sequential(
            # Initial layer with large kernel
            nn.Conv2d(self.input_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Dropout2d(0.1),
            
            # Main conv layers
            # Conv1: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Conv2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            
            # Conv3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            
            # Conv4: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Flatten()
        )
    
    def forward(self, x):
        # Shared encoding
        h = self.shared_encoder(x)
        h = self.bottleneck(h)
        
        # Semantic parameters
        semantic_params = self.semantic_head(h)
        semantic_mu = semantic_params[:, :self.semantic_dim]
        semantic_logvar = semantic_params[:, self.semantic_dim:]
        
        # Attribute parameters  
        attribute_params = self.attribute_head(h)
        attribute_mu = attribute_params[:, :self.attribute_dim]
        attribute_logvar = attribute_params[:, self.attribute_dim:]
        
        # For Pythae compatibility, combine latents
        combined_mu = torch.cat([semantic_mu, attribute_mu], dim=1)
        combined_logvar = torch.cat([semantic_logvar, attribute_logvar], dim=1)
        
        return ModelOutput(
            embedding=combined_mu,
            log_covariance=combined_logvar,
            semantic_mu=semantic_mu,
            semantic_logvar=semantic_logvar,
            attribute_mu=attribute_mu,
            attribute_logvar=attribute_logvar
        )