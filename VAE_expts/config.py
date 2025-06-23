from dataclasses import dataclass
import torch

@dataclass
class DisentangledVAEConfig:
    """Configuration for factorized disentangled VAE"""
    # Dataset specific configurations
    dataset_name: str = "dsprites"  # dsprites, xyobject, shapes3d
    
    # These will be set based on dataset_name
    input_channels: int = 1
    input_size: int = 64
    semantic_dim: int = 10      # content_dim
    attribute_dim: int = 6       # style_dim
    n_semantic_factors: int = 32    
    n_attribute_factors: int = 16   
    
    # Training settings (will be set based on dataset)
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    beta_start: float = 0.0
    beta_end: float = 4.0
    
    # Other loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.001  # This will be annealed
    factorization_weight: float = 0.01
    sparsity_weight: float = 0.001
    orthogonality_weight: float = 0.01
    
    # Architecture details
    bottleneck_dim: int = 256
    encoder_dropout: float = 0.0
    decoder_type: str = "simple"  # simple, standard, advanced
    
    # Dataset specific
    num_classes: int = 0
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def __post_init__(self):
        """Set dataset-specific configurations"""
        if self.dataset_name == "dsprites":
            self.input_channels = 1
            self.semantic_dim = 10  # content
            self.attribute_dim = 6  # style
            self.bottleneck_dim = 256
            self.encoder_dropout = 0.0
            self.decoder_type = "simple"
            self.epochs = 30
            self.beta_end = 4.0
            
        elif self.dataset_name == "xyobject":
            self.input_channels = 3
            self.semantic_dim = 12  # content
            self.attribute_dim = 8   # style
            self.bottleneck_dim = 512
            self.encoder_dropout = 0.2
            self.decoder_type = "standard"
            self.epochs = 40
            self.beta_end = 5.0
            
        elif self.dataset_name == "shapes3d":
            self.input_channels = 3
            self.semantic_dim = 10  # content
            self.attribute_dim = 12  # style
            self.bottleneck_dim = 512
            self.encoder_dropout = 0.3  # heavy dropout
            self.decoder_type = "advanced"
            self.epochs = 50
            self.beta_end = 6.0