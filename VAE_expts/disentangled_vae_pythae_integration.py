import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Pythae imports
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.pipelines.training import TrainingPipeline
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder

# Standard dataset imports
import torchvision
import torchvision.transforms as transforms

try:
    import medmnist
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("MedMNIST not available. Install with: pip install medmnist")

# ============================================================================
# CONFIGURATION - Avoiding "concept" terminology
# ============================================================================

@dataclass
class DisentangledVAEConfig:
    """Configuration for factorized disentangled VAE"""
    # Architecture
    input_channels: int = 3
    input_size: int = 64
    
    # Latent space structure
    semantic_dim: int = 16      # Instead of "invariant"
    attribute_dim: int = 8      # Instead of "specific"
    
    # Interpretable factorization dimensions
    n_semantic_factors: int = 32    # Not "concepts"!
    n_attribute_factors: int = 16   # Just "factors"
    
    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.001
    factorization_weight: float = 0.01
    sparsity_weight: float = 0.001
    orthogonality_weight: float = 0.01
    
    # Dataset specific
    num_classes: int = 7
    dataset_name: str = "mnist"
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ============================================================================
# SPARSE FACTORIZATION LAYERS
# ============================================================================

class SparseFactorization(nn.Module):
    """
    Sparse factorization layer for interpretable representations.
    Not a "concept layer" - just sparse linear transformation.
    """
    def __init__(self, input_dim, n_factors, target_sparsity=0.3):
        super().__init__()
        
        # Linear factorization with sparsity
        self.factorization = nn.Linear(input_dim, n_factors, bias=False)
        
        # Learnable thresholds for sparsity
        self.thresholds = nn.Parameter(torch.zeros(n_factors))
        
        # Target sparsity level
        self.target_sparsity = target_sparsity
        
        # Optional: Non-negative constraints
        self.activation = nn.ReLU()
        
    def forward(self, z):
        # Linear transformation
        factors = self.factorization(z)
        
        # Soft thresholding for sparsity
        factors = self.activation(factors - self.thresholds)
        
        # Top-k sparsity during training
        if self.training:
            factors = self.apply_topk_sparsity(factors)
        
        # Normalize to [0,1] range
        return torch.sigmoid(factors)
    
    def apply_topk_sparsity(self, x):
        """Apply top-k sparsity constraint"""
        k = int(x.shape[1] * (1 - self.target_sparsity))
        if k > 0:
            topk, indices = torch.topk(x, k, dim=1)
            mask = torch.zeros_like(x)
            mask.scatter_(1, indices, 1)
            return x * mask
        return x

class SparseActivation(nn.Module):
    """Learnable sparse activation for factor extraction"""
    def __init__(self, target_sparsity=0.3):
        super().__init__()
        self.target_sparsity = target_sparsity
        
    def forward(self, x):
        # Top-k activation
        k = int(x.shape[1] * (1 - self.target_sparsity))
        if k > 0:
            topk, indices = torch.topk(x, k, dim=1)
            
            # Soft masking
            mask = torch.zeros_like(x)
            mask.scatter_(1, indices, 1)
            
            return x * mask
        return x

# ============================================================================
# FLEXIBLE ENCODER/DECODER FOR DIFFERENT DATASETS
# ============================================================================

class FlexibleDisentangledEncoder(BaseEncoder):
    """Encoder that works for different image sizes and channels with disentanglement"""
    def __init__(self, model_config):
        BaseEncoder.__init__(self)
        
        # Extract config parameters
        self.input_channels = getattr(model_config, 'input_channels', 1)
        self.input_size = getattr(model_config, 'input_size', 28)
        self.semantic_dim = getattr(model_config, 'semantic_dim', 16)
        self.attribute_dim = getattr(model_config, 'attribute_dim', 8)
        
        # Adaptive architecture based on input size
        if self.input_size <= 32:
            hidden_dims = [32, 64, 128]
        elif self.input_size <= 64:
            hidden_dims = [32, 64, 128, 256]
        else:  # 128 or larger
            hidden_dims = [32, 64, 128, 256, 512]
            
        # Shared backbone
        layers = []
        in_channels = self.input_channels
        
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
            
        self.shared_encoder = nn.Sequential(*layers)
        
        # Calculate output size dynamically
        with torch.no_grad():
            test_input = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            test_output = self.shared_encoder(test_input)
            self.flatten_size = test_output.view(1, -1).shape[1]
        
        # Separate heads for semantic and attribute factors
        self.semantic_head = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.semantic_dim * 2)  # mu and logvar
        )
        
        self.attribute_head = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.attribute_dim * 2)  # mu and logvar
        )
        
    def forward(self, x):
        # Shared encoding
        h = self.shared_encoder(x)
        h = h.view(h.size(0), -1)
        
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
            # Store separate components for disentanglement processing
            semantic_mu=semantic_mu,
            semantic_logvar=semantic_logvar,
            attribute_mu=attribute_mu,
            attribute_logvar=attribute_logvar
        )

class FlexibleDisentangledDecoder(BaseDecoder):
    """Decoder that adapts to different output sizes with disentanglement"""
    def __init__(self, model_config):
        BaseDecoder.__init__(self)
        
        # Extract config parameters
        self.output_channels = getattr(model_config, 'input_channels', 1)
        self.output_size = getattr(model_config, 'input_size', 28)
        self.latent_dim = model_config.latent_dim
        self.semantic_dim = getattr(model_config, 'semantic_dim', 16)
        self.attribute_dim = getattr(model_config, 'attribute_dim', 8)
        
        # Adaptive architecture
        if self.output_size <= 32:
            hidden_dims = [128, 64, 32]
            self.initial_size = 4
        elif self.output_size <= 64:
            hidden_dims = [256, 128, 64, 32]
            self.initial_size = 4
        else:  # 128 or larger
            hidden_dims = [512, 256, 128, 64, 32]
            self.initial_size = 4
            
        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[0] * self.initial_size * self.initial_size),
            nn.ReLU()
        )
        
        # Deconv layers
        layers = []
        in_channels = hidden_dims[0]
        
        for h_dim in hidden_dims[1:]:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
            
        # Final layer
        layers.append(
            nn.ConvTranspose2d(hidden_dims[-1], self.output_channels,
                             kernel_size=4, stride=2, padding=1)
        )
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output activation
        self.output_activation = nn.Sigmoid()
        
    def forward(self, z):
        # Project and reshape
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.initial_size, self.initial_size)
        
        # Decode
        h = self.decoder_layers(h)
        
        # Crop or pad to exact size if needed
        if h.shape[-1] != self.output_size:
            h = F.interpolate(h, size=(self.output_size, self.output_size), 
                            mode='bilinear', align_corners=False)
        
        reconstruction = self.output_activation(h)
        
        return ModelOutput(
            reconstruction=reconstruction
        )

# ============================================================================
# DISENTANGLED VAE MODEL WITH PYTHAE INTEGRATION
# ============================================================================

class FactorizedDisentangledVAE(VAE):
    """
    Disentangled VAE with interpretable latent factorization.
    
    Uses sparse factorization layers to create interpretable 
    latent representations without explicit concept bottlenecks.
    """
    
    def __init__(self, model_config, encoder=None, decoder=None, disentangle_config=None):
        # Initialize the base VAE
        super().__init__(model_config, encoder, decoder)
        
        # Disentanglement configuration
        self.disentangle_config = disentangle_config or DisentangledVAEConfig()
        
        # Sparse factorization layers (not concept extraction!)
        self.semantic_factorization = SparseFactorization(
            self.disentangle_config.semantic_dim,
            self.disentangle_config.n_semantic_factors,
            target_sparsity=0.3
        )
        
        self.attribute_factorization = SparseFactorization(
            self.disentangle_config.attribute_dim,
            self.disentangle_config.n_attribute_factors,
            target_sparsity=0.3
        )
        
        # Reconstruction pathways
        self.semantic_reconstruction = nn.Linear(
            self.disentangle_config.n_semantic_factors,
            self.disentangle_config.semantic_dim
        )
        
        self.attribute_reconstruction = nn.Linear(
            self.disentangle_config.n_attribute_factors,
            self.disentangle_config.attribute_dim
        )
        
        # Optional: Auxiliary task heads
        if hasattr(self.disentangle_config, 'num_classes') and self.disentangle_config.num_classes > 0:
            self.auxiliary_classifier = nn.Linear(
                self.disentangle_config.n_semantic_factors,
                self.disentangle_config.num_classes
            )
        else:
            self.auxiliary_classifier = None
    
    def forward(self, inputs, **kwargs):
        """Forward pass with factorization processing"""
        # Handle different input formats
        if isinstance(inputs, dict):
            x = inputs.get('data', inputs)
            labels = inputs.get('label', None)
        else:
            x = inputs
            labels = kwargs.get('labels', None)
            
        if isinstance(x, dict):
            x = x.get('data', list(x.values())[0])
        
        # Ensure proper format
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Encode
        encoder_output = self.encoder(x)
        
        # Get separate semantic and attribute components
        if hasattr(encoder_output, 'semantic_mu'):
            semantic_mu = encoder_output.semantic_mu
            semantic_logvar = encoder_output.semantic_logvar
            attribute_mu = encoder_output.attribute_mu
            attribute_logvar = encoder_output.attribute_logvar
        else:
            # Split the combined latent
            mu = encoder_output.embedding
            logvar = encoder_output.log_covariance
            
            semantic_mu = mu[:, :self.disentangle_config.semantic_dim]
            semantic_logvar = logvar[:, :self.disentangle_config.semantic_dim]
            attribute_mu = mu[:, self.disentangle_config.semantic_dim:]
            attribute_logvar = logvar[:, self.disentangle_config.semantic_dim:]
        
        # Sample latents
        semantic_z = self.reparameterize(semantic_mu, semantic_logvar)
        attribute_z = self.reparameterize(attribute_mu, attribute_logvar)
        
        # Apply sparse factorization
        semantic_factors = self.semantic_factorization(semantic_z)
        attribute_factors = self.attribute_factorization(attribute_z)
        
        # Reconstruct through factors
        semantic_z_recon = self.semantic_reconstruction(semantic_factors)
        attribute_z_recon = self.attribute_reconstruction(attribute_factors)
        
        # Combine for decoding
        z_combined = torch.cat([semantic_z_recon, attribute_z_recon], dim=1)
        
        # Decode
        decoder_output = self.decoder(z_combined)
        
        # Auxiliary classification if available
        classification = None
        if self.auxiliary_classifier is not None:
            classification = self.auxiliary_classifier(semantic_factors)
        
        # Compute loss components
        combined_mu = torch.cat([semantic_mu, attribute_mu], dim=1)
        combined_logvar = torch.cat([semantic_logvar, attribute_logvar], dim=1)
        
        # Standard VAE loss
        loss, recon_loss, kld = self.loss_function(
            decoder_output.reconstruction, x, combined_mu, combined_logvar, z_combined
        )
        
        # Add disentanglement-specific losses
        disentangle_outputs = {
            'semantic_mu': semantic_mu,
            'semantic_logvar': semantic_logvar,
            'attribute_mu': attribute_mu,
            'attribute_logvar': attribute_logvar,
            'semantic_z': semantic_z,
            'attribute_z': attribute_z,
            'semantic_factors': semantic_factors,
            'attribute_factors': attribute_factors,
            'semantic_z_recon': semantic_z_recon,
            'attribute_z_recon': attribute_z_recon,
            'auxiliary_classification': classification,
            'labels': labels
        }
        
        disentangle_losses = self.compute_disentanglement_losses(x, disentangle_outputs)
        
        # Total loss includes disentanglement losses
        total_loss = loss + disentangle_losses['total']
        
        return ModelOutput(
            loss=total_loss,  # Required by Pythae trainer!
            recon_loss=recon_loss,
            reg_loss=kld,
            recon_x=decoder_output.reconstruction,
            z=z_combined,
            mu=combined_mu,
            log_var=combined_logvar,
            # Additional disentanglement-specific outputs
            semantic_mu=semantic_mu,
            semantic_logvar=semantic_logvar,
            attribute_mu=attribute_mu,
            attribute_logvar=attribute_logvar,
            semantic_z=semantic_z,
            attribute_z=attribute_z,
            semantic_factors=semantic_factors,
            attribute_factors=attribute_factors,
            semantic_z_recon=semantic_z_recon,
            attribute_z_recon=attribute_z_recon,
            auxiliary_classification=classification,  # Renamed to avoid conflict
            **disentangle_losses
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def compute_disentanglement_losses(self, x, outputs):
        """
        Compute losses for factorized representation learning.
        No mention of concepts - just factorization and sparsity.
        """
        batch_size = x.size(0)
        device = x.device
        losses = {}
        
        # 1. Factor sparsity with target
        target_sparsity = 0.3
        semantic_activation = outputs['semantic_factors'].mean()
        attribute_activation = outputs['attribute_factors'].mean()
        
        sparsity_loss = (
            (semantic_activation - target_sparsity).pow(2) +
            (attribute_activation - target_sparsity).pow(2)
        )
        losses['factor_sparsity'] = sparsity_loss * self.disentangle_config.sparsity_weight
        
        # 2. Orthogonality between semantic and attribute factors
        semantic_norm = F.normalize(outputs['semantic_z'], p=2, dim=1)
        attribute_norm = F.normalize(outputs['attribute_z'], p=2, dim=1)
        
        # Handle dimension mismatch
        semantic_dim = semantic_norm.size(1)
        attribute_dim = attribute_norm.size(1)
        
        if semantic_dim != attribute_dim:
            min_dim = min(semantic_dim, attribute_dim)
            semantic_proj = semantic_norm[:, :min_dim]
            attribute_proj = attribute_norm[:, :min_dim]
            orthogonality_loss = torch.abs(torch.mean(semantic_proj * attribute_proj))
        else:
            orthogonality_loss = torch.abs(torch.mean(semantic_norm * attribute_norm))
        
        losses['orthogonality'] = orthogonality_loss * self.disentangle_config.orthogonality_weight
        
        # 3. Information bottleneck constraint
        semantic_consistency = F.mse_loss(
            outputs['semantic_z_recon'], outputs['semantic_z'].detach()
        )
        attribute_consistency = F.mse_loss(
            outputs['attribute_z_recon'], outputs['attribute_z'].detach()
        )
        
        losses['information_bottleneck'] = (semantic_consistency + attribute_consistency) * 0.1
        
        # 4. Total Correlation (simplified)
        if batch_size > 1:
            z_samples = torch.cat([outputs['semantic_z'], outputs['attribute_z']], dim=1)
            tc_loss = self.estimate_tc_loss(z_samples)
            losses['tc_loss'] = tc_loss * 0.01
        else:
            losses['tc_loss'] = torch.tensor(0.0, device=device)
        
        # 5. Classification loss (if available)
        if outputs['auxiliary_classification'] is not None and outputs['labels'] is not None:
            losses['classification'] = F.cross_entropy(
                outputs['auxiliary_classification'], outputs['labels']
            ) * 0.1
        else:
            losses['classification'] = torch.tensor(0.0, device=device)
        
        # Total disentanglement loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def estimate_tc_loss(self, z_samples):
        """Simplified Total Correlation estimation"""
        batch_size = z_samples.shape[0]
        
        # Simplified approximation: encourage independence
        # by minimizing correlation between dimensions
        z_centered = z_samples - z_samples.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(z_centered.t(), z_centered) / (batch_size - 1)
        
        # Off-diagonal elements should be small (independence)
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        tc_loss = torch.sum(off_diag.pow(2))
        
        return tc_loss 

# ============================================================================
# DATASET WRAPPERS
# ============================================================================

class MedMNISTWrapper(Dataset):
    """Wrapper for MedMNIST datasets"""
    def __init__(self, dataset_name='pathmnist', split='train', transform=None, size=64):
        if not MEDMNIST_AVAILABLE:
            raise ImportError("MedMNIST not available. Install with: pip install medmnist")
            
        info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        self.dataset = DataClass(split=split, download=True, transform=transform)
        self.num_classes = len(info['label'])
        self.task = info['task']
        self.size = size
        
        # Standard preprocessing
        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
            
        # Ensure proper channel format
        if img.shape[0] == 1 and self.size > 28:
            img = img.repeat(3, 1, 1)  # Convert grayscale to RGB if needed
            
        return {
            'data': img,
            'label': torch.tensor(label).squeeze().long(),
            'index': idx
        }

class MNISTWrapper(Dataset):
    """Wrapper for standard MNIST"""
    def __init__(self, split='train', transform=None, size=28):
        self.size = size
        
        # Load MNIST
        self.dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=(split == 'train'), 
            download=True, 
            transform=transform
        )
        
        self.num_classes = 10
        
        # Standard preprocessing
        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
            
        return {
            'data': img,
            'label': torch.tensor(label).long(),
            'index': idx
        }

def load_dataset(dataset_name, input_size=64, train_size=None, test_size=None):
    """Generic dataset loader"""
    
    if dataset_name == 'mnist':
        train_dataset = MNISTWrapper('train', size=input_size)
        test_dataset = MNISTWrapper('test', size=input_size)
        
        # Convert to numpy for Pythae compatibility
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        
        # Sample if size limits specified
        train_indices = list(range(len(train_dataset)))
        test_indices = list(range(len(test_dataset)))
        
        if train_size:
            train_indices = np.random.choice(train_indices, train_size, replace=False)
        if test_size:
            test_indices = np.random.choice(test_indices, test_size, replace=False)
        
        for idx in train_indices:
            item = train_dataset[idx]
            train_data.append(item['data'].numpy())
            train_labels.append(item['label'].item())
            
        for idx in test_indices:
            item = test_dataset[idx]
            test_data.append(item['data'].numpy())
            test_labels.append(item['label'].item())
        
        return (np.array(train_data), np.array(test_data), 
                np.array(train_labels), np.array(test_labels), 
                train_dataset.num_classes)
    
    elif MEDMNIST_AVAILABLE and dataset_name in medmnist.INFO:
        train_dataset = MedMNISTWrapper(dataset_name, 'train', size=input_size)
        test_dataset = MedMNISTWrapper(dataset_name, 'test', size=input_size)
        
        # Convert to numpy for Pythae compatibility
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        
        # Sample if size limits specified
        train_indices = list(range(len(train_dataset)))
        test_indices = list(range(len(test_dataset)))
        
        if train_size:
            train_indices = np.random.choice(train_indices, min(train_size, len(train_indices)), replace=False)
        if test_size:
            test_indices = np.random.choice(test_indices, min(test_size, len(test_indices)), replace=False)
        
        for idx in train_indices:
            item = train_dataset[idx]
            train_data.append(item['data'].numpy())
            train_labels.append(item['label'].item())
            
        for idx in test_indices:
            item = test_dataset[idx]
            test_data.append(item['data'].numpy())
            test_labels.append(item['label'].item())
        
        return (np.array(train_data), np.array(test_data), 
                np.array(train_labels), np.array(test_labels), 
                train_dataset.num_classes)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported or MedMNIST not available")

# ============================================================================
# CUSTOM TRAINER WITH TENSORBOARD LOGGING
# ============================================================================

class DisentangledVAETrainer(BaseTrainer):
    """Custom trainer with detailed TensorBoard logging for disentangled VAE"""
    
    def __init__(self, model, train_dataset, eval_dataset=None, training_config=None, callbacks=None):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)
        
        # Setup TensorBoard logging
        self.tensorboard_dir = os.path.join(self.training_config.output_dir, 'tensorboard_logs')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Track training step for logging
        self.global_step = 0
        
    def train_step(self, epoch: int):
        """Override train step to add custom logging"""
        # Call parent train step
        epoch_loss = super().train_step(epoch)
        
        # Log disentanglement metrics every epoch
        self.log_disentanglement_metrics(epoch)
        
        # Log sample reconstructions every 10 epochs
        if epoch % 10 == 0:
            self.log_sample_reconstructions(epoch)
            
        # Log factor prototypes every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            self.log_factor_prototypes(epoch)
            
        return epoch_loss
    
    def log_disentanglement_metrics(self, epoch):
        """Log disentanglement-specific metrics"""
        
                 # Get a batch for analysis
        try:
            data_iter = iter(self.train_loader)
            batch = next(data_iter)
            
            if isinstance(batch, dict):
                x = batch['data'][:16].to(self.device)
            else:
                x = batch[0][:16].to(self.device)
            
            # Ensure x is 4D (batch, channels, height, width)
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dimension if missing
            
            self.model.eval()
            with torch.no_grad():
                inputs = {"data": x}
                model_output = self.model(inputs)
                
                # Log factor activation statistics
                if hasattr(model_output, 'semantic_factors'):
                    semantic_factors = model_output.semantic_factors
                    self.writer.add_scalar("factors/semantic_mean", semantic_factors.mean().item(), epoch)
                    self.writer.add_scalar("factors/semantic_std", semantic_factors.std().item(), epoch)
                    self.writer.add_scalar("factors/semantic_sparsity", 
                                         (semantic_factors < 0.1).float().mean().item(), epoch)
                    
                    # Log per-factor activation statistics
                    for i in range(min(8, semantic_factors.shape[1])):
                        factor_vals = semantic_factors[:, i]
                        self.writer.add_scalar(f"factors/semantic_{i}/mean", factor_vals.mean().item(), epoch)
                        self.writer.add_scalar(f"factors/semantic_{i}/activation_rate", 
                                             (factor_vals > 0.5).float().mean().item(), epoch)
                
                if hasattr(model_output, 'attribute_factors'):
                    attribute_factors = model_output.attribute_factors
                    self.writer.add_scalar("factors/attribute_mean", attribute_factors.mean().item(), epoch)
                    self.writer.add_scalar("factors/attribute_std", attribute_factors.std().item(), epoch)
                    self.writer.add_scalar("factors/attribute_sparsity", 
                                         (attribute_factors < 0.1).float().mean().item(), epoch)
                    
                    # Log per-factor activation statistics
                    for i in range(min(8, attribute_factors.shape[1])):
                        factor_vals = attribute_factors[:, i]
                        self.writer.add_scalar(f"factors/attribute_{i}/mean", factor_vals.mean().item(), epoch)
                        self.writer.add_scalar(f"factors/attribute_{i}/activation_rate", 
                                             (factor_vals > 0.5).float().mean().item(), epoch)
                
        except Exception as e:
            print(f"Could not log disentanglement metrics: {e}")
        
        self.model.train()
    
    def _optimizers_step(self, model_output):
        """Override optimizer step to track global step"""
        # Call parent optimizer step
        super()._optimizers_step(model_output)
        
        # Increment global step
        self.global_step += 1
        
        # Log loss components every 100 steps
        if self.global_step % 100 == 0:
            if hasattr(model_output, 'loss'):
                self.writer.add_scalar("loss/total", model_output.loss.item(), self.global_step)
            if hasattr(model_output, 'recon_loss'):
                self.writer.add_scalar("loss/reconstruction", model_output.recon_loss.item(), self.global_step)
            if hasattr(model_output, 'reg_loss'):
                self.writer.add_scalar("loss/kld", model_output.reg_loss.item(), self.global_step)
            
            # Log disentanglement-specific losses if available
            if hasattr(model_output, 'factor_sparsity'):
                self.writer.add_scalar("loss/factor_sparsity", model_output.factor_sparsity.item(), self.global_step)
            if hasattr(model_output, 'orthogonality'):
                self.writer.add_scalar("loss/orthogonality", model_output.orthogonality.item(), self.global_step)
            if hasattr(model_output, 'tc_loss'):
                self.writer.add_scalar("loss/total_correlation", model_output.tc_loss.item(), self.global_step)
            if hasattr(model_output, 'information_bottleneck'):
                self.writer.add_scalar("loss/information_bottleneck", model_output.information_bottleneck.item(), self.global_step)
            if hasattr(model_output, 'classification') and hasattr(model_output, 'labels') and model_output.labels is not None:
                # Only log if we have a valid classification loss scalar
                try:
                    if hasattr(model_output, 'auxiliary_classification') and model_output.auxiliary_classification is not None:
                        # Compute classification loss if not already computed
                        if model_output.auxiliary_classification.dim() > 0 and model_output.labels is not None:
                            class_loss = F.cross_entropy(model_output.auxiliary_classification, model_output.labels)
                            self.writer.add_scalar("loss/classification", class_loss.item(), self.global_step)
                except Exception as e:
                    pass  # Skip logging if there's an issue
    
    def log_sample_reconstructions(self, epoch):
        """Log sample reconstructions to TensorBoard"""
        self.model.eval()
        
        try:
            # Get a small batch from training data
            data_iter = iter(self.train_loader)
            batch = next(data_iter)
            
            # Handle different data formats
            if isinstance(batch, dict):
                x = batch['data'][:8]  # Take first 8 samples
            elif isinstance(batch, (list, tuple)):
                x = batch[0][:8]
            else:
                x = batch[:8]
            
            x = x.to(self.device)
            
            # Ensure x is 4D (batch, channels, height, width)
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dimension if missing
            
            with torch.no_grad():
                # Forward pass
                inputs = {"data": x}
                model_output = self.model(inputs)
                
                # Get reconstructions
                reconstructions = model_output.recon_x
                
                # Create comparison grid
                comparison = torch.cat([x, reconstructions], dim=0)
                
                # Log to TensorBoard
                self.writer.add_images(f"reconstructions/epoch_{epoch}", comparison, epoch, dataformats='NCHW')
                
                # Log factor activations as images if available
                if hasattr(model_output, 'semantic_factors'):
                    semantic_factors = model_output.semantic_factors[:8]
                    # Reshape factors for visualization (as 1D heatmaps)
                    semantic_heatmap = semantic_factors.unsqueeze(1).unsqueeze(3)
                    self.writer.add_images(f"factors/semantic_epoch_{epoch}", semantic_heatmap, epoch, dataformats='NCHW')
                
                if hasattr(model_output, 'attribute_factors'):
                    attribute_factors = model_output.attribute_factors[:8]
                    # Reshape factors for visualization (as 1D heatmaps)
                    attribute_heatmap = attribute_factors.unsqueeze(1).unsqueeze(3)
                    self.writer.add_images(f"factors/attribute_epoch_{epoch}", attribute_heatmap, epoch, dataformats='NCHW')
                    
        except Exception as e:
            print(f"Could not log sample reconstructions: {e}")
        
        self.model.train()
    
    def log_factor_prototypes(self, epoch):
        """Log factor prototypes to TensorBoard"""
        self.model.eval()
        
        try:
            # Get a reasonable sample for prototype finding
            all_semantic_factors = []
            all_attribute_factors = []
            all_images = []
            
            # Collect samples
            sample_count = 0
            max_samples = 500
            
            for batch in self.train_loader:
                if sample_count >= max_samples:
                    break
                    
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                # Ensure x is 4D (batch, channels, height, width)
                if x.dim() == 3:
                    x = x.unsqueeze(1)  # Add channel dimension if missing
                
                with torch.no_grad():
                    inputs = {"data": x}
                    model_output = self.model(inputs)
                    
                    if hasattr(model_output, 'semantic_factors'):
                        all_semantic_factors.append(model_output.semantic_factors.cpu())
                        all_attribute_factors.append(model_output.attribute_factors.cpu())
                        all_images.append(x.cpu())
                        
                        sample_count += x.shape[0]
            
            if all_semantic_factors:
                all_semantic_factors = torch.cat(all_semantic_factors, dim=0)
                all_attribute_factors = torch.cat(all_attribute_factors, dim=0)
                all_images = torch.cat(all_images, dim=0)
                
                # Find prototypes for first few factors
                max_factors_to_log = min(4, all_semantic_factors.shape[1])
                
                for factor_idx in range(max_factors_to_log):
                    # Semantic factor prototypes
                    factor_activations = all_semantic_factors[:, factor_idx]
                    top_indices = torch.argsort(factor_activations, descending=True)[:3]
                    
                    prototype_images = all_images[top_indices]
                    self.writer.add_images(f"prototypes/semantic_{factor_idx}_epoch_{epoch}", 
                                         prototype_images, epoch, dataformats='NCHW')
                
                # Attribute factor prototypes
                max_attr_factors = min(4, all_attribute_factors.shape[1])
                for factor_idx in range(max_attr_factors):
                    factor_activations = all_attribute_factors[:, factor_idx]
                    top_indices = torch.argsort(factor_activations, descending=True)[:3]
                    
                    prototype_images = all_images[top_indices]
                    self.writer.add_images(f"prototypes/attribute_{factor_idx}_epoch_{epoch}", 
                                         prototype_images, epoch, dataformats='NCHW')
                            
        except Exception as e:
            print(f"Could not log factor prototypes: {e}")
        
        self.model.train()
    
    def __del__(self):
        """Clean up TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()

# ============================================================================
# CUSTOM TRAINING PIPELINE
# ============================================================================

class DisentangledVAETrainingPipeline(TrainingPipeline):
    """Custom training pipeline that uses our DisentangledVAETrainer"""
    
    def __call__(self, train_data, eval_data=None):
        """Run training with custom trainer"""
        
        # Create custom trainer
        trainer = DisentangledVAETrainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            training_config=self.training_config
        )
        
        # Run training
        trainer.train()
        
        return trainer

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_disentanglement(model, test_data, test_labels, device='cpu', max_samples=1000):
    """Analyze learned disentangled representations"""
    model.eval()
    
    # Use subset for faster computation
    if len(test_data) > max_samples:
        indices = np.random.choice(len(test_data), max_samples, replace=False)
        test_data = test_data[indices]
        test_labels = test_labels[indices]
    
    # Process in batches
    batch_size = 100
    semantic_factors = []
    attribute_factors = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_data = torch.tensor(test_data[i:i+batch_size], dtype=torch.float32).to(device)
            inputs = {"data": batch_data}
            
            model_output = model(inputs)
            
            if hasattr(model_output, 'semantic_factors'):
                semantic_factors.append(model_output.semantic_factors.cpu().numpy())
            if hasattr(model_output, 'attribute_factors'):
                attribute_factors.append(model_output.attribute_factors.cpu().numpy())
    
    if semantic_factors:
        semantic_factors = np.concatenate(semantic_factors, axis=0)
    if attribute_factors:
        attribute_factors = np.concatenate(attribute_factors, axis=0)
    
    # Compute disentanglement metrics
    results = {}
    
    if len(semantic_factors) > 0:
        results['semantic_sparsity'] = (semantic_factors < 0.1).mean()
        results['semantic_entropy'] = compute_entropy_np(semantic_factors).mean()
        results['semantic_activation_rate'] = (semantic_factors > 0.5).mean()
        
        # Per-factor statistics
        results['semantic_factor_means'] = semantic_factors.mean(axis=0)
        results['semantic_factor_stds'] = semantic_factors.std(axis=0)
    
    if len(attribute_factors) > 0:
        results['attribute_sparsity'] = (attribute_factors < 0.1).mean()
        results['attribute_entropy'] = compute_entropy_np(attribute_factors).mean()
        results['attribute_activation_rate'] = (attribute_factors > 0.5).mean()
        
        # Per-factor statistics
        results['attribute_factor_means'] = attribute_factors.mean(axis=0)
        results['attribute_factor_stds'] = attribute_factors.std(axis=0)
    
    # Mutual information with labels if available
    if len(semantic_factors) > 0 and test_labels is not None:
        results['semantic_label_mi'] = compute_mutual_information_np(semantic_factors, test_labels)
    
    return results

def compute_entropy_np(probs):
    """Compute entropy of probability distributions (numpy version)"""
    eps = 1e-8
    return -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))

def compute_mutual_information_np(factors, labels):
    """Estimate mutual information between factors and labels (numpy version)"""
    mi_scores = []
    
    for i in range(factors.shape[1]):
        factor = factors[:, i]
        
        # Discretize factor
        factor_binary = (factor > 0.5).astype(int)
        
        # Compute MI with label
        unique_labels = np.unique(labels)
        h_factor = compute_entropy_np(factor_binary.mean())
        
        h_factor_given_label = 0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                p_label = mask.mean()
                if mask.sum() > 0:
                    h_conditional = compute_entropy_np(factor_binary[mask].mean())
                    h_factor_given_label += p_label * h_conditional
        
        mi = h_factor - h_factor_given_label
        mi_scores.append(mi)
    
    return mi_scores

def visualize_disentanglement_results(model, test_data, test_labels, device='cpu', save_prefix="disentangled"):
    """Visualize disentanglement results"""
    model.eval()
    
    # Select random samples for visualization
    num_samples = 8
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    samples = torch.tensor(test_data[indices], dtype=torch.float32).to(device)
    sample_labels = test_labels[indices]
    
    with torch.no_grad():
        inputs = {"data": samples}
        model_output = model(inputs)
        
        # Get reconstructions and factor activations
        reconstructions = model_output.recon_x.cpu().numpy()
        
        semantic_factors = model_output.semantic_factors.cpu().numpy() if hasattr(model_output, 'semantic_factors') else None
        attribute_factors = model_output.attribute_factors.cpu().numpy() if hasattr(model_output, 'attribute_factors') else None
        
        # Plot comparison
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 8))
        
        for i in range(num_samples):
            # Original
            if samples.shape[1] == 1:  # Grayscale
                axes[0, i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                axes[0, i].imshow(samples[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f"Original\nLabel: {sample_labels[i]}")
            axes[0, i].axis('off')
            
            # Reconstruction
            if reconstructions.shape[1] == 1:  # Grayscale
                axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
            else:  # RGB
                axes[1, i].imshow(reconstructions[i].transpose(1, 2, 0))
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')
            
            # Factor activations
            if semantic_factors is not None and attribute_factors is not None:
                all_factors = np.concatenate([semantic_factors[i], attribute_factors[i]])
                axes[2, i].bar(range(len(all_factors)), all_factors)
                axes[2, i].set_title(f"Factors\n(S:{len(semantic_factors[i])}, A:{len(attribute_factors[i])})")
                axes[2, i].set_ylim(0, 1)
                
                # Add vertical line to separate semantic and attribute
                axes[2, i].axvline(x=len(semantic_factors[i])-0.5, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_reconstructions.png", dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Configuration
    disentangle_config = DisentangledVAEConfig(
        input_channels=1,  # MNIST is grayscale
        input_size=28,
        semantic_dim=16,
        attribute_dim=8,
        n_semantic_factors=32,
        n_attribute_factors=16,
        num_classes=10,
        batch_size=128,
        epochs=50,
        dataset_name="mnist"
    )
    
    # Device configuration
    device = torch.device(disentangle_config.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {disentangle_config.dataset_name} dataset...")
    train_data, test_data, train_labels, test_labels, num_classes = load_dataset(
        disentangle_config.dataset_name,
        input_size=disentangle_config.input_size,
        train_size=10000,
        test_size=2000
    )
    
    # Update config with actual number of classes
    disentangle_config.num_classes = num_classes
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Configure Pythae trainer
    training_config = BaseTrainerConfig(
        output_dir='disentangled_vae_output',
        num_epochs=disentangle_config.epochs,
        learning_rate=disentangle_config.learning_rate,
        per_device_train_batch_size=disentangle_config.batch_size,
        per_device_eval_batch_size=disentangle_config.batch_size,
        train_dataloader_num_workers=2,
        eval_dataloader_num_workers=2,
        steps_saving=5,
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.01, "betas": (0.9, 0.999)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 3, "factor": 0.5}
    )
    
    # Configure VAE model
    model_config = VAEConfig(
        input_dim=(disentangle_config.input_channels, disentangle_config.input_size, disentangle_config.input_size),
        latent_dim=disentangle_config.semantic_dim + disentangle_config.attribute_dim
    )
    
    # Add disentanglement-specific config
    model_config.input_channels = disentangle_config.input_channels
    model_config.input_size = disentangle_config.input_size
    model_config.semantic_dim = disentangle_config.semantic_dim
    model_config.attribute_dim = disentangle_config.attribute_dim
    
    # Create model
    print("Creating Factorized Disentangled VAE...")
    model = FactorizedDisentangledVAE(
        model_config=model_config,
        encoder=FlexibleDisentangledEncoder(model_config),
        decoder=FlexibleDisentangledDecoder(model_config),
        disentangle_config=disentangle_config
    )
    
    print(f"Model architecture:")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters())} parameters")
    print(f"  Decoder: {sum(p.numel() for p in model.decoder.parameters())} parameters")
    print(f"  Factorization layers: {sum(p.numel() for p in model.semantic_factorization.parameters()) + sum(p.numel() for p in model.attribute_factorization.parameters())} parameters")
    
    # Create training pipeline
    print("Setting up training pipeline with TensorBoard logging...")
    pipeline = DisentangledVAETrainingPipeline(
        training_config=training_config,
        model=model
    )
    
    # Train the model
    print("Starting training...")
    pipeline(
        train_data=train_data,
        eval_data=test_data
    )
    
    # Save the model
    print("Saving trained model...")
    model.save('disentangled_vae_output/manual_save')
    
    # Use the trained model directly (training completed successfully)
    print("Using the successfully trained model for analysis...")
    trained_model = model.to(device)
    
    # Analyze disentanglement
    print("Analyzing disentanglement...")
    results = analyze_disentanglement(trained_model, test_data, test_labels, device=device)
    
    print("\nDisentanglement Analysis Results:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 0:
                print(f"{key}: mean={np.mean(value):.3f}, std={np.std(value):.3f}")
        else:
            print(f"{key}: {value:.3f}")
    
    # Visualize results
    print("Visualizing disentanglement results...")
    visualize_disentanglement_results(trained_model, test_data, test_labels, device=device)
    
    print("\nTraining completed!")
    print("TensorBoard logs available at: disentangled_vae_output/tensorboard_logs/")
    print("Run 'tensorboard --logdir=disentangled_vae_output/tensorboard_logs/' to view training metrics.")

if __name__ == "__main__":
    main() 