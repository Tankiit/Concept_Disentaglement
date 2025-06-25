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
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Force stdout/stderr to be unbuffered for real-time output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
# BALANCED WEIGHT STRATEGY FOR PREVENTING POSTERIOR COLLAPSE
# ============================================================================

@dataclass
class BalancedWeightConfig:
    """Configuration for balanced weight strategy"""
    # KL Annealing
    kl_annealing_type: str = "cyclical"  # "linear", "sigmoid", "cyclical"
    kl_start_weight: float = 0.0
    kl_end_weight: float = 0.001
    kl_annealing_epochs: int = 30
    kl_cycle_length: int = 10  # For cyclical annealing
    
    # Free Bits
    use_free_bits: bool = True
    free_bits_lambda: float = 2.0  # Minimum bits per dimension
    
    # Capacity Constraint
    use_capacity_constraint: bool = True
    capacity_min: float = 0.0
    capacity_max: float = 25.0  # Maximum capacity in nats
    capacity_num_iters: int = 25000  # Steps to reach max capacity
    
    # Dynamic Balancing
    use_dynamic_balancing: bool = True
    target_recon_kl_ratio: float = 100.0  # Target reconstruction/KL ratio
    balance_update_freq: int = 100  # Update weights every N steps
    
    # Loss Monitoring
    monitor_collapse: bool = True
    kl_threshold: float = 0.1  # Below this indicates collapse
    inactive_dims_threshold: float = 0.5  # Fraction of dims allowed to be inactive

class BalancedWeightScheduler:
    """Scheduler for balanced weight strategy to prevent posterior collapse"""
    
    def __init__(self, config: BalancedWeightConfig, model_config):
        self.config = config
        self.model_config = model_config
        self.current_step = 0
        self.current_epoch = 0
        
        # Track loss history for dynamic balancing
        self.loss_history = {
            'reconstruction': [],
            'kl': [],
            'kl_per_dim': {},  # Track per-dimension KL
            'active_dims': []
        }
        
        # Current weights
        self.current_kl_weight = config.kl_start_weight
        self.current_capacity = config.capacity_min
        
    def get_kl_weight(self, epoch: int, step: int) -> float:
        """Get current KL weight based on annealing strategy"""
        self.current_epoch = epoch
        self.current_step = step
        
        if self.config.kl_annealing_type == "linear":
            # Linear annealing
            progress = min(epoch / self.config.kl_annealing_epochs, 1.0)
            weight = self.config.kl_start_weight + \
                    (self.config.kl_end_weight - self.config.kl_start_weight) * progress
                    
        elif self.config.kl_annealing_type == "sigmoid":
            # Sigmoid annealing (smoother transition)
            progress = min(epoch / self.config.kl_annealing_epochs, 1.0)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            weight = self.config.kl_start_weight + \
                    (self.config.kl_end_weight - self.config.kl_start_weight) * sigmoid_progress
                    
        elif self.config.kl_annealing_type == "cyclical":
            # Cyclical annealing (helps escape local minima)
            cycle = epoch // self.config.kl_cycle_length
            cycle_progress = (epoch % self.config.kl_cycle_length) / self.config.kl_cycle_length
            
            # Each cycle gets progressively higher minimum
            cycle_min = self.config.kl_start_weight + \
                       (self.config.kl_end_weight - self.config.kl_start_weight) * \
                       (cycle / (self.config.kl_annealing_epochs / self.config.kl_cycle_length))
            
            weight = cycle_min + (self.config.kl_end_weight - cycle_min) * cycle_progress
        else:
            weight = self.config.kl_end_weight
            
        self.current_kl_weight = weight
        return weight
    
    def apply_free_bits(self, kl_loss: torch.Tensor, kl_per_dim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply free bits to prevent individual dimensions from collapsing"""
        if not self.config.use_free_bits:
            return kl_loss
            
        device = kl_loss.device
        
        if kl_per_dim is not None:
            # Apply free bits per dimension
            free_bits = self.config.free_bits_lambda
            free_bits_tensor = torch.tensor(free_bits, device=device, dtype=kl_per_dim.dtype)
            kl_per_dim = torch.maximum(kl_per_dim, free_bits_tensor)
            return kl_per_dim.sum()
        else:
            # Apply to total KL
            batch_size = kl_loss.shape[0] if kl_loss.dim() > 0 else 1
            total_dims = self.model_config.semantic_dim + self.model_config.attribute_dim
            free_bits_total = self.config.free_bits_lambda * total_dims
            free_bits_tensor = torch.tensor(free_bits_total, device=device, dtype=kl_loss.dtype)
            return torch.maximum(kl_loss, free_bits_tensor)
    
    def get_capacity_constraint(self) -> float:
        """Get current capacity constraint value"""
        if not self.config.use_capacity_constraint:
            return self.config.capacity_max
            
        # Linear increase in capacity
        progress = min(self.current_step / self.config.capacity_num_iters, 1.0)
        capacity = self.config.capacity_min + \
                  (self.config.capacity_max - self.config.capacity_min) * progress
        
        self.current_capacity = capacity
        return capacity
    
    def apply_capacity_constraint(self, kl_loss: torch.Tensor) -> torch.Tensor:
        """Apply capacity constraint to KL loss"""
        if not self.config.use_capacity_constraint:
            return kl_loss
            
        capacity = self.get_capacity_constraint()
        capacity_tensor = torch.tensor(capacity, device=kl_loss.device, dtype=kl_loss.dtype)
        # Constrained KL = |KL - C|
        return torch.abs(kl_loss - capacity_tensor)
    
    def update_loss_history(self, losses: Dict[str, torch.Tensor]):
        """Update loss history for monitoring and dynamic balancing"""
        if 'reconstruction' in losses:
            self.loss_history['reconstruction'].append(losses['reconstruction'].item())
        if 'kl' in losses:
            self.loss_history['kl'].append(losses['kl'].item())
            
        # Track active dimensions
        if 'kl_per_dim' in losses:
            kl_per_dim = losses['kl_per_dim'].detach()
            active_dims = (kl_per_dim > self.config.kl_threshold).float().mean()
            self.loss_history['active_dims'].append(active_dims.item())
    
    def get_dynamic_weights(self) -> Dict[str, float]:
        """Dynamically adjust weights based on loss ratios"""
        if not self.config.use_dynamic_balancing:
            return {'kl_weight': self.current_kl_weight}
            
        # Need enough history
        if len(self.loss_history['reconstruction']) < 10:
            return {'kl_weight': self.current_kl_weight}
            
        # Calculate recent average losses
        recent_recon = np.mean(self.loss_history['reconstruction'][-10:])
        recent_kl = np.mean(self.loss_history['kl'][-10:])
        
        # Avoid division by zero
        if recent_kl < 1e-6:
            # KL collapsed, reduce weight significantly
            adjusted_kl_weight = self.current_kl_weight * 0.1
        else:
            # Calculate current ratio
            current_ratio = recent_recon / recent_kl
            
            # Adjust weight to achieve target ratio
            adjustment = self.config.target_recon_kl_ratio / current_ratio
            adjustment = np.clip(adjustment, 0.5, 2.0)  # Limit adjustment range
            
            adjusted_kl_weight = self.current_kl_weight * adjustment
            
        return {'kl_weight': adjusted_kl_weight}
    
    def check_posterior_collapse(self) -> Dict[str, bool]:
        """Check for signs of posterior collapse"""
        if not self.config.monitor_collapse:
            return {'collapsed': False, 'inactive_dims': False}
            
        results = {}
        
        # Check if KL is too low
        if len(self.loss_history['kl']) > 0:
            recent_kl = np.mean(self.loss_history['kl'][-10:])
            results['collapsed'] = recent_kl < self.config.kl_threshold
        else:
            results['collapsed'] = False
            
        # Check inactive dimensions
        if len(self.loss_history['active_dims']) > 0:
            recent_active = np.mean(self.loss_history['active_dims'][-10:])
            results['inactive_dims'] = recent_active < (1 - self.config.inactive_dims_threshold)
        else:
            results['inactive_dims'] = False
            
        return results
    
    def plot_training_dynamics(self, save_path: Optional[str] = None):
        """Plot training dynamics to visualize balance"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        if self.loss_history['reconstruction']:
            ax.plot(self.loss_history['reconstruction'], label='Reconstruction', alpha=0.7)
        if self.loss_history['kl']:
            ax.plot(self.loss_history['kl'], label='KL', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 2: Loss ratio
        ax = axes[0, 1]
        if len(self.loss_history['reconstruction']) > 0 and len(self.loss_history['kl']) > 0:
            recon = np.array(self.loss_history['reconstruction'])
            kl = np.array(self.loss_history['kl'])
            ratio = recon / (kl + 1e-6)
            ax.plot(ratio, alpha=0.7)
            ax.axhline(y=self.config.target_recon_kl_ratio, color='r', linestyle='--', 
                      label=f'Target Ratio ({self.config.target_recon_kl_ratio})')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reconstruction/KL Ratio')
            ax.set_title('Loss Balance')
            ax.legend()
            ax.set_yscale('log')
        
        # Plot 3: Active dimensions
        ax = axes[1, 0]
        if self.loss_history['active_dims']:
            ax.plot(self.loss_history['active_dims'], alpha=0.7)
            ax.axhline(y=1.0, color='g', linestyle='--', label='All Active')
            ax.axhline(y=1-self.config.inactive_dims_threshold, color='r', linestyle='--', 
                      label='Collapse Threshold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Fraction Active Dims')
            ax.set_title('Latent Dimension Activity')
            ax.legend()
            ax.set_ylim(0, 1.1)
        
        # Plot 4: KL weight schedule
        ax = axes[1, 1]
        epochs = range(self.config.kl_annealing_epochs + 10)
        kl_weights = [self.get_kl_weight(e, 0) for e in epochs]
        ax.plot(epochs, kl_weights, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Weight')
        ax.set_title(f'KL Weight Schedule ({self.config.kl_annealing_type})')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# ENHANCED LOSS COMPUTATION WITH BALANCED WEIGHTS
# ============================================================================

class BalancedVAELoss(nn.Module):
    """VAE loss with balanced weight strategy"""
    
    def __init__(self, model_config, balance_config: BalancedWeightConfig):
        super().__init__()
        self.model_config = model_config
        self.scheduler = BalancedWeightScheduler(balance_config, model_config)
        
    def forward(self, recon_x, x, mu, logvar, epoch=0, step=0):
        """Compute balanced VAE loss"""
        batch_size = x.shape[0]
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        
        # 2. KL divergence with per-dimension tracking
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.sum(dim=1).mean()
        
        # 3. Apply free bits
        if self.scheduler.config.use_free_bits:
            kl_loss_adjusted = self.scheduler.apply_free_bits(kl_loss, kl_per_dim.mean(dim=0))
        else:
            kl_loss_adjusted = kl_loss
            
        # 4. Apply capacity constraint
        if self.scheduler.config.use_capacity_constraint:
            kl_loss_adjusted = self.scheduler.apply_capacity_constraint(kl_loss_adjusted)
            
        # 5. Get current KL weight
        kl_weight = self.scheduler.get_kl_weight(epoch, step)
        
        # 6. Get dynamic adjustments
        dynamic_weights = self.scheduler.get_dynamic_weights()
        kl_weight = dynamic_weights.get('kl_weight', kl_weight)
        
        # 7. Compute total loss
        total_loss = recon_loss + kl_weight * kl_loss_adjusted
        
        # 8. Update history
        self.scheduler.update_loss_history({
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'kl_per_dim': kl_per_dim.mean(dim=0)
        })
        
        # 9. Check for collapse
        collapse_status = self.scheduler.check_posterior_collapse()
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': kl_weight,
            'active_dims': (kl_per_dim.mean(dim=0) > 0.1).float().mean(),
            'collapse_detected': collapse_status['collapsed'],
            'capacity': self.scheduler.current_capacity
        }

# ============================================================================
# INTEGRATION WITH EXISTING MODEL
# ============================================================================

def integrate_balanced_weights_with_model(model, disentangle_config):
    """Integrate balanced weight strategy with existing model"""
    
    # Create balanced weight configuration - MUCH more conservative for MPS
    balance_config = BalancedWeightConfig(
        # KL Annealing - start from 0 and VERY gradually increase
        kl_annealing_type="linear",  # Linear is more stable than cyclical for MPS
        kl_start_weight=0.0,
        kl_end_weight=min(0.0001, disentangle_config.kl_weight * 0.1),  # Much smaller target
        kl_annealing_epochs=100,  # Much longer annealing
        kl_cycle_length=20,
        
        # Free bits - MUCH more aggressive to prevent collapse
        use_free_bits=True,
        free_bits_lambda=5.0,  # Much higher minimum bits per dimension
        
        # Capacity constraint - very gradual increase
        use_capacity_constraint=True,
        capacity_min=0.0,
        capacity_max=10.0,  # Much smaller max capacity
        capacity_num_iters=50000,  # Much slower increase
        
        # Dynamic balancing - very conservative
        use_dynamic_balancing=True,
        target_recon_kl_ratio=1000.0,  # Much higher ratio (favor reconstruction)
        balance_update_freq=50,  # More frequent updates
        
        # Monitoring - very sensitive
        monitor_collapse=True,
        kl_threshold=0.01,  # Much lower threshold
        inactive_dims_threshold=0.1  # Much stricter
    )
    
    # Create balanced loss module
    model.balanced_loss = BalancedVAELoss(disentangle_config, balance_config)
    
    # Override the loss computation
    original_compute_losses = model.compute_disentanglement_losses
    
    def balanced_loss_computation(x, outputs):
        """Enhanced loss computation with balanced weights"""
        device = x.device
        losses = {}
        
        # Get current training step/epoch from model if available
        epoch = getattr(model, 'current_epoch', 0)
        step = getattr(model, 'global_step', 0)
        
        # Compute balanced VAE loss for semantic and attribute separately
        if 'semantic_mu' in outputs and 'semantic_logvar' in outputs:
            semantic_balanced = model.balanced_loss(
                outputs.get('recon_x', x),  # Use reconstruction if available
                x,
                outputs['semantic_mu'],
                outputs['semantic_logvar'],
                epoch=epoch,
                step=step
            )
            for key, value in semantic_balanced.items():
                losses[f'semantic_{key}'] = value
                
        if 'attribute_mu' in outputs and 'attribute_logvar' in outputs:
            attribute_balanced = model.balanced_loss(
                outputs.get('recon_x', x),
                x,
                outputs['attribute_mu'],
                outputs['attribute_logvar'],
                epoch=epoch,
                step=step
            )
            for key, value in attribute_balanced.items():
                losses[f'attribute_{key}'] = value
        
        # Call original computation for other losses
        original_losses = original_compute_losses(x, outputs)
        
        # Merge losses, but use balanced KL losses
        for key, value in original_losses.items():
            if 'kl' not in key.lower():  # Keep non-KL losses
                losses[key] = value
                
        # Adjust total based on collapse detection
        if losses.get('semantic_collapse_detected', False) or losses.get('attribute_collapse_detected', False):
            print("⚠️  Posterior collapse detected! Adjusting weights...")
            # Reduce KL weight further if collapse detected
            for key in losses:
                if 'kl' in key and 'weight' not in key:
                    losses[key] = losses[key] * 0.1
        
        # Compute total with balanced weights
        losses['total'] = sum(v for k, v in losses.items() 
                            if isinstance(v, torch.Tensor) and 'weight' not in k 
                            and 'collapse' not in k and 'active' not in k)
        
        return losses
    
    # Replace the method
    model.compute_disentanglement_losses = balanced_loss_computation
    
    # Add training step tracking
    model.global_step = 0
    model.current_epoch = 0
    
    return model

# ============================================================================
# QUICK FIXES FOR EXISTING MODEL
# ============================================================================

def apply_quick_balanced_fixes(model, config):
    """Quick fixes you can apply immediately to your existing training"""
    
    # 1. Simple cyclical annealing schedule
    def get_cyclical_kl_weight(epoch, base_weight=0.001, cycle_length=10, n_cycles=5):
        """Simple cyclical KL annealing"""
        if epoch >= cycle_length * n_cycles:
            return base_weight
        
        cycle = epoch // cycle_length
        cycle_progress = (epoch % cycle_length) / cycle_length
        
        # Start each cycle from 0
        cycle_min = 0.0
        cycle_max = base_weight * (cycle + 1) / n_cycles
        
        return cycle_min + (cycle_max - cycle_min) * cycle_progress
    
    # 2. Free bits implementation
    def apply_free_bits(kl_loss, free_bits=3.0):
        """Apply free bits to prevent collapse"""
        free_bits_tensor = torch.tensor(free_bits, device=kl_loss.device, dtype=kl_loss.dtype)
        return torch.maximum(kl_loss, free_bits_tensor)
    
    # 3. Monitor active dimensions
    def get_active_dims(mu, logvar, threshold=0.5):
        """Count active dimensions"""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        active = (kl_per_dim.mean(dim=0) > threshold).float().mean()
        return active
    
    # 4. Override your compute_disentanglement_losses
    original_compute = model.compute_disentanglement_losses
    
    def balanced_compute(x, outputs):
        """Quick balanced loss computation"""
        losses = original_compute(x, outputs)
        
        # Get current epoch (you'll need to track this)
        epoch = getattr(model, 'current_epoch', 0)
        
        # Apply cyclical annealing
        kl_weight = get_cyclical_kl_weight(epoch)
        
        # Recompute KL losses with free bits
        if 'semantic_mu' in outputs:
            semantic_kl = -0.5 * torch.sum(
                1 + outputs['semantic_logvar'] - 
                outputs['semantic_mu'].pow(2) - 
                outputs['semantic_logvar'].exp(), dim=1
            ).mean()
            semantic_kl = apply_free_bits(semantic_kl, free_bits=2.0)
            losses['semantic_kl'] = semantic_kl * kl_weight
            
            # Monitor active dims
            active_dims = get_active_dims(outputs['semantic_mu'], outputs['semantic_logvar'])
            if active_dims < 0.5:
                print(f"⚠️  Low semantic active dims: {active_dims:.2%}")
        
        if 'attribute_mu' in outputs:
            attribute_kl = -0.5 * torch.sum(
                1 + outputs['attribute_logvar'] - 
                outputs['attribute_mu'].pow(2) - 
                outputs['attribute_logvar'].exp(), dim=1
            ).mean()
            attribute_kl = apply_free_bits(attribute_kl, free_bits=2.0)
            losses['attribute_kl'] = attribute_kl * kl_weight
            
            # Monitor active dims
            active_dims = get_active_dims(outputs['attribute_mu'], outputs['attribute_logvar'])
            if active_dims < 0.5:
                print(f"⚠️  Low attribute active dims: {active_dims:.2%}")
        
        # Recompute total
        losses['total'] = sum(v for k, v in losses.items() 
                            if isinstance(v, torch.Tensor) and k != 'total')
        
        return losses
    
    model.compute_disentanglement_losses = balanced_compute
    model.current_epoch = 0  # Add epoch tracking
    
    return model

# ============================================================================
# ENHANCED TRAINING LOOP WITH MONITORING
# ============================================================================

def log_reconstructions_to_tensorboard(model, original_images, model_outputs, writer, step, dataset_name, device):
    """Log reconstruction comparisons and factor analysis to TensorBoard"""
    
    with torch.no_grad():
        model.eval()
        
        # Get reconstructions from model outputs
        if hasattr(model_outputs, 'recon_x'):
            reconstructions = model_outputs.recon_x
        else:
            # Fallback: run inference again
            outputs = model({'data': original_images})
            reconstructions = outputs.recon_x
        
        # Ensure we have the right number of samples
        n_samples = min(8, original_images.shape[0], reconstructions.shape[0])
        originals = original_images[:n_samples]
        recons = reconstructions[:n_samples]
        
        # 1. Log original vs reconstruction comparison
        comparison_grid = create_reconstruction_comparison_grid(
            originals, recons, dataset_name
        )
        writer.add_image('Reconstructions/Original_vs_Reconstructed', 
                        comparison_grid, step)
        
        # 2. Log reconstruction error heatmaps
        error_heatmaps = create_reconstruction_error_heatmaps(originals, recons)
        writer.add_image('Reconstructions/Error_Heatmaps', 
                        error_heatmaps, step)
        
        # 3. Log factor activations if available
        if hasattr(model_outputs, 'semantic_factors') and hasattr(model_outputs, 'attribute_factors'):
            factor_viz = create_factor_activation_visualization(
                model_outputs.semantic_factors[:n_samples],
                model_outputs.attribute_factors[:n_samples]
            )
            writer.add_image('Factors/Activation_Patterns', factor_viz, step)
        
        # 4. Log latent space visualizations
        if hasattr(model_outputs, 'semantic_mu') and hasattr(model_outputs, 'attribute_mu'):
            latent_viz = create_latent_space_visualization(
                model_outputs.semantic_mu[:n_samples],
                model_outputs.attribute_mu[:n_samples]
            )
            writer.add_image('Latent/Semantic_Attribute_Space', latent_viz, step)
        
        # 5. Log reconstruction quality metrics
        mse_per_sample = torch.mean((originals - recons) ** 2, dim=[1, 2, 3])
        writer.add_histogram('Reconstructions/MSE_Per_Sample', mse_per_sample, step)
        writer.add_scalar('Reconstructions/Average_MSE', mse_per_sample.mean(), step)
        
        # 6. Log pixel-wise statistics
        writer.add_histogram('Reconstructions/Original_Pixel_Values', originals, step)
        writer.add_histogram('Reconstructions/Reconstructed_Pixel_Values', recons, step)
        
        model.train()

def create_reconstruction_comparison_grid(originals, reconstructions, dataset_name):
    """Create a grid showing original vs reconstructed images"""
    import torchvision.utils as vutils
    
    n_samples = originals.shape[0]
    
    # Handle different image formats
    if originals.shape[1] == 1:  # Grayscale
        originals = originals.repeat(1, 3, 1, 1)  # Convert to RGB for visualization
        reconstructions = reconstructions.repeat(1, 3, 1, 1)
    
    # Normalize to [0, 1] range
    originals = torch.clamp(originals, 0, 1)
    reconstructions = torch.clamp(reconstructions, 0, 1)
    
    # Create interleaved grid: original, reconstruction, original, reconstruction, ...
    grid_images = []
    for i in range(n_samples):
        grid_images.append(originals[i])
        grid_images.append(reconstructions[i])
    
    grid = vutils.make_grid(
        torch.stack(grid_images), 
        nrow=2,  # 2 columns: original, reconstruction
        padding=2,
        normalize=False,
        pad_value=1.0  # White padding
    )
    
    return grid

def create_reconstruction_error_heatmaps(originals, reconstructions):
    """Create heatmaps showing reconstruction errors"""
    import torchvision.utils as vutils
    
    # Compute pixel-wise absolute error
    error = torch.abs(originals - reconstructions)
    
    # Convert to grayscale if needed for error visualization
    if error.shape[1] == 3:
        error = 0.299 * error[:, 0] + 0.587 * error[:, 1] + 0.114 * error[:, 2]
        error = error.unsqueeze(1)
    
    # Normalize error to [0, 1] for visualization
    error_normalized = []
    for i in range(error.shape[0]):
        err = error[i]
        err_min, err_max = err.min(), err.max()
        if err_max > err_min:
            err_norm = (err - err_min) / (err_max - err_min)
        else:
            err_norm = err
        error_normalized.append(err_norm)
    
    error_tensor = torch.stack(error_normalized)
    
    # Apply colormap (convert to RGB heatmap)
    heatmaps = []
    for i in range(error_tensor.shape[0]):
        # Simple colormap: blue (low error) to red (high error)
        err = error_tensor[i, 0]  # Remove channel dimension
        
        # Create RGB heatmap
        red = err  # High error = more red
        blue = 1 - err  # Low error = more blue
        green = torch.zeros_like(err)
        
        heatmap = torch.stack([red, green, blue], dim=0)
        heatmaps.append(heatmap)
    
    grid = vutils.make_grid(heatmaps, nrow=4, padding=2, normalize=False)
    return grid

def create_factor_activation_visualization(semantic_factors, attribute_factors):
    """Create visualization of factor activations"""
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    n_samples = semantic_factors.shape[0]
    n_semantic = semantic_factors.shape[1]
    n_attribute = attribute_factors.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 2*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Semantic factors
        axes[i, 0].bar(range(n_semantic), semantic_factors[i].cpu().numpy())
        axes[i, 0].set_title(f'Sample {i+1}: Semantic Factors')
        axes[i, 0].set_ylim(0, 1)
        
        # Attribute factors
        axes[i, 1].bar(range(n_attribute), attribute_factors[i].cpu().numpy())
        axes[i, 1].set_title(f'Sample {i+1}: Attribute Factors')
        axes[i, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Convert to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    
    plt.close(fig)
    buf.close()
    
    return img_tensor

def create_latent_space_visualization(semantic_mu, attribute_mu):
    """Create 2D visualization of latent space"""
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    
    semantic_np = semantic_mu.cpu().numpy()
    attribute_np = attribute_mu.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Semantic space
    if semantic_np.shape[1] > 2:
        pca_semantic = PCA(n_components=2)
        semantic_2d = pca_semantic.fit_transform(semantic_np)
    else:
        semantic_2d = semantic_np
    
    axes[0].scatter(semantic_2d[:, 0], semantic_2d[:, 1], alpha=0.7, s=50)
    axes[0].set_title('Semantic Latent Space (2D PCA)')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].grid(True, alpha=0.3)
    
    # Attribute space
    if attribute_np.shape[1] > 2:
        pca_attribute = PCA(n_components=2)
        attribute_2d = pca_attribute.fit_transform(attribute_np)
    else:
        attribute_2d = attribute_np
    
    axes[1].scatter(attribute_2d[:, 0], attribute_2d[:, 1], alpha=0.7, s=50, color='orange')
    axes[1].set_title('Attribute Latent Space (2D PCA)')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    
    plt.close(fig)
    buf.close()
    
    return img_tensor

def save_reconstruction_samples(model, data_loader, epoch, save_dir, device, n_samples=16):
    """Save reconstruction samples to disk"""
    model.eval()
    
    # Create save directory
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(data_loader))
        if isinstance(batch, dict):
            x = batch['data'][:n_samples].to(device)
            labels = batch.get('label', None)
        else:
            x = batch[0][:n_samples].to(device)
            labels = batch[1][:n_samples] if len(batch) > 1 else None
        
        # Get reconstructions
        outputs = model({'data': x, 'label': labels})
        recons = outputs.recon_x
        
        # Create comparison figure
        fig, axes = plt.subplots(4, n_samples//2, figsize=(20, 8))
        
        for i in range(min(n_samples//2, x.shape[0])):
            # Original
            if x.shape[1] == 1:  # Grayscale
                axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap='gray')
                axes[2, i].imshow(x[i+n_samples//2, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                axes[0, i].imshow(x[i].cpu().permute(1, 2, 0).numpy())
                axes[2, i].imshow(x[i+n_samples//2].cpu().permute(1, 2, 0).numpy())
            
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            axes[2, i].set_title(f'Original {i+1+n_samples//2}')
            axes[2, i].axis('off')
            
            # Reconstruction
            if recons.shape[1] == 1:  # Grayscale
                axes[1, i].imshow(recons[i, 0].cpu().numpy(), cmap='gray')
                axes[3, i].imshow(recons[i+n_samples//2, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                axes[1, i].imshow(torch.clamp(recons[i], 0, 1).cpu().permute(1, 2, 0).numpy())
                axes[3, i].imshow(torch.clamp(recons[i+n_samples//2], 0, 1).cpu().permute(1, 2, 0).numpy())
            
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            axes[3, i].set_title(f'Reconstructed {i+1+n_samples//2}')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, 'reconstructions.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual factor activations if available
        if hasattr(outputs, 'semantic_factors') and hasattr(outputs, 'attribute_factors'):
            save_factor_analysis(outputs, epoch_dir, labels)
    
    model.train()

def save_factor_analysis(outputs, save_dir, labels=None):
    """Save detailed factor analysis"""
    
    semantic_factors = outputs.semantic_factors.cpu().numpy()
    attribute_factors = outputs.attribute_factors.cpu().numpy()
    
    # Factor activation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Semantic factors heatmap
    im1 = axes[0].imshow(semantic_factors.T, aspect='auto', cmap='viridis')
    axes[0].set_title('Semantic Factor Activations')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Factor Index')
    plt.colorbar(im1, ax=axes[0])
    
    # Attribute factors heatmap
    im2 = axes[1].imshow(attribute_factors.T, aspect='auto', cmap='viridis')
    axes[1].set_title('Attribute Factor Activations')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Factor Index')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'factor_activations.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Factor statistics
    stats = {
        'semantic_sparsity': (semantic_factors < 0.1).mean(),
        'attribute_sparsity': (attribute_factors < 0.1).mean(),
        'semantic_mean_activation': semantic_factors.mean(),
        'attribute_mean_activation': attribute_factors.mean(),
        'semantic_active_factors': (semantic_factors > 0.5).sum(axis=1).mean(),
        'attribute_active_factors': (attribute_factors > 0.5).sum(axis=1).mean()
    }
    
    # Save statistics
    with open(os.path.join(save_dir, 'factor_stats.txt'), 'w') as f:
        f.write("Factor Analysis Statistics\n")
        f.write("=" * 30 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")

def train_with_balanced_weights(model, train_loader, val_loader, config, epochs=100, device='cuda' if torch.cuda.is_available() else 'mps'):
    """Training loop with balanced weight monitoring, tqdm progress bars, and TensorBoard logging"""
    
    # Force output flushing for real-time display
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Configure tqdm for better terminal output
    tqdm.monitor_interval = 0
    tqdm.miniters = 1
    
    print("🚀 Starting Balanced Weight Training...")
    print("=" * 60)
    sys.stdout.flush()
    
    # MPS device fixes
    if device == 'mps':
        print("🍎 Detected MPS (Apple Silicon) - applying compatibility fixes...")
        # Disable some MPS-incompatible operations
        torch.backends.mps.allow_tf32 = False
        sys.stdout.flush()
    
    # Integrate balanced weights
    model = integrate_balanced_weights_with_model(model, config)
    model = model.to(device)
    
    # Emergency collapse detection and recovery
    collapse_recovery_mode = False
    consecutive_collapses = 0
    
    # Setup TensorBoard logging
    log_dir = f'balanced_vae_logs/{config.dataset_name}_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"   Run: tensorboard --logdir={log_dir}")
    print(f"🔧 Device: {device}")
    print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()
    
    # Optimizer with scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'collapse_events': [],
        'kl_weights': [],
        'active_dims': [],
        'capacities': []
    }
    
    # Training loop with epoch progress bar
    print(f"\n🏃 Starting training for {epochs} epochs...")
    sys.stdout.flush()
    
    epoch_pbar = tqdm(
        range(epochs), 
        desc="🎯 Training Progress", 
        unit="epoch",
        ncols=100,
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True
    )
    
    for epoch in epoch_pbar:
        model.current_epoch = epoch
        model.train()
        
        # Training metrics for this epoch
        epoch_metrics = {
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'semantic_loss': [],
            'attribute_loss': [],
            'active_dims': [],
            'kl_weights': [],
            'capacities': []
        }
        
        # Training batch progress bar
        batch_pbar = tqdm(
            train_loader, 
            desc=f"📈 Epoch {epoch+1:3d}/{epochs}", 
            leave=False, 
            unit="batch",
            ncols=120,
            file=sys.stdout,
            dynamic_ncols=True,
            miniters=1
        )
        
        for batch_idx, batch in enumerate(batch_pbar):
            # Get data
            if isinstance(batch, dict):
                x = batch['data'].to(device)
                labels = batch.get('label', None)
            else:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
            
            # Forward pass
            outputs = model({'data': x, 'label': labels})
            
            # Backward pass
            optimizer.zero_grad()
            outputs.loss.backward()
            
            # Gradient clipping to prevent instability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update global step
            model.global_step += 1
            
            # Collect metrics
            epoch_metrics['train_loss'].append(outputs.loss.item())
            
            # Get detailed loss components
            if hasattr(outputs, 'recon_loss'):
                epoch_metrics['recon_loss'].append(outputs.recon_loss.item())
            if hasattr(outputs, 'reg_loss'):
                epoch_metrics['kl_loss'].append(outputs.reg_loss.item())
            
            # Get balanced weight metrics
            if hasattr(model, 'balanced_loss'):
                scheduler_info = model.balanced_loss.scheduler
                current_kl_weight = scheduler_info.current_kl_weight
                current_capacity = scheduler_info.current_capacity
                
                epoch_metrics['kl_weights'].append(current_kl_weight)
                epoch_metrics['capacities'].append(current_capacity)
                
                # Get active dimensions
                if scheduler_info.loss_history['active_dims']:
                    active_dims = scheduler_info.loss_history['active_dims'][-1]
                    epoch_metrics['active_dims'].append(active_dims)
                
                # Check for collapse and apply emergency recovery
                collapse_status = scheduler_info.check_posterior_collapse()
                current_loss = outputs.loss.item()
                
                # Emergency collapse detection (very high loss or 0% active dims)
                emergency_collapse = (current_loss > 10000 or 
                                    (epoch_metrics['active_dims'] and 
                                     epoch_metrics['active_dims'][-1] < 0.01))
                
                if collapse_status['collapsed'] or emergency_collapse:
                    history['collapse_events'].append((epoch, batch_idx))
                    consecutive_collapses += 1
                    
                    # Emergency recovery actions
                    if consecutive_collapses > 5 or emergency_collapse:
                        print(f"\n🚨 EMERGENCY COLLAPSE DETECTED! Loss: {current_loss:.1f}")
                        print("   Applying emergency recovery...")
                        
                        # Drastically reduce KL weight
                        scheduler_info.current_kl_weight *= 0.01
                        
                        # Reset optimizer state
                        optimizer.state = defaultdict(dict)
                        
                        # Reduce learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        
                        consecutive_collapses = 0
                        collapse_recovery_mode = True
                        sys.stdout.flush()
                    
                    batch_pbar.set_postfix({
                        'Loss': f"{current_loss:.1f}",
                        'KL_Weight': f"{current_kl_weight:.8f}",
                        'COLLAPSE': "🚨" if emergency_collapse else "⚠️"
                    })
                else:
                    consecutive_collapses = max(0, consecutive_collapses - 1)
                    batch_pbar.set_postfix({
                        'Loss': f"{current_loss:.4f}",
                        'KL_Weight': f"{current_kl_weight:.6f}",
                        'Active': f"{active_dims:.1%}" if epoch_metrics['active_dims'] else "N/A"
                    })
            
                         # Log to TensorBoard every 50 steps
                if model.global_step % 50 == 0:
                     # Basic losses
                    writer.add_scalar('Loss/Train_Total', outputs.loss.item(), model.global_step)
                    if hasattr(outputs, 'recon_loss'):
                         writer.add_scalar('Loss/Reconstruction', outputs.recon_loss.item(), model.global_step)
                    if hasattr(outputs, 'reg_loss'):
                         writer.add_scalar('Loss/KL_Divergence', outputs.reg_loss.item(), model.global_step)
                 
                    # Balanced weight metrics
                    if hasattr(model, 'balanced_loss'):
                         scheduler_info = model.balanced_loss.scheduler
                         writer.add_scalar('BalancedWeights/KL_Weight', scheduler_info.current_kl_weight, model.global_step)
                         writer.add_scalar('BalancedWeights/Capacity', scheduler_info.current_capacity, model.global_step)
                     
                    # Active dimensions
                    if scheduler_info.loss_history['active_dims']:
                         writer.add_scalar('BalancedWeights/Active_Dimensions', 
                                         scheduler_info.loss_history['active_dims'][-1], model.global_step)
                     
                    # Log reconstruction/KL ratio
                    if (scheduler_info.loss_history['reconstruction'] and 
                         scheduler_info.loss_history['kl']):
                         recent_recon = np.mean(scheduler_info.loss_history['reconstruction'][-10:])
                         recent_kl = np.mean(scheduler_info.loss_history['kl'][-10:])
                         if recent_kl > 1e-6:
                             ratio = recent_recon / recent_kl
                             writer.add_scalar('BalancedWeights/Recon_KL_Ratio', ratio, model.global_step)
                 
                    # Gradient norm
                    writer.add_scalar('Training/Gradient_Norm', grad_norm, model.global_step)
                    writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], model.global_step)
                 
                    # Log sample reconstructions every 200 steps
                    if model.global_step % 200 == 0:
                         log_reconstructions_to_tensorboard(
                         model, x[:8], outputs, writer, model.global_step, 
                         config.dataset_name, device
                     )
        
        # End of epoch - compute averages
        avg_train_loss = np.mean(epoch_metrics['train_loss'])
        history['train_loss'].append(avg_train_loss)
        
        if epoch_metrics['kl_weights']:
            avg_kl_weight = np.mean(epoch_metrics['kl_weights'])
            history['kl_weights'].append(avg_kl_weight)
        
        if epoch_metrics['active_dims']:
            avg_active_dims = np.mean(epoch_metrics['active_dims'])
            history['active_dims'].append(avg_active_dims)
        
        if epoch_metrics['capacities']:
            avg_capacity = np.mean(epoch_metrics['capacities'])
            history['capacities'].append(avg_capacity)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_metrics = {
                'val_loss': [],
                'val_recon_loss': [],
                'val_kl_loss': []
            }
            
            # Validation progress bar
            val_pbar = tqdm(
                val_loader, 
                desc=f"🔍 Validation {epoch+1:3d}", 
                leave=False, 
                unit="batch",
                ncols=100,
                file=sys.stdout,
                dynamic_ncols=True,
                miniters=1
            )
            
            # Store first batch for reconstruction logging
            first_val_batch = None
            first_val_outputs = None
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_pbar):
                    if isinstance(batch, dict):
                        x = batch['data'].to(device)
                        labels = batch.get('label', None)
                    else:
                        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        labels = batch[1].to(device) if len(batch) > 1 else None
                    
                    outputs = model({'data': x, 'label': labels})
                    val_metrics['val_loss'].append(outputs.loss.item())
                    
                    if hasattr(outputs, 'recon_loss'):
                        val_metrics['val_recon_loss'].append(outputs.recon_loss.item())
                    if hasattr(outputs, 'reg_loss'):
                        val_metrics['val_kl_loss'].append(outputs.reg_loss.item())
                    
                    val_pbar.set_postfix({'Val_Loss': f"{outputs.loss.item():.4f}"})
                    
                    # Store first batch for reconstruction visualization
                    if batch_idx == 0:
                        first_val_batch = x[:8]  # Take first 8 samples
                        first_val_outputs = outputs
            
            # Validation averages
            avg_val_loss = np.mean(val_metrics['val_loss'])
            history['val_loss'].append(avg_val_loss)
            
            # Log validation metrics
            writer.add_scalar('Loss/Validation_Total', avg_val_loss, epoch)
            if val_metrics['val_recon_loss']:
                writer.add_scalar('Loss/Validation_Reconstruction', 
                                np.mean(val_metrics['val_recon_loss']), epoch)
            if val_metrics['val_kl_loss']:
                writer.add_scalar('Loss/Validation_KL', 
                                np.mean(val_metrics['val_kl_loss']), epoch)
            
            # Log validation reconstructions
            if first_val_batch is not None and first_val_outputs is not None:
                log_reconstructions_to_tensorboard(
                    model, first_val_batch, first_val_outputs, writer, 
                    epoch, config.dataset_name, device
                )
                
                # Save reconstruction samples to disk every 10 epochs
                if epoch % 10 == 0:
                    save_reconstruction_samples(
                        model, val_loader, epoch, log_dir, device, n_samples=16
                    )
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Update epoch progress bar with validation info
            epoch_pbar.set_postfix({
                'Train_Loss': f"{avg_train_loss:.4f}",
                'Val_Loss': f"{avg_val_loss:.4f}",
                'KL_Weight': f"{history['kl_weights'][-1]:.6f}" if history['kl_weights'] else "N/A",
                'Active_Dims': f"{history['active_dims'][-1]:.1%}" if history['active_dims'] else "N/A"
            })
            
            # Log epoch summary with explicit flushing
            summary_msg = f"\n📊 Epoch {epoch+1} Summary:"
            summary_msg += f"\n   📉 Train Loss: {avg_train_loss:.4f}"
            summary_msg += f"\n   📊 Val Loss: {avg_val_loss:.4f}"
            if history['kl_weights']:
                summary_msg += f"\n   ⚖️  KL Weight: {history['kl_weights'][-1]:.6f}"
            if history['active_dims']:
                summary_msg += f"\n   🧠 Active Dims: {history['active_dims'][-1]:.1%}"
            if history['collapse_events']:
                recent_collapses = [e for e in history['collapse_events'] if e[0] == epoch]
                if recent_collapses:
                    summary_msg += f"\n   ⚠️  Collapse events this epoch: {len(recent_collapses)}"
            
            tqdm.write(summary_msg)
            sys.stdout.flush()
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        if history['kl_weights']:
            writer.add_scalar('BalancedWeights/KL_Weight_Epoch', history['kl_weights'][-1], epoch)
        if history['active_dims']:
            writer.add_scalar('BalancedWeights/Active_Dims_Epoch', history['active_dims'][-1], epoch)
        if history['capacities']:
            writer.add_scalar('BalancedWeights/Capacity_Epoch', history['capacities'][-1], epoch)
    
    # Training completed
    epoch_pbar.close()
    
    # Final logging and visualization
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📈 Total epochs: {epochs}")
    print(f"⚠️  Total collapse events: {len(history['collapse_events'])}")
    if history['train_loss']:
        print(f"📉 Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"📊 Final val loss: {history['val_loss'][-1]:.4f}")
    if history['kl_weights']:
        print(f"⚖️  Final KL weight: {history['kl_weights'][-1]:.6f}")
    if history['active_dims']:
        print(f"🧠 Final active dims: {history['active_dims'][-1]:.1%}")
    
    sys.stdout.flush()
    
    # Log final training curves
    if history['train_loss']:
        for i, loss in enumerate(history['train_loss']):
            writer.add_scalar('Curves/Train_Loss', loss, i * 5)  # Every 5 epochs
    
    if history['val_loss']:
        for i, loss in enumerate(history['val_loss']):
            writer.add_scalar('Curves/Val_Loss', loss, i * 5)
    
    # Plot and save training dynamics
    print("\n📈 Plotting training dynamics...")
    if hasattr(model, 'balanced_loss'):
        model.balanced_loss.scheduler.plot_training_dynamics(
            os.path.join(log_dir, 'balanced_training_dynamics.png')
        )
        print(f"   Training dynamics saved to: {log_dir}/balanced_training_dynamics.png")
    
    # Log collapse events summary
    if history['collapse_events']:
        collapse_epochs = [e[0] for e in history['collapse_events']]
        collapse_histogram = np.bincount(collapse_epochs, minlength=epochs)
        for epoch, count in enumerate(collapse_histogram):
            writer.add_scalar('Monitoring/Collapse_Events_Per_Epoch', count, epoch)
    
    # Close TensorBoard writer
    writer.close()
    print(f"📊 TensorBoard logs saved to: {log_dir}")
    print(f"   View with: tensorboard --logdir={log_dir}")
    
    return model, history

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
    device: str = 'cuda' if torch.cuda.is_available() else 'mps'

@dataclass
class MedMNISTVAEConfig:
    """Optimized config specifically for MedMNIST datasets"""
    
    # Dataset-specific sizing (MedMNIST is 28x28 native, but we can upsample)
    input_channels: int = 3
    input_size: int = 64  # Good compromise for MedMNIST
    
    # Adjusted latent dims for medical data complexity
    semantic_dim: int = 20    # Slightly larger than before
    attribute_dim: int = 12   # Adjusted
    
    # Factor counts adjusted for medical patterns
    n_semantic_factors: int = 40   
    n_attribute_factors: int = 20  
    
    # CRITICAL: Loss weights tuned for MedMNIST
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.0005      # Much smaller than your current
    sparsity_weight: float = 0.005  # Moderate sparsity
    orthogonality_weight: float = 0.002
    
    # Training settings optimized for medical data
    batch_size: int = 32       # Good balance for 64x64
    learning_rate: float = 1e-3
    epochs: int = 100
    
    # Dataset info
    num_classes: int = 7
    dataset_name: str = "pathmnist"
    device: str = 'cuda' if torch.cuda.is_available() else 'mps'

# ============================================================================
# SPARSE FACTORIZATION LAYERS
# ============================================================================

class SparseFactorization(nn.Module):
    """
    Sparse factorization layer for interpretable representations.
    Not a "concept layer" - just sparse linear transformation.
    """
    def __init__(self, input_dim, n_factors, target_sparsity=0.3, use_improved=False):
        super().__init__()
        
        # Use improved sparsity layer for medical data
        if use_improved:
            self.layer = MedMNISTSparsityLayer(input_dim, n_factors, target_sparsity)
        else:
            # Original implementation
            # Linear factorization with sparsity
            self.factorization = nn.Linear(input_dim, n_factors, bias=False)
            
            # Learnable thresholds for sparsity
            self.thresholds = nn.Parameter(torch.zeros(n_factors))
            
            # Target sparsity level
            self.target_sparsity = target_sparsity
            
            # Optional: Non-negative constraints
            self.activation = nn.ReLU()
            self.layer = None
        
    def forward(self, z):
        if self.layer is not None:
            return self.layer(z)
        else:
            # Original implementation
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
        
        # Check if this is for medical data
        is_medical = hasattr(self.disentangle_config, 'dataset_name') and 'mnist' in self.disentangle_config.dataset_name.lower() and self.disentangle_config.dataset_name != 'mnist'
        
        # Sparse factorization layers (not concept extraction!)
        self.semantic_factorization = SparseFactorization(
            self.disentangle_config.semantic_dim,
            self.disentangle_config.n_semantic_factors,
            target_sparsity=0.3,
            use_improved=is_medical
        )
        
        self.attribute_factorization = SparseFactorization(
            self.disentangle_config.attribute_dim,
            self.disentangle_config.n_attribute_factors,
            target_sparsity=0.3,
            use_improved=is_medical
        )
        
        # Medical reconstruction loss for medical datasets
        if is_medical:
            self.medical_recon_loss = MedicalReconstructionLoss()
            self.use_medical_loss = True
        else:
            self.medical_recon_loss = None
            self.use_medical_loss = False
        
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
            'labels': labels,
            'recon_x': decoder_output.reconstruction  # Add reconstruction for medical loss
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
        
        # Medical reconstruction loss if available
        if self.use_medical_loss and self.medical_recon_loss is not None:
            # Get reconstruction from the outputs
            recon_x = outputs.get('recon_x')
            if recon_x is not None:
                medical_recon, components = self.medical_recon_loss(recon_x, x)
                losses['medical_reconstruction'] = medical_recon * 0.5
            else:
                losses['medical_reconstruction'] = torch.tensor(0.0, device=device)
        else:
            losses['medical_reconstruction'] = torch.tensor(0.0, device=device)
        
        # 1. Factor sparsity with target (FIXED computation)
        target_sparsity = 0.3
        semantic_activation = outputs['semantic_factors'].mean()
        attribute_activation = outputs['attribute_factors'].mean()
        
        # Improved sparsity loss that actually achieves target
        semantic_sparsity_actual = (outputs['semantic_factors'] < 0.1).float().mean()
        attribute_sparsity_actual = (outputs['attribute_factors'] < 0.1).float().mean()
        
        sparsity_loss = (
            (semantic_sparsity_actual - target_sparsity).pow(2) +
            (attribute_sparsity_actual - target_sparsity).pow(2)
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
            try:
                tc_loss = self.estimate_tc_loss(z_samples)
                losses['tc_loss'] = tc_loss * 0.01
            except NotImplementedError:
                # Fallback for MPS: simplified correlation loss
                z_centered = z_samples - z_samples.mean(dim=0, keepdim=True)
                cov_matrix = torch.mm(z_centered.t(), z_centered) / (batch_size - 1)
                off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
                tc_loss = torch.sum(off_diag.pow(2))
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
    """Wrapper for MedMNIST datasets with correct preprocessing"""
    def __init__(self, dataset_name='pathmnist', split='train', transform=None, size=64, use_medmnist_preprocessing=True):
        if not MEDMNIST_AVAILABLE:
            raise ImportError("MedMNIST not available. Install with: pip install medmnist")
            
        info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        self.dataset = DataClass(split=split, download=True, transform=None)  # Don't apply transform yet
        self.num_classes = len(info['label'])
        self.task = info['task']
        self.size = size
        self.dataset_name = dataset_name
        
        # Use MedMNIST-specific preprocessing or custom transform
        if use_medmnist_preprocessing and transform is None:
            self.transform = MedMNISTPreprocessing.get_medmnist_transforms(
                dataset_name=dataset_name, 
                input_size=size, 
                normalize=True
            )
        else:
            # Fallback to standard preprocessing
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
        
        # Temporarily skip this logging to avoid channel mismatch issues
        # TODO: Fix channel mismatch between model expectation and data format
        try:
            # Just log basic metrics without running model forward pass
            self.writer.add_scalar("training/epoch", epoch, epoch)
            
        except Exception as e:
            print(f"Could not log disentanglement metrics: {e}")
        
        return
    
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
        # Temporarily skip this logging to avoid channel mismatch issues
        # TODO: Fix channel mismatch between model expectation and data format
        try:
            # Just log a placeholder for now
            self.writer.add_scalar("reconstructions/epoch", epoch, epoch)
            
        except Exception as e:
            print(f"Could not log sample reconstructions: {e}")
        
        return
    
    def log_factor_prototypes(self, epoch):
        """Log factor prototypes to TensorBoard"""
        # Temporarily skip this logging to avoid channel mismatch issues
        # TODO: Fix channel mismatch between model expectation and data format
        try:
            # Just log a placeholder for now
            self.writer.add_scalar("prototypes/epoch", epoch, epoch)
            
        except Exception as e:
            print(f"Could not log factor prototypes: {e}")
        
        return
    
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
# CRITICAL FIXES FOR PATHOLOGY VAE RECONSTRUCTION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg16

# ============================================================================
# 1. PATHOLOGY-SPECIFIC RECONSTRUCTION LOSS
# ============================================================================

class PathologyReconstructionLoss(nn.Module):
    """
    Specialized loss for pathology images that preserves diagnostic detail
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'mps'):
        super().__init__()
        self.device = device
        
        # Use VGG features for perceptual loss (preserves texture/structure)
        vgg = vgg16(pretrained=True).features[:16]  # Up to conv3_3
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        
        # Sobel filters for edge detection (critical for pathology)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, recon, target):
        batch_size = recon.shape[0]
        
        # 1. Pixel-level losses
        mse_loss = F.mse_loss(recon, target)
        l1_loss = F.l1_loss(recon, target)
        
        # 2. Perceptual loss (preserves pathological structures)
        recon_features = self.vgg(recon)
        target_features = self.vgg(target)
        perceptual_loss = F.mse_loss(recon_features, target_features)
        
        # 3. Edge preservation loss (critical for cell boundaries)
        edge_loss = self.compute_edge_loss(recon, target)
        
        # 4. Color consistency loss (important for staining)
        color_loss = self.compute_color_loss(recon, target)
        
        # 5. High-frequency detail loss
        detail_loss = self.compute_detail_loss(recon, target)
        
        # Weighted combination for pathology
        total_loss = (
            0.4 * mse_loss +           # Basic reconstruction
            0.3 * perceptual_loss +    # Structural preservation  
            0.15 * edge_loss +         # Cell boundary preservation
            0.1 * color_loss +         # Staining preservation
            0.05 * detail_loss         # Fine detail preservation
        )
        
        return total_loss, {
            'mse': mse_loss.detach(),
            'perceptual': perceptual_loss.detach(),
            'edge': edge_loss.detach(),
            'color': color_loss.detach(),
            'detail': detail_loss.detach()
        }
    
    def compute_edge_loss(self, recon, target):
        """Preserve cell boundaries and tissue structures"""
        # Convert to grayscale for edge detection
        recon_gray = 0.299 * recon[:, 0] + 0.587 * recon[:, 1] + 0.114 * recon[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Add channel dimension
        recon_gray = recon_gray.unsqueeze(1)
        target_gray = target_gray.unsqueeze(1)
        
        # Apply Sobel filters
        recon_edges_x = F.conv2d(recon_gray, self.sobel_x, padding=1)
        recon_edges_y = F.conv2d(recon_gray, self.sobel_y, padding=1)
        recon_edges = torch.sqrt(recon_edges_x**2 + recon_edges_y**2)
        
        target_edges_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        return F.mse_loss(recon_edges, target_edges)
    
    def compute_color_loss(self, recon, target):
        """Preserve staining characteristics"""
        # Color histogram loss in LAB space would be ideal, 
        # but we'll use RGB channel correlation
        recon_flat = recon.view(recon.shape[0], 3, -1)
        target_flat = target.view(target.shape[0], 3, -1)
        
        # Compute channel-wise statistics
        recon_mean = recon_flat.mean(dim=2)
        target_mean = target_flat.mean(dim=2)
        
        recon_std = recon_flat.std(dim=2)
        target_std = target_flat.std(dim=2)
        
        mean_loss = F.mse_loss(recon_mean, target_mean)
        std_loss = F.mse_loss(recon_std, target_std)
        
        return mean_loss + std_loss
    
    def compute_detail_loss(self, recon, target):
        """Preserve fine pathological details using high-pass filter"""
        # Simple high-pass filter (Laplacian)
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                dtype=torch.float32, device=self.device)
        laplacian = laplacian.view(1, 1, 3, 3)
        
        # Apply to each channel
        detail_loss = 0
        for i in range(3):
            recon_detail = F.conv2d(recon[:, i:i+1], laplacian, padding=1)
            target_detail = F.conv2d(target[:, i:i+1], laplacian, padding=1)
            detail_loss += F.mse_loss(recon_detail, target_detail)
        
        return detail_loss / 3

# ============================================================================
# 2. IMPROVED SPARSITY FOR PATHOLOGY FACTORS
# ============================================================================

class PathologySparsityLayer(nn.Module):
    """
    Sparsity layer that learns pathology-specific factors
    """
    def __init__(self, input_dim, n_factors, target_sparsity=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_factors = n_factors
        self.target_sparsity = target_sparsity
        
        # Main factorization with bias
        self.factorization = nn.Linear(input_dim, n_factors, bias=True)
        
        # Learnable temperature for competitive activation
        self.temperature = nn.Parameter(torch.ones(1) * 3.0)
        
        # Group normalization for stability
        num_groups = min(8, n_factors // 4)
        self.group_norm = nn.GroupNorm(num_groups, n_factors)
        
        # Initialize with Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.factorization.weight)
        nn.init.zeros_(self.factorization.bias)
        
    def forward(self, z):
        # Linear transformation
        factors = self.factorization(z)
        
        # Group normalization
        factors = self.group_norm(factors)
        
        # Competitive activation with learnable temperature
        factors = torch.softmax(factors / self.temperature, dim=1)
        
        # Apply sparsity constraint
        if self.training:
            factors = self.apply_competitive_sparsity(factors)
        
        return factors
    
    def apply_competitive_sparsity(self, x):
        """
        Competitive sparsity: only top-k factors per sample are active
        """
        # Calculate k for target sparsity
        k = max(1, int(self.n_factors * (1 - self.target_sparsity)))
        
        # Get top-k indices per sample
        topk_values, topk_indices = torch.topk(x, k, dim=1)
        
        # Create sparse representation
        sparse_factors = torch.zeros_like(x)
        sparse_factors.scatter_(1, topk_indices, topk_values)
        
        # Renormalize to maintain total activation
        sparse_factors = sparse_factors / sparse_factors.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return sparse_factors

# ============================================================================
# 3. PATHOLOGY-AWARE ENCODER/DECODER
# ============================================================================

class PathologyEncoder(nn.Module):
    """Enhanced encoder for pathology images"""
    
    def __init__(self, input_channels=3, input_size=64, semantic_dim=20, attribute_dim=12):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.attribute_dim = attribute_dim
        
        # Multi-scale feature extraction (important for pathology)
        self.features_scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.features_scale2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),  # Downsample
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.features_scale3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.features_scale4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.flatten_size = 256 * 4 * 4
        
        # Separate semantic and attribute encoders
        self.semantic_encoder = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, semantic_dim * 2)
        )
        
        self.attribute_encoder = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, attribute_dim * 2)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.features_scale1(x)
        feat2 = self.features_scale2(feat1)
        feat3 = self.features_scale3(feat2)
        feat4 = self.features_scale4(feat3)
        
        # Flatten
        features = feat4.view(feat4.size(0), -1)
        
        # Encode semantic and attribute
        semantic_params = self.semantic_encoder(features)
        semantic_mu = semantic_params[:, :self.semantic_dim]
        semantic_logvar = semantic_params[:, self.semantic_dim:]
        
        attribute_params = self.attribute_encoder(features)
        attribute_mu = attribute_params[:, :self.attribute_dim]
        attribute_logvar = attribute_params[:, self.attribute_dim:]
        
        return {
            'semantic_mu': semantic_mu,
            'semantic_logvar': semantic_logvar,
            'attribute_mu': attribute_mu,
            'attribute_logvar': attribute_logvar
        }

class PathologyDecoder(nn.Module):
    """Enhanced decoder for pathology images"""
    
    def __init__(self, semantic_dim=20, attribute_dim=12, output_channels=3, output_size=64):
        super().__init__()
        
        latent_dim = semantic_dim + attribute_dim
        
        # Project to feature space
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU()
        )
        
        # Progressive upsampling with skip connections
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        # Final refinement layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, output_channels, 3, 1, 1),
            nn.Sigmoid()  # Ensure [0,1] output
        )
        
    def forward(self, z):
        # Project and reshape
        h = self.fc(z)
        h = h.view(-1, 256, 4, 4)
        
        # Progressive upsampling
        h = self.upsample1(h)
        h = self.upsample2(h)
        h = self.upsample3(h)
        h = self.upsample4(h)
        
        # Final refinement
        output = self.final_conv(h)
        
        return output

# ============================================================================
# 4. MODIFIED LOSS COMPUTATION FOR YOUR MODEL
# ============================================================================

def apply_pathology_fixes_to_your_model(model, device='cuda' if torch.cuda.is_available() else 'mps'):
    """Apply these fixes to your existing model"""
    
    # Add pathology reconstruction loss
    model.pathology_recon_loss = PathologyReconstructionLoss(device=device)
    
    # Replace sparsity layers with pathology-specific ones
    if hasattr(model, 'semantic_factorization'):
        model.semantic_factorization = PathologySparsityLayer(
            model.disentangle_config.semantic_dim,
            model.disentangle_config.n_semantic_factors,
            target_sparsity=0.2  # More selective for pathology
        ).to(device)
    
    if hasattr(model, 'attribute_factorization'):
        model.attribute_factorization = PathologySparsityLayer(
            model.disentangle_config.attribute_dim,
            model.disentangle_config.n_attribute_factors,
            target_sparsity=0.2
        ).to(device)
    
    # Override the loss computation
    original_compute_losses = model.compute_disentanglement_losses
    
    def pathology_loss_computation(x, outputs):
        """Enhanced loss computation for pathology"""
        device = x.device
        losses = {}
        
        # 1. Enhanced reconstruction loss
        if 'recon_x' in outputs:
            enhanced_recon, recon_components = model.pathology_recon_loss(outputs['recon_x'], x)
            losses['enhanced_reconstruction'] = enhanced_recon
            
            # Add component losses for monitoring
            for name, value in recon_components.items():
                losses[f'recon_{name}'] = value
        
        # 2. Much smaller KL loss for pathology
        if 'semantic_mu' in outputs and 'semantic_logvar' in outputs:
            semantic_kl = -0.5 * torch.sum(1 + outputs['semantic_logvar'] - 
                                         outputs['semantic_mu'].pow(2) - 
                                         outputs['semantic_logvar'].exp(), dim=1).mean()
            losses['semantic_kl'] = semantic_kl * 0.0001  # Very small weight
        
        if 'attribute_mu' in outputs and 'attribute_logvar' in outputs:
            attribute_kl = -0.5 * torch.sum(1 + outputs['attribute_logvar'] - 
                                          outputs['attribute_mu'].pow(2) - 
                                          outputs['attribute_logvar'].exp(), dim=1).mean()
            losses['attribute_kl'] = attribute_kl * 0.0001  # Very small weight
        
        # 3. Improved sparsity loss
        if 'semantic_factors' in outputs and 'attribute_factors' in outputs:
            # Target sparsity of 0.2 (more selective for pathology)
            semantic_sparsity = outputs['semantic_factors']
            attribute_sparsity = outputs['attribute_factors']
            
            # Entropy-based sparsity (encourages few active factors)
            semantic_entropy = -torch.sum(semantic_sparsity * torch.log(semantic_sparsity + 1e-8), dim=1).mean()
            attribute_entropy = -torch.sum(attribute_sparsity * torch.log(attribute_sparsity + 1e-8), dim=1).mean()
            
            # We want LOW entropy (concentrated activation)
            losses['sparsity_entropy'] = (semantic_entropy + attribute_entropy) * 0.01
        
        # 4. Factor diversity loss (ensure different factors for different pathologies)
        if 'semantic_factors' in outputs:
            # Encourage different factor patterns for different samples
            semantic_factors = outputs['semantic_factors']
            batch_size = semantic_factors.shape[0]
            
            if batch_size > 1:
                # MPS-compatible pairwise distance computation
                try:
                    distances = torch.pdist(semantic_factors, p=2)
                    diversity_loss = -distances.mean()  # Negative = encourage diversity
                except NotImplementedError:
                    # Fallback for MPS: manual pairwise distance computation
                    expanded_a = semantic_factors.unsqueeze(1)  # [batch, 1, factors]
                    expanded_b = semantic_factors.unsqueeze(0)  # [1, batch, factors]
                    distances = torch.norm(expanded_a - expanded_b, p=2, dim=2)  # [batch, batch]
                    # Get upper triangular part (excluding diagonal)
                    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                    if mask.sum() > 0:
                        diversity_loss = -distances[mask].mean()
                    else:
                        diversity_loss = torch.tensor(0.0, device=device)
                
                losses['factor_diversity'] = diversity_loss * 0.001
            else:
                losses['factor_diversity'] = torch.tensor(0.0, device=device)
        
        # 5. Total loss
        losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        
        return losses
    
    # Replace the method
    model.compute_disentanglement_losses = pathology_loss_computation
    
    return model

# ============================================================================
# 6. COMPLETE INTEGRATION SCRIPT
# ============================================================================

import os
import glob
from pathlib import Path

def create_pathology_vae_model(config, device='cuda' if torch.cuda.is_available() else 'mps'):
    """
    Create a complete pathology-optimized VAE model
    """
    from pythae.models import VAEConfig
    from pythae.models.base.base_utils import ModelOutput
    
    # Update config for pathology
    pathology_config = DisentangledVAEConfig(
        input_channels=3,
        input_size=config.get('input_size', 64),
        semantic_dim=20,  # More dimensions for complex pathology
        attribute_dim=12,
        n_semantic_factors=40,  # More factors for diverse pathology types
        n_attribute_factors=20,
        batch_size=16,  # Smaller batches for diversity
        learning_rate=5e-4,
        epochs=150,
        # Critical: much smaller KL weight
        kl_weight=0.0001,
        sparsity_weight=0.01,
        factorization_weight=0.001,
        orthogonality_weight=0.001,
        dataset_name="pathmnist"
    )
    
    # Create base VAE config for Pythae
    vae_config = VAEConfig(
        input_dim=(3, pathology_config.input_size, pathology_config.input_size),
        latent_dim=pathology_config.semantic_dim + pathology_config.attribute_dim
    )
    
    # Add pathology-specific attributes
    vae_config.input_channels = 3
    vae_config.input_size = pathology_config.input_size
    vae_config.semantic_dim = pathology_config.semantic_dim
    vae_config.attribute_dim = pathology_config.attribute_dim
    
    # Create pathology-specific encoder/decoder
    encoder = PathologyEncoder(
        input_channels=3,
        input_size=pathology_config.input_size,
        semantic_dim=pathology_config.semantic_dim,
        attribute_dim=pathology_config.attribute_dim
    )
    
    decoder = PathologyDecoder(
        semantic_dim=pathology_config.semantic_dim,
        attribute_dim=pathology_config.attribute_dim,
        output_channels=3,
        output_size=pathology_config.input_size
    )
    
    # Wrap for Pythae compatibility
    class PythaeEncoderWrapper(BaseEncoder):
        def __init__(self, pathology_encoder):
            super().__init__()
            self.pathology_encoder = pathology_encoder
            
        def forward(self, x):
            output = self.pathology_encoder(x)
            combined_mu = torch.cat([output['semantic_mu'], output['attribute_mu']], dim=1)
            combined_logvar = torch.cat([output['semantic_logvar'], output['attribute_logvar']], dim=1)
            
            return ModelOutput(
                embedding=combined_mu,
                log_covariance=combined_logvar,
                semantic_mu=output['semantic_mu'],
                semantic_logvar=output['semantic_logvar'],
                attribute_mu=output['attribute_mu'],
                attribute_logvar=output['attribute_logvar']
            )
    
    class PythaeDecoderWrapper(BaseDecoder):
        def __init__(self, pathology_decoder):
            super().__init__()
            self.pathology_decoder = pathology_decoder
            
        def forward(self, z):
            reconstruction = self.pathology_decoder(z)
            return ModelOutput(reconstruction=reconstruction)
    
    # Create the model
    model = FactorizedDisentangledVAE(
        model_config=vae_config,
        encoder=PythaeEncoderWrapper(encoder),
        decoder=PythaeDecoderWrapper(decoder),
        disentangle_config=pathology_config
    )
    
    # Apply pathology fixes
    model = apply_pathology_fixes_to_your_model(model, device)
    
    return model, pathology_config

# ============================================================================
# 7. ENHANCED TRAINING LOOP FOR PATHOLOGY
# ============================================================================

class PathologyVAETrainer:
    """Specialized trainer for pathology VAEs"""
    
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'mps'):
        self.model = model
        self.config = config
        self.device = device
        self.best_loss = float('inf')
        
    def train(self, train_loader, val_loader, epochs=150):
        """Training with pathology-specific adjustments"""
        
        # Optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.decoder.parameters(), 'lr': self.config.learning_rate * 2},  # Higher LR for decoder
            {'params': self.model.semantic_factorization.parameters(), 'lr': self.config.learning_rate * 0.5},
            {'params': self.model.attribute_factorization.parameters(), 'lr': self.config.learning_rate * 0.5}
        ], weight_decay=0.01)
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = defaultdict(float)
            
            for batch_idx, batch in enumerate(train_loader):
                # Get data
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                    labels = batch.get('label', None)
                else:
                    x, labels = batch
                    x = x.to(self.device)
                
                # Forward pass
                outputs = self.model({'data': x, 'label': labels})
                
                # Backward pass
                optimizer.zero_grad()
                outputs.loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track losses
                for key in ['loss', 'enhanced_reconstruction', 'semantic_kl', 'attribute_kl', 
                           'sparsity_entropy', 'factor_diversity']:
                    if hasattr(outputs, key):
                        train_losses[key] += getattr(outputs, key).item()
                
                # Log progress
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {outputs.loss.item():.4f}")
                    
                    # Check reconstruction quality
                    with torch.no_grad():
                        recon_error = F.mse_loss(outputs.recon_x, x).item()
                        print(f"  Reconstruction MSE: {recon_error:.4f}")
                        
                        # Check factor sparsity
                        if hasattr(outputs, 'semantic_factors'):
                            semantic_sparsity = (outputs.semantic_factors < 0.1).float().mean().item()
                            print(f"  Semantic sparsity: {semantic_sparsity:.3f}")
            
            # Validation phase
            if epoch % 5 == 0:
                val_loss = self.validate(val_loader)
                print(f"\nEpoch {epoch} - Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch)
                    
                # Visualize reconstructions
                self.visualize_reconstructions(val_loader, epoch)
            
            scheduler.step()
    
    def validate(self, val_loader):
        """Validation with focus on reconstruction quality"""
        self.model.eval()
        total_loss = 0
        total_recon_error = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['data'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                outputs = self.model({'data': x})
                total_loss += outputs.loss.item()
                
                # Detailed reconstruction error
                recon_error = F.mse_loss(outputs.recon_x, x).item()
                total_recon_error += recon_error
        
        avg_loss = total_loss / len(val_loader)
        avg_recon = total_recon_error / len(val_loader)
        
        print(f"  Average reconstruction error: {avg_recon:.4f}")
        
        return avg_loss
    
    def visualize_reconstructions(self, val_loader, epoch):
        """Visualize pathology reconstructions with detail comparison"""
        self.model.eval()
        
        # Get a batch
        batch = next(iter(val_loader))
        if isinstance(batch, dict):
            x = batch['data'][:8].to(self.device)
        else:
            x = batch[0][:8].to(self.device)
        
        with torch.no_grad():
            outputs = self.model({'data': x})
            
            # Create detailed comparison
            fig, axes = plt.subplots(4, 8, figsize=(24, 12))
            
            for i in range(8):
                # Original
                img = x[i].cpu().permute(1, 2, 0).numpy()
                axes[0, i].imshow(img)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstruction
                recon = outputs.recon_x[i].cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(recon)
                axes[1, i].set_title('Reconstruction')
                axes[1, i].axis('off')
                
                # Difference map
                diff = np.abs(img - recon)
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'Error: {diff.mean():.3f}')
                axes[2, i].axis('off')
                
                # Factor activations
                if hasattr(outputs, 'semantic_factors'):
                    factors = outputs.semantic_factors[i].cpu().numpy()
                    axes[3, i].bar(range(len(factors)), factors)
                    axes[3, i].set_ylim(0, 1)
                    axes[3, i].set_title(f'Active: {(factors > 0.1).sum()}')
            
            plt.tight_layout()
            plt.savefig(f'pathology_recon_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        torch.save(checkpoint, f'pathology_vae_best.pth')
        print(f"Saved best model at epoch {epoch}")

# ============================================================================
# 8. ANALYSIS TOOLS FOR PATHOLOGY FACTORS
# ============================================================================

def analyze_pathology_factors(model, test_loader, device='cuda' if torch.cuda.is_available() else 'mps'):
    """Analyze what pathology features each factor captures"""
    
    model.eval()
    
    # Collect factor activations and corresponding images
    all_factors = []
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['data'].to(device)
                labels = batch.get('label', torch.zeros(x.shape[0]))
            else:
                x, labels = batch
                x = x.to(device)
            
            outputs = model({'data': x})
            
            if hasattr(outputs, 'semantic_factors'):
                all_factors.append(outputs.semantic_factors.cpu())
                all_images.append(x.cpu())
                all_labels.append(labels.cpu())
    
    all_factors = torch.cat(all_factors)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    
    # Find prototype images for each factor
    n_factors = all_factors.shape[1]
    n_prototypes = 5
    
    fig, axes = plt.subplots(n_factors, n_prototypes, figsize=(15, 3*n_factors))
    
    for factor_idx in range(min(n_factors, 10)):  # Show first 10 factors
        # Get top activated images for this factor
        factor_activations = all_factors[:, factor_idx]
        top_indices = torch.argsort(factor_activations, descending=True)[:n_prototypes]
        
        for j, idx in enumerate(top_indices):
            img = all_images[idx].permute(1, 2, 0).numpy()
            axes[factor_idx, j].imshow(img)
            axes[factor_idx, j].set_title(f'Act: {factor_activations[idx]:.3f}')
            axes[factor_idx, j].axis('off')
        
        # Label the factor
        axes[factor_idx, 0].set_ylabel(f'Factor {factor_idx}', rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig('pathology_factor_prototypes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute factor-pathology correlations
    unique_labels = torch.unique(all_labels)
    factor_label_corr = torch.zeros(n_factors, len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        for j in range(n_factors):
            # Average activation for this pathology type
            avg_activation = all_factors[mask, j].mean()
            factor_label_corr[j, i] = avg_activation
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(factor_label_corr.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Average Activation')
    plt.xlabel('Pathology Type')
    plt.ylabel('Factor Index')
    plt.title('Factor-Pathology Correlation Matrix')
    plt.tight_layout()
    plt.savefig('factor_pathology_correlation.png', dpi=150)
    plt.show()
    
    return factor_label_corr

# ============================================================================
# 9. MAIN EXECUTION FOR PATHOLOGY
# ============================================================================

def main_pathology():
    """Main execution for pathology VAE"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating pathology-optimized VAE...")
    model, config = create_pathology_vae_model({}, device)
    model = model.to(device)
    
    # Load PathMNIST data
    print("Loading PathMNIST dataset...")
    train_data, test_data, train_labels, test_labels, num_classes = load_dataset(
        'pathmnist',
        input_size=64,
        train_size=5000,  # Use subset for faster testing
        test_size=1000
    )
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Train model
    print("Training pathology VAE...")
    trainer = PathologyVAETrainer(model, config, device)
    trainer.train(train_loader, test_loader, epochs=50)  # Reduced for testing
    
    # Analyze results
    print("Analyzing learned factors...")
    factor_correlations = analyze_pathology_factors(model, test_loader, device)
    
    print("\nTraining complete!")
    print("Check the generated images for reconstruction quality and factor analysis.")

# ============================================================================
# MEDMNIST-SPECIFIC FIXES FOR VAE
# ============================================================================

class MedMNISTPreprocessing:
    """Correct preprocessing for MedMNIST datasets"""
    
    @staticmethod
    def analyze_dataset_stats(dataset_name='pathmnist'):
        """Get actual statistics from MedMNIST data"""
        # MedMNIST datasets have different stats per dataset
        medmnist_stats = {
            'pathmnist': {
                'mean': [0.7404, 0.5311, 0.7058],  # Actual PathMNIST stats
                'std': [0.1944, 0.2420, 0.1595]
            },
            'chestmnist': {
                'mean': [0.4975],  # Grayscale
                'std': [0.2485]
            },
            'dermamnist': {
                'mean': [0.7635, 0.5461, 0.5705],
                'std': [0.1408, 0.1529, 0.1691]
            },
            'octmnist': {
                'mean': [0.4975],  # Grayscale  
                'std': [0.2485]
            },
            'pneumoniamnist': {
                'mean': [0.4975],  # Grayscale
                'std': [0.2485]
            },
            'retinamnist': {
                'mean': [0.4134, 0.2607, 0.1329],
                'std': [0.2866, 0.1899, 0.1045]
            }
        }
        return medmnist_stats.get(dataset_name, {'mean': [0.5], 'std': [0.5]})
    
    @staticmethod
    def get_medmnist_transforms(dataset_name='pathmnist', input_size=64, normalize=True):
        """Correct transforms for MedMNIST (NOT ImageNet!)"""
        
        # Get dataset-specific stats
        stats = MedMNISTPreprocessing.analyze_dataset_stats(dataset_name)
        
        transforms_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
        
        # Apply MedMNIST-specific normalization
        if normalize:
            transforms_list.append(
                transforms.Normalize(mean=stats['mean'], std=stats['std'])
            )
        
        return transforms.Compose(transforms_list)
    
    @staticmethod
    def denormalize_medmnist(tensor, dataset_name='pathmnist'):
        """Denormalize MedMNIST tensors for visualization"""
        stats = MedMNISTPreprocessing.analyze_dataset_stats(dataset_name)
        
        mean = torch.tensor(stats['mean']).view(-1, 1, 1)
        std = torch.tensor(stats['std']).view(-1, 1, 1)
        
        return tensor * std + mean

class MedicalReconstructionLoss(nn.Module):
    """Reconstruction loss tailored for medical images"""
    
    def __init__(self, focal_weight=0.1):
        super().__init__()
        self.focal_weight = focal_weight
        
    def forward(self, recon, target):
        # 1. MSE Loss (primary for medical images)
        mse_loss = F.mse_loss(recon, target, reduction='mean')
        
        # 2. L1 Loss (helps with fine details in medical images)
        l1_loss = F.l1_loss(recon, target, reduction='mean')
        
        # 3. Focal Loss component (emphasizes hard-to-reconstruct regions)
        diff = torch.abs(recon - target)
        focal_loss = torch.mean(diff ** 2 * (1 - torch.exp(-diff)))
        
        # Combine losses
        total_loss = (
            0.6 * mse_loss + 
            0.3 * l1_loss + 
            self.focal_weight * focal_loss
        )
        
        return total_loss, {
            'mse': mse_loss.detach(),
            'l1': l1_loss.detach(),
            'focal': focal_loss.detach()
        }

class MedMNISTSparsityLayer(nn.Module):
    """Improved sparsity layer for MedMNIST"""
    
    def __init__(self, input_dim, n_factors, target_sparsity=0.3):
        super().__init__()
        
        self.target_sparsity = target_sparsity
        self.n_factors = n_factors
        
        # Main factorization
        self.factorization = nn.Linear(input_dim, n_factors, bias=True)
        
        # Learnable sparsity threshold per factor
        self.thresholds = nn.Parameter(torch.zeros(n_factors))
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(n_factors)
        
    def forward(self, z):
        # Linear transformation
        factors = self.factorization(z)
        
        # Batch normalize for stability
        factors = self.bn(factors)
        
        # Apply learnable thresholds
        factors = factors - self.thresholds.unsqueeze(0)
        
        # ReLU activation
        factors = F.relu(factors)
        
        # Adaptive top-k sparsity based on current statistics
        if self.training:
            factors = self.adaptive_topk(factors)
        
        # Sigmoid to get probabilities
        return torch.sigmoid(factors)
    
    def adaptive_topk(self, x):
        """Adaptive top-k that actually achieves target sparsity"""
        current_active = (x > 0).float().mean()
        target_active = 1 - self.target_sparsity
        
        # Calculate k to achieve target sparsity
        k = max(1, int(self.n_factors * target_active))
        
        # Apply top-k per sample
        topk_vals, topk_indices = torch.topk(x, k, dim=1)
        
        # Create sparse representation
        sparse_factors = torch.zeros_like(x)
        sparse_factors.scatter_(1, topk_indices, topk_vals)
        
        return sparse_factors

# ============================================================================
# DEBUG AND UTILITY FUNCTIONS
# ============================================================================

def debug_current_training_issues(model, dataloader, device):
    """Debug what's actually wrong with your current training"""
    
    model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        if isinstance(batch, dict):
            x = batch['data'][:4].to(device)
        else:
            x = batch[0][:4].to(device)
        
        print("🔍 DEBUGGING YOUR CURRENT MODEL:")
        print(f"Input shape: {x.shape}")
        print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"Input mean: {x.mean():.3f}, std: {x.std():.3f}")
        
        # Forward pass
        outputs = model(x)
        
        print(f"\n📊 MODEL OUTPUTS:")
        print(f"Reconstruction range: [{outputs.recon_x.min():.3f}, {outputs.recon_x.max():.3f}]")
        print(f"Reconstruction mean: {outputs.recon_x.mean():.3f}")
        
        # Check losses
        recon_loss = F.mse_loss(outputs.recon_x, x)
        print(f"\n❌ CURRENT ISSUES:")
        print(f"Reconstruction loss: {recon_loss.item():.3f} (should be < 1.0)")
        
        if hasattr(outputs, 'semantic_factors'):
            sem_sparsity = (outputs.semantic_factors < 0.1).float().mean()
            print(f"Semantic sparsity: {sem_sparsity.item():.3f} (should be ~0.3)")
            
        if hasattr(outputs, 'attribute_factors'):
            attr_sparsity = (outputs.attribute_factors < 0.1).float().mean()
            print(f"Attribute sparsity: {attr_sparsity.item():.3f} (should be ~0.3)")
            
        print(f"\n✅ FIXES NEEDED:")
        if recon_loss.item() > 2.0:
            print("- Reconstruction loss too high: reduce KL weight to 0.0005")
        if hasattr(outputs, 'semantic_factors') and (outputs.semantic_factors < 0.1).float().mean() < 0.2:
            print("- Sparsity too low: increase sparsity weight to 0.005")
        
        return {
            'recon_loss': recon_loss.item(),
            'input_stats': {'mean': x.mean().item(), 'std': x.std().item()},
            'output_stats': {'mean': outputs.recon_x.mean().item(), 'std': outputs.recon_x.std().item()}
        }

def apply_immediate_fixes_to_model(model, config):
    """Apply immediate fixes to your existing model"""
    
    # Fix 1: Replace factorization layers if this is medical data
    is_medical = hasattr(config, 'dataset_name') and 'mnist' in config.dataset_name.lower() and config.dataset_name != 'mnist'
    
    if is_medical:
        print("🔧 Applying MedMNIST-specific fixes...")
        
        if hasattr(model, 'semantic_factorization'):
            model.semantic_factorization = MedMNISTSparsityLayer(
                config.semantic_dim,
                config.n_semantic_factors,
                target_sparsity=0.3
            ).to(next(model.parameters()).device)
        
        if hasattr(model, 'attribute_factorization'):
            model.attribute_factorization = MedMNISTSparsityLayer(
                config.attribute_dim,
                config.n_attribute_factors,
                target_sparsity=0.3
            ).to(next(model.parameters()).device)
        
        # Add medical reconstruction loss
        model.medical_recon_loss = MedicalReconstructionLoss().to(next(model.parameters()).device)
        model.use_medical_loss = True
        
        print("✅ Applied MedMNIST-specific improvements!")
    
    return model

def create_medmnist_training_config(dataset_name='pathmnist'):
    """Create optimized training configuration for MedMNIST"""
    
    # Use MedMNIST-specific config
    disentangle_config = MedMNISTVAEConfig(
        dataset_name=dataset_name,
        input_channels=3 if dataset_name in ['pathmnist', 'dermamnist', 'retinamnist'] else 1,
        input_size=64,
        semantic_dim=20,
        attribute_dim=12,
        n_semantic_factors=40,
        n_attribute_factors=20,
        batch_size=32,
        epochs=100,
        kl_weight=0.0005,  # Critical fix!
        sparsity_weight=0.005,  # Critical fix!
        orthogonality_weight=0.002
    )
    
    # Configure Pythae trainer with medical-optimized settings
    training_config = BaseTrainerConfig(
        output_dir=f'medmnist_{dataset_name}_output',
        num_epochs=disentangle_config.epochs,
        learning_rate=disentangle_config.learning_rate,
        per_device_train_batch_size=disentangle_config.batch_size,
        per_device_eval_batch_size=disentangle_config.batch_size,
        train_dataloader_num_workers=2,
        eval_dataloader_num_workers=2,
        steps_saving=10,
        optimizer_cls="Adam",
        optimizer_params={"weight_decay": 0.01, "betas": (0.9, 0.999)},
        scheduler_cls="StepLR",
        scheduler_params={"step_size": 50, "gamma": 0.5}
    )
    
    return disentangle_config, training_config

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main(dataset_name="mnist", use_medmnist_config=False):
    """Main execution function with support for MedMNIST datasets"""
    
    # Configuration - choose between standard and MedMNIST optimized
    if use_medmnist_config or (dataset_name != "mnist" and "mnist" in dataset_name.lower()):
        print(f"🏥 Using MedMNIST-optimized configuration for {dataset_name}")
        disentangle_config, training_config = create_medmnist_training_config(dataset_name)
    else:
        print(f"📊 Using standard configuration for {dataset_name}")
        # Standard configuration
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
            dataset_name=dataset_name
        )
        
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

def demo_medmnist_training():
    """Demonstration of MedMNIST training with all fixes applied"""
    
    print("🏥 MEDMNIST TRAINING DEMO WITH ALL FIXES")
    print("=" * 60)
    
    # Available MedMNIST datasets
    available_datasets = ['pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist']
    
    for dataset_name in ['pathmnist']:  # Demo with PathMNIST
        print(f"\n🔬 Training on {dataset_name.upper()}")
        
        try:
            # Run training with MedMNIST optimizations
            main(dataset_name=dataset_name, use_medmnist_config=True)
            print(f"✅ Successfully completed training on {dataset_name}")
            
        except Exception as e:
            print(f"❌ Error training on {dataset_name}: {e}")
            continue

# ============================================================================
# BALANCED WEIGHT STRATEGY DEMONSTRATION
# ============================================================================

def test_terminal_output():
    """Test if terminal output is working properly"""
    import sys
    import time
    
    print("🧪 TERMINAL OUTPUT TEST")
    print("=" * 40)
    sys.stdout.flush()
    
    # Test basic output
    for i in range(5):
        print(f"Test line {i+1}/5...")
        sys.stdout.flush()
        time.sleep(0.5)
    
    # Test tqdm
    print("\n📊 Testing progress bar...")
    sys.stdout.flush()
    
    for i in tqdm(range(10), desc="Progress Test", ncols=80):
        time.sleep(0.1)
    
    print("✅ Terminal output test completed!")
    print("If you can see this, terminal output is working properly.")
    sys.stdout.flush()

def demo_balanced_weight_strategy():
    """Demonstration of balanced weight strategy to prevent posterior collapse"""
    
    # First test terminal output
    test_terminal_output()
    
    print("\n⚖️  BALANCED WEIGHT STRATEGY DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how to prevent posterior collapse in VAEs using:")
    print("- Linear KL annealing (MPS-compatible)")
    print("- Aggressive free bits per dimension")
    print("- Conservative capacity constraints")
    print("- Emergency collapse recovery")
    print("- Real-time monitoring and adjustment")
    print()
    
    # Configuration - Use MNIST for MPS compatibility
    dataset_name = "mnist"  # Use MNIST for better MPS compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    print(f"🍎 Detected device: {device}")
    if device.type == 'mps':
        print("   Applying MPS-specific optimizations for Apple Silicon")
    print()
    
    # Create MPS-optimized config
    disentangle_config = DisentangledVAEConfig(
        dataset_name=dataset_name,
        input_channels=1,  # MNIST is grayscale
        input_size=28,     # MNIST native size
        semantic_dim=8,    # Smaller for MPS stability
        attribute_dim=4,   # Smaller for MPS stability
        n_semantic_factors=16,  # Much smaller for MPS
        n_attribute_factors=8,  # Much smaller for MPS
        batch_size=8,      # Very small batch for MPS
        epochs=10,         # Fewer epochs for demo
        kl_weight=0.0001,  # Very small target weight
        learning_rate=1e-4, # Smaller learning rate
        num_classes=10
    )
    
    print(f"🔧 MPS-Optimized Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Device: {device}")
    print(f"  Input size: {disentangle_config.input_size}x{disentangle_config.input_size}")
    print(f"  Semantic dimensions: {disentangle_config.semantic_dim}")
    print(f"  Attribute dimensions: {disentangle_config.attribute_dim}")
    print(f"  Semantic factors: {disentangle_config.n_semantic_factors}")
    print(f"  Attribute factors: {disentangle_config.n_attribute_factors}")
    print(f"  Batch size: {disentangle_config.batch_size}")
    print(f"  Target KL weight: {disentangle_config.kl_weight}")
    print()
    
    # Load dataset
    print("📊 Loading dataset...")
    train_data, test_data, train_labels, test_labels, num_classes = load_dataset(
        dataset_name,
        input_size=disentangle_config.input_size,
        train_size=500,  # Very small subset for MPS demo
        test_size=100
    )
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Classes: {num_classes}")
    print(f"  Data shape: {train_data.shape}")
    print(f"  Data range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    
    # Create model
    print("\n🏗️  Creating model...")
    model_config = VAEConfig(
        input_dim=(disentangle_config.input_channels, disentangle_config.input_size, disentangle_config.input_size),
        latent_dim=disentangle_config.semantic_dim + disentangle_config.attribute_dim
    )
    
    # Add disentanglement-specific config
    model_config.input_channels = disentangle_config.input_channels
    model_config.input_size = disentangle_config.input_size
    model_config.semantic_dim = disentangle_config.semantic_dim
    model_config.attribute_dim = disentangle_config.attribute_dim
    
    model = FactorizedDisentangledVAE(
        model_config=model_config,
        encoder=FlexibleDisentangledEncoder(model_config),
        decoder=FlexibleDisentangledDecoder(model_config),
        disentangle_config=disentangle_config
    )
    
    print(f"  Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Option 1: Apply quick fixes (minimal integration)
    print("\n🚀 Option 1: Quick Balanced Fixes")
    model_quick = apply_quick_balanced_fixes(model, disentangle_config)
    print("  ✅ Quick fixes applied:")
    print("    - Cyclical KL annealing")
    print("    - Free bits (minimum 2.0 bits per dimension)")
    print("    - Active dimension monitoring")
    
    # Option 2: Full integration (recommended)
    print("\n🎯 Option 2: Full Balanced Weight Integration")
    model_full = integrate_balanced_weights_with_model(model, disentangle_config)
    print("  ✅ Full integration applied:")
    print("    - Advanced cyclical annealing")
    print("    - Capacity constraints")
    print("    - Dynamic weight balancing")
    print("    - Automatic collapse detection")
    print("    - Training dynamics visualization")
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=disentangle_config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=disentangle_config.batch_size, shuffle=False)
    
    # Demonstrate training with balanced weights
    print("\n🏃 Training with Balanced Weights...")
    print("This will show real-time monitoring of:")
    print("- KL weight schedule")
    print("- Active dimension tracking")
    print("- Collapse detection")
    print("- Dynamic weight adjustments")
    print()
    
    try:
        # Train with full balanced weight strategy
        trained_model, history = train_with_balanced_weights(
            model_full, 
            train_loader, 
            test_loader, 
            disentangle_config, 
            epochs=disentangle_config.epochs,
            device=device
        )
        
        print("\n✅ Training completed successfully!")
        print(f"  Collapse events detected: {len(history['collapse_events'])}")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        # Show final statistics
        if hasattr(trained_model, 'balanced_loss'):
            scheduler = trained_model.balanced_loss.scheduler
            print(f"  Final KL weight: {scheduler.current_kl_weight:.6f}")
            print(f"  Final capacity: {scheduler.current_capacity:.2f}")
            
            if scheduler.loss_history['active_dims']:
                final_active = scheduler.loss_history['active_dims'][-1]
                print(f"  Final active dimensions: {final_active:.2%}")
        
        print("\n📈 Training dynamics plot saved as 'balanced_training_dynamics.png'")
        
    except Exception as e:
        print(f"❌ Training error: {e}")

if __name__ == "__main__":
    # You can run different configurations:
    
    # 1. Standard MNIST training
    # main(dataset_name="mnist", use_medmnist_config=False)
    
    # 2. MedMNIST training with optimizations (RECOMMENDED for medical data)
    # main(dataset_name="pathmnist", use_medmnist_config=True)
    
    # 3. Demo of MedMNIST training
    # demo_medmnist_training()
    
    # 4. PATHOLOGY-OPTIMIZED VAE (NEW - BEST FOR MEDICAL IMAGES)
    # main_pathology()
    
    # 5. BALANCED WEIGHT STRATEGY DEMO (NEW - PREVENTS POSTERIOR COLLAPSE)
    demo_balanced_weight_strategy()
    
    # Default: run balanced weight demo
    # main()

 