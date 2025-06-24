from disentangled_vae_pythae_integration import FactorizedDisentangledVAE, DisentangledVAETrainer, DisentangledVAETrainingPipeline
import torch
import torch.nn.functional as F

from pythae.pipelines.training import TrainingPipeline
from pythae.models.base.base_utils import ModelOutput

import numpy as np
import medmnist

class FactorizedDisentangledVAEWithAnnealing(FactorizedDisentangledVAE):
    """Extended VAE with beta annealing support"""
    
    def __init__(self, model_config, encoder=None, decoder=None, disentangle_config=None):
        super().__init__(model_config, encoder, decoder, disentangle_config)
        self.current_epoch = 0
        self.total_epochs = self.disentangle_config.epochs
        self.beta_start = self.disentangle_config.beta_start
        self.beta_end = self.disentangle_config.beta_end
        
    def set_epoch(self, epoch):
        """Set current epoch for beta annealing"""
        self.current_epoch = epoch
        
    def get_current_beta(self):
        """Calculate current beta value based on linear annealing"""
        if self.current_epoch >= self.total_epochs:
            return self.beta_end
        
        # Linear annealing
        progress = self.current_epoch / self.total_epochs
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        
        return beta
    
    def loss_function(self, recon_x, x, mu, log_var, z):
        """Override loss function to use annealed beta"""
        # Reconstruction loss
        if x.shape[1] == 1:  # Binary images (dSprites)
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]
        else:  # RGB images
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
        
        # KL divergence with annealed beta
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
        
        # Apply beta annealing
        beta = self.get_current_beta()
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
class DisentangledVAETrainerWithAnnealing(DisentangledVAETrainer):
    """Custom trainer that handles beta annealing"""
    
    def train_step(self, epoch: int):
        """Override train step to update model epoch"""
        # Update model's current epoch for beta annealing
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
            
            # Log current beta
            if hasattr(self.model, 'get_current_beta'):
                current_beta = self.model.get_current_beta()
                self.writer.add_scalar("training/beta", current_beta, epoch)
        
        # Call parent train step
        return super().train_step(epoch)
    
class DisentangledVAETrainingPipelineWithAnnealing(TrainingPipeline):
    """Custom training pipeline with beta annealing support"""
    
    def __call__(self, train_data, eval_data=None):
        """Run training with custom trainer"""
        
        # Create custom trainer with annealing
        trainer = DisentangledVAETrainerWithAnnealing(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            training_config=self.training_config
        )
        
        # Run training
        trainer.train()
        
        return trainer
    
def load_dataset(dataset_name, input_size=64, train_size=None, test_size=None):
    """Generic dataset loader that returns numpy arrays for Pythae compatibility"""
    
    if dataset_name == 'dsprites':
        # dSprites dataset loader
        # You need to implement this based on your dSprites data location
        # dSprites images are 64x64 binary (black and white)
        
        # Example structure (you need to fill in the actual loading code):
        """
        # Load dSprites data from your source
        # dsprites_data = np.load('path/to/dsprites.npz')
        # images = dsprites_data['imgs']  # Shape: (737280, 64, 64)
        # labels = dsprites_data['latents_values']  # Shape: (737280, 6)
        
        # Convert to float32 and add channel dimension
        # images = images.astype(np.float32)[:, np.newaxis, :, :]  # Shape: (N, 1, 64, 64)
        
        # Split into train/test (e.g., 90/10 split)
        # n_samples = len(images)
        # n_train = int(0.9 * n_samples)
        # indices = np.random.permutation(n_samples)
        # train_indices = indices[:n_train]
        # test_indices = indices[n_train:]
        """
        
        # Placeholder - replace with actual implementation
        raise NotImplementedError("""
        Please implement dSprites dataset loader. Example:
        
        # Load dSprites NPZ file
        data = np.load('path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        images = data['imgs'].astype(np.float32)
        
        # Add channel dimension (N, 64, 64) -> (N, 1, 64, 64)
        images = images[:, np.newaxis, :, :]
        
        # Create labels (use latent factors or indices)
        labels = np.arange(len(images))  # or use data['latents_values']
        
        # Split and sample
        ...
        """)
        
    elif dataset_name == 'xyobject':
        # XYObject dataset loader
        # RGB images, 64x64
        
        # Placeholder - replace with actual implementation
        raise NotImplementedError("""
        Please implement XYObject dataset loader. Example:
        
        # Load XYObject data
        images = load_xyobject_images()  # Should return (N, 3, 64, 64) RGB images
        labels = load_xyobject_labels()  # Factor labels or indices
        
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Split and sample
        ...
        """)
        
    elif dataset_name == 'shapes3d':
        # 3D Shapes dataset loader
        # RGB images with lighting, 64x64
        
        # Placeholder - replace with actual implementation
        raise NotImplementedError("""
        Please implement 3D Shapes dataset loader. Example:
        
        # Load 3D Shapes data
        import h5py
        with h5py.File('path/to/3dshapes.h5', 'r') as f:
            images = f['images'][:]  # Shape: (480000, 64, 64, 3)
            labels = f['labels'][:]  # Shape: (480000, 6)
        
        # Convert to channels-first format and normalize
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        
        # Split and sample
        ...
        """)
        
    elif dataset_name == 'mnist':
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
        raise ValueError(f"Dataset {dataset_name} not supported")