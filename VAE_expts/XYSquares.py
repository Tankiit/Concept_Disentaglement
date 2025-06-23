import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from weightwatcher import WeightWatcher


class ColoredSquaresDataset(torch.utils.data.Dataset):
    """Synthetic dataset with known ground truth factors"""
    def __init__(self, num_samples=1000, img_size=32):
        self.num_samples = num_samples
        self.img_size = img_size
        self.factors = np.zeros((num_samples, 4))  # [x_pos, y_pos, color_r, color_g]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random factors
        x_pos = np.random.uniform(0.1, 0.9)
        y_pos = np.random.uniform(0.1, 0.9)
        color_r = np.random.uniform(0.2, 1.0)
        color_g = np.random.uniform(0.2, 1.0)
        color_b = 0.0  # Fixed blue channel
        
        # Store factors
        self.factors[idx] = [x_pos, y_pos, color_r, color_g]
        
        # Create image
        img = np.zeros((3, self.img_size, self.img_size))
        square_size = int(self.img_size * 0.2)
        
        # Calculate square position
        x_start = int((self.img_size - square_size) * x_pos)
        y_start = int((self.img_size - square_size) * y_pos)
        
        # Draw colored square
        img[0, y_start:y_start+square_size, x_start:x_start+square_size] = color_r
        img[1, y_start:y_start+square_size, x_start:x_start+square_size] = color_g
        img[2, y_start:y_start+square_size, x_start:x_start+square_size] = color_b
        
        return {
            'image': torch.tensor(img, dtype=torch.float32),
            'factors': torch.tensor([x_pos, y_pos, color_r, color_g], dtype=torch.float32)
        }

# ============================================================================
# IMPROVED ARCHITECTURE FOR COLORED SQUARES
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ImprovedEncoder(nn.Module):
    """Enhanced encoder with residual blocks"""
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Latent space projection
        self.fc = nn.Linear(128*4*4, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ImprovedDecoder(nn.Module):
    """Enhanced decoder with skip connections"""
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128*4*4)
        
        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)  # 4x4 -> 8x8
        self.bn1 = nn.BatchNorm2d(128)
        
        self.res_blocks1 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 8x8 -> 16x16
        self.bn2 = nn.BatchNorm2d(64)
        
        self.res_blocks2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.upconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16x16 -> 32x32
        self.bn3 = nn.BatchNorm2d(32)
        
        self.final_conv = nn.Conv2d(32, 3, 3, padding=1)
        
        # Skip connections
        self.skip1 = nn.Conv2d(128, 128, 1)
        self.skip2 = nn.Conv2d(64, 64, 1)
    
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 128, 4, 4)
        
        # First upsampling
        x = F.relu(self.bn1(self.upconv1(x)))
        x = self.res_blocks1(x)
        
        # Second upsampling
        x = F.relu(self.bn2(self.upconv2(x)))
        x = self.res_blocks2(x)
        
        # Final upsampling
        x = F.relu(self.bn3(self.upconv3(x)))
        x = torch.sigmoid(self.final_conv(x))
        return x

class ImprovedVAE(nn.Module):
    """Complete improved VAE for colored squares"""
    def __init__(self, latent_dim=8, concept_dim=None):
        super().__init__()
        self.encoder = ImprovedEncoder(latent_dim)
        self.decoder = ImprovedDecoder(latent_dim)
        
        # Optional concept bottleneck
        self.concept_dim = concept_dim
        if concept_dim:
            self.concept_layer = nn.Sequential(
                nn.Linear(latent_dim, concept_dim),
                nn.Sigmoid()
            )
            self.inverse_layer = nn.Linear(concept_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        if self.concept_dim:
            concepts = self.concept_layer(z)
            z_recon = self.inverse_layer(concepts)
            x_recon = self.decoder(z_recon)
            return x_recon, mu, logvar, z, concepts, z_recon
        else:
            x_recon = self.decoder(z)
            return x_recon, mu, logvar, z

# ============================================================================
# ADAPTIVE LOSS BALANCER
# ============================================================================

class AdaptiveLossBalancer(nn.Module):
    """Dynamically balances loss terms using uncertainty weighting"""
    def __init__(self, num_losses, init_weights=None):
        super().__init__()
        # Learnable log-variance parameters
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
        # Initialize with sensible weights if provided
        if init_weights:
            for i, w in enumerate(init_weights):
                self.log_vars.data[i] = np.log(1/w)
    
    def forward(self, losses):
        """
        losses: list of loss tensors in fixed order
        Returns: weighted loss sum
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

# ============================================================================
# ENHANCED LOSS FUNCTION
# ============================================================================

class EnhancedVAELoss(nn.Module):
    """Custom loss for colored squares reconstruction"""
    def __init__(self, alpha=0.8, beta=0.001):
        super().__init__()
        self.alpha = alpha  # Weight for color loss
        self.beta = beta     # KL weight
    
    def forward(self, recon_x, x, mu, logvar):
        # Position-sensitive reconstruction loss
        # Create a mask for the square region
        square_mask = (x > 0.1).float()
        
        # MSE loss with emphasis on square region
        mse_loss = F.mse_loss(recon_x, x, reduction='none')
        position_loss = (mse_loss * square_mask).mean()
        
        # Color-specific loss (only on square pixels)
        color_loss = F.l1_loss(
            recon_x * square_mask, 
            x * square_mask,
            reduction='mean'
        )
        
        # Combined reconstruction loss
        recon_loss = position_loss + self.alpha * color_loss
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss.mean()
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

# ============================================================================
# MODIFIED TRAINING FUNCTION
# ============================================================================

def train_improved_model(model, dataloader, epochs=50, lr=1e-3, has_concepts=False, device='cpu'):
    """Simplified training function - use train_guaranteed for advanced features"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            
            if has_concepts:
                x_recon, mu, logvar, z, concepts, z_recon = model(images)
                # Simple concept consistency loss
                consistency_loss = F.mse_loss(z, z_recon)
            else:
                x_recon, mu, logvar, z = model(images)
                consistency_loss = torch.tensor(0.0, device=device)
                
            # Basic VAE loss
            recon_loss = F.mse_loss(x_recon, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()
            
            # Combined loss
            total_loss_tensor = recon_loss + 0.001 * kl_loss
            if has_concepts:
                total_loss_tensor += 0.01 * consistency_loss
                
            total_loss_tensor.backward()
            optimizer.step()
            total_loss += total_loss_tensor.item()
            
        # Print epoch info
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    return model

# ============================================================================
# HIGH PRECISION VAE WITH GUARANTEED RECONSTRUCTION
# ============================================================================

class HighPrecisionVAE(nn.Module):
    """Fixed architecture with guaranteed gradient flow"""
    def __init__(self, latent_dim=16, concept_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim
        
        # Encoder (simplified but effective)
        self.encoder = nn.Sequential(
            CoordConv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Latent projection
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        
        # Concept bottleneck
        self.concept_layer = nn.Sequential(
            nn.Linear(latent_dim, concept_dim),
            nn.Sigmoid()
        )
        self.inverse_layer = nn.Linear(concept_dim, latent_dim)
        
        # ===================================================================
        # FIXED DECODER ARCHITECTURE (GRADIENT-GUARANTEED)
        # ===================================================================
        self.decoder = FixedDecoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Concept bottleneck
        concepts = self.concept_layer(z)
        z_recon = self.inverse_layer(concepts)
        
        # Decode - using fixed architecture
        x_recon = self.decoder(z_recon)
        return x_recon, mu, logvar, z, concepts, z_recon

# ========== SPECIALIZED MODULES ==========
class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with gated mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        gate = self.sigmoid(self.gate(x))
        out = out * gate + residual
        return F.relu(out)

class CoordConv2d(nn.Module):
    """Adds coordinate channels to preserve spatial information"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.add_coords = AddCoords()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        
    def forward(self, x):
        x = self.add_coords(x)
        return self.conv(x)

class AddCoords(nn.Module):
    """Adds x and y coordinate channels"""
    def forward(self, x):
        batch, _, height, width = x.size()
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch, 1, height, width)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch, 1, height, width)
        return torch.cat([x, x_coords, y_coords], dim=1)

class SpatialAttention(nn.Module):
    """Focuses on square position"""
    def __init__(self, channels):
        super().__init__()
        # Initialize with small values
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(1))  # Start at 1
        
        # Proper initialization
        for conv in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0.1)

    def forward(self, x):
        # Add residual connection
        return x + self.gamma * self._attention(x)  # Residual added
    
    def _attention(self, x):
        batch, C, H, W = x.size()
        query = self.query(x).view(batch, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H*W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        value = self.value(x).view(batch, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        return out

class ColorAttention(nn.Module):
    """Preserves color information"""
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels//4),
            nn.ReLU(),
            nn.Linear(channels//4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention = self.fc(x)
        attention = attention.view(x.size(0), x.size(1), 1, 1)
        return x * attention + x

class FixedDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256):
        super().__init__()
        # Multiple skip connections at different scales
        self.direct_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.skip_fc_16 = nn.Linear(latent_dim, 32 * 16 * 16)  # Skip to 16x16
        self.skip_fc_32 = nn.Linear(latent_dim, 16 * 32 * 32)  # Skip to 32x32
        
        # Main path with enhanced gradient flow
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        
        # Use GroupNorm instead of BatchNorm for better gradient flow
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim + 64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),  # GroupNorm instead of BatchNorm
            nn.LeakyReLU(0.2, inplace=False)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 32, 64, 4, 2, 1),  # +32 for skip connection
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=False)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64 + 16, 32, 3, padding=1),  # +16 for skip connection
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 3, 1)
        )
        
        # Proper initialization for better gradient flow
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                # Xavier initialization for better gradient flow
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, z):
        # Multiple skip connections
        skip_8 = self.direct_fc(z).view(-1, 64, 8, 8)
        skip_16 = self.skip_fc_16(z).view(-1, 32, 16, 16)
        skip_32 = self.skip_fc_32(z).view(-1, 16, 32, 32)
        
        # Main path with residual connections
        h = self.fc(z).view(-1, 256, 4, 4)
        h = F.interpolate(h, scale_factor=2)  # To 8x8
        h = torch.cat([h, skip_8], dim=1)
        
        h = self.up1(h)  # 16x16
        h = torch.cat([h, skip_16], dim=1)  # Add skip connection
        
        h = self.up2(h)  # 32x32
        h = torch.cat([h, skip_32], dim=1)  # Add skip connection
        
        # Don't use sigmoid here - let the loss function handle it
        return self.final(h)

# ============================================================================
# FIXED LOSS COMPUTATION
# ============================================================================

def compute_fixed_loss(x, outputs):
    batch_size = x.size(0)
    
    # 1. Multi-scale reconstruction loss
    recon = outputs['x_recon']
    
    # Pixel-level losses
    l1_loss = F.l1_loss(recon, x, reduction='mean')
    mse_loss = F.mse_loss(recon, x, reduction='mean')
    
    # Perceptual loss (simplified - using downsampled versions)
    x_down = F.avg_pool2d(x, 4)
    recon_down = F.avg_pool2d(recon, 4)
    perceptual_loss = F.mse_loss(recon_down, x_down)
    
    # Combined reconstruction (weighted for XYSquares simplicity)
    recon_loss = 0.5 * l1_loss + 0.3 * mse_loss + 0.2 * perceptual_loss
    
    # 2. KL with free bits (prevent collapse)
    kl = -0.5 * torch.sum(
        1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
    ) / batch_size
    
    # Free bits: minimum KL per dimension
    kl_per_dim = kl / outputs['mu'].shape[1]
    kl_loss = torch.maximum(kl_per_dim * outputs['mu'].shape[1], 
                           torch.tensor(3.0, device=x.device))  # Minimum total KL = 3.0
    
    # 3. Concept loss with target activation
    concept_mean = outputs['concepts'].mean()
    target_activation = 0.25  # 25% sparse
    concept_loss = (concept_mean - target_activation).pow(2)
    
    # 4. REMOVE orthogonality for now (it's not helping)
    
    # Fixed weights (no adaptive nonsense)
    total_loss = (
        1000.0 * recon_loss +    # Much higher!
        0.01 * kl_loss +         # Small but non-zero
        1.0 * concept_loss       # Reasonable
    )
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'concept_loss': concept_loss,
        'l1': l1_loss,
        'mse': mse_loss
    }


# ============================================================================
# GUARANTEED TRAINING FUNCTION
# ============================================================================

def train_guaranteed(model, train_loader, epochs=50, device='mps'):
    # Initialize decoder weights properly
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
    model.decoder.apply(weights_init)

    """Training with guaranteed reconstruction quality and weight monitoring"""
    # 1. Optimizer with higher learning rate for decoder
    decoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)
    
    # Balanced learning rates
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': 8e-3},   # Higher for encoder
        {'params': decoder_params, 'lr': 6e-3}  # Slightly lower for decoder
    ], weight_decay=1e-6)  # Reduced weight decay
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=8e-3,  # Match the higher encoder LR
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3  # Longer warmup
    )
    
    # Balanced gradient scaling
    def scale_decoder_gradients(grad):
        return grad * 5.0  # Reduced from 10x
    
    def scale_encoder_gradients(grad):
        return grad * 15.0  # Much higher for encoder
    
    def scale_latent_gradients(grad):
        return grad * 8.0  # Boost latent space gradients
    
    # Register gradient hooks with better targeting
    hooks = []
    for name, param in model.named_parameters():
        if 'decoder' in name and param.requires_grad:
            hook = param.register_hook(scale_decoder_gradients)
            hooks.append(hook)
        elif 'encoder' in name and param.requires_grad:
            hook = param.register_hook(scale_encoder_gradients)
            hooks.append(hook)
        elif any(x in name for x in ['fc_mu', 'fc_logvar', 'concept_layer', 'inverse_layer']) and param.requires_grad:
            hook = param.register_hook(scale_latent_gradients)
            hooks.append(hook)
    
    # Initialize WeightWatcher and tracking variables
    watcher = WeightWatcher()
    weight_metrics = {
        'epoch': [],
        'global_alpha': [],
        'global_stable_rank': [],
        'global_norm': [],
        'encoder_alpha': [],
        'decoder_alpha': [],
        'concept_alpha': [],
        'latent_alpha': []
    }
    
    # 2. Custom loss function with gradient-friendly design
    def guaranteed_loss(recon_logits, original, mu, logvar, z, z_recon, epoch):
        # Apply sigmoid to logits for reconstruction
        recon = torch.sigmoid(recon_logits)
        
        # Create binary mask for foreground
        mask = (original > 0.1).float()
        
        # Use BCE loss for better gradients (since we have logits)
        bce_loss = F.binary_cross_entropy_with_logits(recon_logits, original, reduction='none')
        position_loss = (bce_loss * mask).mean()
        
        # Color-specific loss (only on square) - use L1 on sigmoid output
        color_loss = F.l1_loss(recon * mask, original * mask)
        
        # Edge preservation with reduced weight
        grad_original = torch.abs(original - F.avg_pool2d(original, 3, stride=1, padding=1))
        grad_recon = torch.abs(recon - F.avg_pool2d(recon, 3, stride=1, padding=1))
        edge_loss = F.mse_loss(grad_recon * mask, grad_original * mask)
        
        # KL with annealing - reduce weight further
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss.mean()
        kl_weight = min(1.0, epoch/50) * 0.0001  # Much slower annealing, smaller weight
        
        # Add latent normalization with smaller weight
        latent_norm = torch.norm(z, dim=1).mean()
        latent_reg = 0.001 * F.mse_loss(latent_norm, torch.tensor(1.0, device=z.device))
        
        # Concept consistency with smaller weight
        consistency_loss = F.mse_loss(z, z_recon)
        
        return (
            100.0 * position_loss +   # 10x higher! Force strong gradients
            50.0 * color_loss +       # 10x higher for color accuracy
            1.0 * edge_loss +         # Increased edge weight
            kl_weight * kl_loss +
            0.1 * consistency_loss +  # Increased consistency weight
            latent_reg
        )
    
    # 3. Training loop with monitoring
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_grad_check_done = False
        
        for batch in train_loader:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar, z, concepts, z_recon = model(images)
            
            # Compute loss
            loss = guaranteed_loss(x_recon, images, mu, logvar, z, z_recon, epoch)
            
            # Backprop
            loss.backward()
            
            # Gradient sanity check - first batch of each epoch
            if not epoch_grad_check_done:
                print(f"\n=== Gradient Sanity Check - Epoch {epoch+1} ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad_norm = {param.grad.norm().item():.4f}")
                print("==========================================\n")
                epoch_grad_check_done = True
            
            # Reduced gradient clipping to allow larger gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        # Print epoch info
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f}, LR={current_lr:.6f}")
        
        # ===================================================================
        # WEIGHT WATCHER ANALYSIS (AT END OF EACH EPOCH)
        # ===================================================================
        model.eval()
        try:
            details = watcher.analyze(model=model, layers=[nn.Conv2d, nn.ConvTranspose2d, nn.Linear])
            summary = watcher.get_summary(details)
            
            # Record global metrics
            weight_metrics['epoch'].append(epoch+1)
            weight_metrics['global_alpha'].append(summary['alpha'].mean())
            weight_metrics['global_stable_rank'].append(summary['stable_rank'].mean())
            weight_metrics['global_norm'].append(summary['norm'].mean())
            
            # Record component-specific metrics
            enc_details = watcher.analyze(model=model.encoder, layers=[nn.Conv2d, nn.Linear])
            dec_details = watcher.analyze(model=model.decoder, layers=[nn.ConvTranspose2d, nn.Linear])
            concept_details = watcher.analyze(model=model.concept_layer, layers=[nn.Linear])
            latent_details = watcher.analyze(model=[model.fc_mu, model.fc_logvar], layers=[nn.Linear])
            
            enc_summary = watcher.get_summary(enc_details) if not enc_details.empty else None
            dec_summary = watcher.get_summary(dec_details) if not dec_details.empty else None
            concept_summary = watcher.get_summary(concept_details) if not concept_details.empty else None
            latent_summary = watcher.get_summary(latent_details) if not latent_details.empty else None
            
            weight_metrics['encoder_alpha'].append(enc_summary['alpha'].mean() if enc_summary is not None else 0)
            weight_metrics['decoder_alpha'].append(dec_summary['alpha'].mean() if dec_summary is not None else 0)
            weight_metrics['concept_alpha'].append(concept_summary['alpha'].mean() if concept_summary is not None else 0)
            weight_metrics['latent_alpha'].append(latent_summary['alpha'].mean() if latent_summary is not None else 0)
            
            print(f"  Weight Analysis: Global α={weight_metrics['global_alpha'][-1]:.2f}, "
                  f"Encoder α={weight_metrics['encoder_alpha'][-1]:.2f}, "
                  f"Decoder α={weight_metrics['decoder_alpha'][-1]:.2f}")
        except Exception as e:
            print(f"  Weight analysis failed: {e}")
            # Fill with zeros if analysis fails
            weight_metrics['epoch'].append(epoch+1)
            for key in ['global_alpha', 'global_stable_rank', 'global_norm', 
                       'encoder_alpha', 'decoder_alpha', 'concept_alpha', 'latent_alpha']:
                weight_metrics[key].append(0.0)
        
        # Early stopping check
        if epoch > 10 and avg_loss < 0.01:
            print(f"Converged at epoch {epoch}")
            break
    
    # Clean up gradient hooks
    for hook in hooks:
        hook.remove()
    
    return model, weight_metrics

# ========== DIAGNOSTIC VISUALIZATION ==========
def diagnostic_visualization(model, test_sample, device='cpu'):
    """Comprehensive model diagnostics"""
    model.eval()
    with torch.no_grad():
        # Move sample to device
        test_sample = test_sample.to(device)
        
        # Get model outputs
        x_recon, mu, logvar, z, concepts, z_recon = model(test_sample.unsqueeze(0))
        x_recon = x_recon[0]
        
        # Move back to CPU for visualization
        test_sample = test_sample.cpu()
        x_recon = x_recon.cpu()
        
        # Create visualizations
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axs[0, 0].imshow(test_sample.permute(1, 2, 0))
        axs[0, 0].set_title("Original")
        axs[0, 0].axis('off')
        
        # Reconstruction
        axs[0, 1].imshow(x_recon.permute(1, 2, 0))
        axs[0, 1].set_title("Reconstruction")
        axs[0, 1].axis('off')
        
        # Error heatmap
        error = torch.abs(test_sample - x_recon).mean(dim=0)
        im = axs[0, 2].imshow(error.cpu(), cmap='hot')
        axs[0, 2].set_title("Error Map")
        axs[0, 2].axis('off')
        plt.colorbar(im, ax=axs[0, 2])
        
        # Concept activations
        concepts = concepts[0].cpu().numpy()
        axs[1, 0].bar(range(len(concepts)), concepts)
        axs[1, 0].set_title("Concept Activations")
        axs[1, 0].set_xlabel("Concept Index")
        axs[1, 0].set_ylabel("Activation")
        
        # Latent space visualization
        z = z[0].cpu().numpy()
        axs[1, 1].plot(z)
        axs[1, 1].set_title("Latent Vector")
        axs[1, 1].set_xlabel("Dimension")
        axs[1, 1].set_ylabel("Value")
        
        # Reconstruction quality metrics
        mse = F.mse_loss(x_recon, test_sample).item()
        psnr = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(x_recon, test_sample))).item()
        axs[1, 2].text(0.1, 0.8, f"MSE: {mse:.6f}", transform=axs[1, 2].transAxes, fontsize=12)
        axs[1, 2].text(0.1, 0.6, f"PSNR: {psnr:.2f} dB", transform=axs[1, 2].transAxes, fontsize=12)
        axs[1, 2].text(0.1, 0.4, f"Concepts: {concepts}", transform=axs[1, 2].transAxes, fontsize=10)
        axs[1, 2].set_title("Quality Metrics")
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("model_diagnostics.png", dpi=150)
        plt.show()

# ============================================================================
# PLOT WEIGHT METRICS
# ============================================================================

def plot_weight_metrics(metrics):
    """Visualize weight statistics over training epochs"""
    # Debug print to understand the data structure
    print(f"Debug - Metrics lengths:")
    for key, values in metrics.items():
        print(f"  {key}: {len(values)}")
    
    # Ensure all arrays have the same length
    min_length = min(len(values) for values in metrics.values())
    print(f"Using minimum length: {min_length}")
    
    if min_length == 0:
        print("No metrics data available for plotting")
        return
    
    # Truncate all arrays to the same length
    for key in metrics:
        metrics[key] = metrics[key][:min_length]
    
    plt.figure(figsize=(15, 10))
    
    # Alpha values plot
    plt.subplot(2, 2, 1)
    if len(metrics['global_alpha']) > 0:
        plt.plot(metrics['epoch'], metrics['global_alpha'], 'b-', label='Global')
        plt.plot(metrics['epoch'], metrics['encoder_alpha'], 'r-', label='Encoder')
        plt.plot(metrics['epoch'], metrics['decoder_alpha'], 'g-', label='Decoder')
        plt.plot(metrics['epoch'], metrics['concept_alpha'], 'm-', label='Concept')
        plt.plot(metrics['epoch'], metrics['latent_alpha'], 'c-', label='Latent')
    plt.title('Alpha (Power Law Exponent) Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha')
    plt.legend()
    plt.grid(True)
    
    # Stable rank plot
    plt.subplot(2, 2, 2)
    if len(metrics['global_stable_rank']) > 0:
        plt.plot(metrics['epoch'], metrics['global_stable_rank'], 'b-o')
    plt.title('Global Stable Rank Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Stable Rank')
    plt.grid(True)
    
    # Norm plot
    plt.subplot(2, 2, 3)
    if len(metrics['global_norm']) > 0:
        plt.plot(metrics['epoch'], metrics['global_norm'], 'r-o')
    plt.title('Global Frobenius Norm Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.grid(True)
    
    # Alpha distribution comparison
    plt.subplot(2, 2, 4)
    if len(metrics['encoder_alpha']) > 0:
        components = ['Encoder', 'Decoder', 'Concept', 'Latent']
        final_alphas = [
            metrics['encoder_alpha'][-1] if metrics['encoder_alpha'] else 0,
            metrics['decoder_alpha'][-1] if metrics['decoder_alpha'] else 0,
            metrics['concept_alpha'][-1] if metrics['concept_alpha'] else 0,
            metrics['latent_alpha'][-1] if metrics['latent_alpha'] else 0
        ]
        plt.bar(components, final_alphas, color=['red', 'green', 'magenta', 'cyan'])
    plt.title('Final Alpha by Component')
    plt.ylabel('Alpha')
    
    plt.tight_layout()
    plt.savefig("weight_metrics_evolution.png", dpi=150)
    plt.show()

# ============================================================================
# VISUALIZATION UTILITIES (ENHANCED)
# ============================================================================

def visualize_reconstructions(model, dataloader, num_samples=5, model_name="VAE", device='cpu'):
    """Enhanced visualization with factor comparison"""
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        batch = next(iter(dataloader))
        images = batch['image'][:num_samples].to(device)
        factors = batch['factors'][:num_samples]
        
        # Get reconstructions
        if "Identifiable" in model_name or "Precision" in model_name:
            reconstructions, _, _, _, _, _ = model(images)
        else:
            reconstructions, _, _, _ = model(images)
            
        # Move to CPU for visualization
        images = images.cpu()
        reconstructions = reconstructions.cpu()
        
        # Convert to numpy and denormalize
        originals = images.numpy().transpose(0, 2, 3, 1)
        reconstructions = reconstructions.numpy().transpose(0, 2, 3, 1)
        
        # Plot comparison
        plt.figure(figsize=(15, 4*num_samples))
        for i in range(num_samples):
            # Original
            plt.subplot(num_samples, 3, 3*i+1)
            plt.imshow(originals[i])
            plt.title(f"Original {i+1}\nX:{factors[i,0]:.2f}, Y:{factors[i,1]:.2f}\nR:{factors[i,2]:.2f}, G:{factors[i,3]:.2f}")
            plt.axis('off')
            
            # Reconstruction
            plt.subplot(num_samples, 3, 3*i+2)
            plt.imshow(reconstructions[i])
            plt.title(f"{model_name} Reconstruction")
            plt.axis('off')
            
            # Error heatmap
            plt.subplot(num_samples, 3, 3*i+3)
            error = np.abs(originals[i] - reconstructions[i])
            # Enhance error visibility
            error = np.clip(error * 5, 0, 1)
            plt.imshow(error, cmap='hot', vmin=0, vmax=1)
            plt.title("Error Magnified 5x")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f"{model_name.lower()}_reconstructions.png", dpi=150)
        plt.show()

# ============================================================================
# EXECUTION WITH IMPROVED MODELS
# ============================================================================

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    dataset = ColoredSquaresDataset(num_samples=1000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # # Train improved baseline VAE
    # print("Training improved baseline VAE...")
    # baseline_vae = ImprovedVAE(latent_dim=16).to(device)
    # baseline_vae = train_improved_model(baseline_vae, train_loader, epochs=50, device=device)
    
    # # Train improved identifiable VAE
    # print("\nTraining improved identifiable VAE...")
    # ident_vae = ImprovedVAE(latent_dim=16, concept_dim=4).to(device)
    # ident_vae = train_improved_model(ident_vae, train_loader, epochs=50, has_concepts=True, device=device)
    
    # Train high precision VAE with weight monitoring
    print("\nTraining high precision VAE with guaranteed reconstruction...")
    precision_vae = HighPrecisionVAE(latent_dim=16, concept_dim=4).to(device)
    precision_vae, weight_metrics = train_guaranteed(precision_vae, train_loader, epochs=50, device=device)
    
    # # Visualize results
    # print("\nVisualizing improved baseline VAE reconstructions...")
    # visualize_reconstructions(baseline_vae, test_loader, model_name="Improved Baseline", device=device)
    
    # print("\nVisualizing improved identifiable VAE reconstructions...")
    # visualize_reconstructions(ident_vae, test_loader, model_name="Improved Identifiable", device=device)
    
    print("\nVisualizing high precision VAE reconstructions...")
    visualize_reconstructions(precision_vae, test_loader, model_name="High Precision", device=device)
    
    # Plot weight metrics evolution
    print("\nPlotting weight metrics evolution...")
    plot_weight_metrics(weight_metrics)
    
    # Diagnostic visualization for high precision model
    print("\nRunning diagnostic visualization...")
    test_sample = next(iter(test_loader))['image'][0]
    diagnostic_visualization(precision_vae, test_sample, device=device)
    
   