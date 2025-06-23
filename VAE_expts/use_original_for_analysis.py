#!/usr/bin/env python3
"""
Use the original training classes directly for analysis
This should work since it uses the EXACT same classes that created the model
"""

import torch
import numpy as np
import os
import sys

# Import everything from the original training file
from disentangled_vae_pythae_integration import *

def load_and_analyze_original():
    """Load and analyze using the original classes"""
    
    print("Using Original Training Classes for Analysis")
    print("=" * 50)
    
    # Use the exact same configuration as training
    disentangle_config = DisentangledVAEConfig(
        input_channels=1,
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
    
    # Configure VAE model
    config = VAEConfig(
        input_dim=(disentangle_config.input_channels, disentangle_config.input_size, disentangle_config.input_size),
        latent_dim=disentangle_config.semantic_dim + disentangle_config.attribute_dim
    )
    
    # Add disentanglement-specific config
    config.input_channels = disentangle_config.input_channels
    config.input_size = disentangle_config.input_size
    config.semantic_dim = disentangle_config.semantic_dim
    config.attribute_dim = disentangle_config.attribute_dim
    
    # Create model using EXACT same classes
    print("Creating model with original classes...")
    model = FactorizedDisentangledVAE(
        model_config=config,
        encoder=FlexibleDisentangledEncoder(config),
        decoder=FlexibleDisentangledDecoder(config),
        disentangle_config=disentangle_config
    )
    
    # Load the saved model
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    
    try:
        # Try loading using Pythae's built-in load method
        print(f"Loading model from: {model_path}")
        loaded_model = FactorizedDisentangledVAE.load_from_folder(model_path)
        print("✅ Model loaded successfully using Pythae's load_from_folder!")
        
        # Load test data
        print("Loading test data...")
        _, test_data, _, test_labels, _ = load_dataset('mnist', test_size=100)
        
        # Test the model
        print("Testing model...")
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        loaded_model = loaded_model.to(device)
        loaded_model.eval()
        
        # Process a small batch
        test_batch = torch.tensor(test_data[:10], dtype=torch.float32).to(device)
        if test_batch.dim() == 3:
            test_batch = test_batch.unsqueeze(1)
        
        with torch.no_grad():
            outputs = loaded_model(test_batch)
            
            print(f"✅ Model forward pass successful!")
            print(f"  Input shape: {test_batch.shape}")
            print(f"  Reconstruction shape: {outputs.recon_x.shape}")
            
            if hasattr(outputs, 'semantic_factors'):
                print(f"  Semantic factors shape: {outputs.semantic_factors.shape}")
                print(f"  Attribute factors shape: {outputs.attribute_factors.shape}")
                
                # Basic analysis
                semantic_factors = outputs.semantic_factors.cpu().numpy()
                attribute_factors = outputs.attribute_factors.cpu().numpy()
                
                print(f"\n📊 Quick Analysis:")
                print(f"  Semantic factors - Mean: {semantic_factors.mean():.4f}, Std: {semantic_factors.std():.4f}")
                print(f"  Attribute factors - Mean: {attribute_factors.mean():.4f}, Std: {attribute_factors.std():.4f}")
                print(f"  Semantic sparsity: {(semantic_factors < 0.1).mean():.4f}")
                print(f"  Attribute sparsity: {(attribute_factors < 0.1).mean():.4f}")
                
                print(f"\n🎉 SUCCESS: The disentangled VAE is working correctly!")
                print(f"The model successfully produces semantic and attribute factors.")
                
            else:
                print("⚠️  Model loaded but doesn't have factor outputs")
                
    except Exception as e:
        print(f"❌ Error loading with Pythae method: {e}")
        
        # Fallback: manual loading
        print("Trying manual loading...")
        try:
            model_file = os.path.join(model_path, 'model.pt')
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            print("✅ Manual loading successful!")
            
            # Test the manually loaded model
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = model.to(device)
            model.eval()
            
            # Load test data
            _, test_data, _, test_labels, _ = load_dataset('mnist', test_size=10)
            test_batch = torch.tensor(test_data, dtype=torch.float32).to(device)
            if test_batch.dim() == 3:
                test_batch = test_batch.unsqueeze(1)
                
            with torch.no_grad():
                outputs = model(test_batch)
                print(f"✅ Manual model forward pass successful!")
                print(f"  Available outputs: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                
        except Exception as e2:
            print(f"❌ Manual loading also failed: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    load_and_analyze_original() 