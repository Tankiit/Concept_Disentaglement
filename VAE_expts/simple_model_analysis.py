#!/usr/bin/env python3
"""
Simple analysis script for the trained disentangled VAE model
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import os
from tqdm import tqdm

# Import from our main file - use the EXACT same classes that were used during training
from disentangled_vae_pythae_integration import (
    FactorizedDisentangledVAE, DisentangledVAEConfig,
    FlexibleDisentangledEncoder, FlexibleDisentangledDecoder,
    load_dataset
)

# Pythae imports
from pythae.models import VAEConfig

def load_model_weights_directly(model_path, device='mps'):
    """Load model weights directly from the .pt file"""
    try:
        # Create the model with EXACT configuration used during training
        disentangle_config = DisentangledVAEConfig(
            input_channels=1,  # MNIST is grayscale
            input_size=28,
            semantic_dim=16,
            attribute_dim=8,
            n_semantic_factors=32,  # CRITICAL: Must match training config
            n_attribute_factors=16,  # CRITICAL: Must match training config
            num_classes=10,
            batch_size=128,
            epochs=50,
            dataset_name="mnist"
        )
        
        # Configure VAE model (matching training setup)
        config = VAEConfig(
            input_dim=(disentangle_config.input_channels, disentangle_config.input_size, disentangle_config.input_size),
            latent_dim=disentangle_config.semantic_dim + disentangle_config.attribute_dim
        )
        
        # Add disentanglement-specific config (matching training setup)
        config.input_channels = disentangle_config.input_channels
        config.input_size = disentangle_config.input_size
        config.semantic_dim = disentangle_config.semantic_dim
        config.attribute_dim = disentangle_config.attribute_dim
        
        # Use the EXACT same encoder and decoder classes that were used during training
        # This ensures the architecture matches perfectly
        encoder = FlexibleDisentangledEncoder(config)
        decoder = FlexibleDisentangledDecoder(config)
        
        model = FactorizedDisentangledVAE(
            model_config=config,
            encoder=encoder,
            decoder=decoder,
            disentangle_config=disentangle_config  # Pass the disentangle config!
        ).to(device)
        
        # Load the weights from model.pt
        model_file = os.path.join(model_path, 'model.pt')
        if os.path.exists(model_file):
            print(f"Loading weights from: {model_file}")
            checkpoint = torch.load(model_file, map_location=device)
            
            # Filter out problematic keys (auxiliary classifier with wrong dimensions)
            filtered_state = {}
            for k, v in checkpoint.items():
                if 'auxiliary_classifier' in k:
                    print(f"Skipping {k} due to dimension mismatch")
                    continue
                filtered_state[k] = v
            
            # Load filtered state
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
            
            if missing_keys:
                print(f"Missing keys (using random init): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
            
            print("✓ Model loaded successfully with weight corrections!")
            return model
        else:
            print(f"Model file not found: {model_file}")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def extract_factors_simple(model, data, device='mps', batch_size=50):
    """Extract semantic and attribute factors from data"""
    if model is None:
        return None
        
    model.eval()
    semantic_factors = []
    attribute_factors = []
    reconstructions = []
    
    print("Extracting factor representations...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            try:
                batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
                
                # Ensure correct input format (batch_size, channels, height, width)
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(1)
                
                # Forward pass
                outputs = model(batch)
                
                # Extract the factors
                if hasattr(outputs, 'semantic_factors') and hasattr(outputs, 'attribute_factors'):
                    semantic_factors.append(outputs.semantic_factors.cpu().numpy())
                    attribute_factors.append(outputs.attribute_factors.cpu().numpy())
                    reconstructions.append(outputs.recon_x.cpu().numpy())
                else:
                    print("Model output doesn't have expected factor attributes")
                    break
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                break
    
    if semantic_factors:
        return {
            'semantic': np.concatenate(semantic_factors, axis=0),
            'attribute': np.concatenate(attribute_factors, axis=0),
            'reconstructions': np.concatenate(reconstructions, axis=0)
        }
    else:
        print("No factors extracted successfully")
        return None

def rq1_disentanglement_quality(semantic_factors, attribute_factors, labels):
    """RQ1: How well do semantic vs attribute factors capture different information?"""
    print("\n" + "="*60)
    print("RQ1: DISENTANGLEMENT QUALITY ANALYSIS")
    print("="*60)
    
    # 1. Mutual Information with digit classes
    print("\n1. Mutual Information with Digit Classes:")
    print("-" * 45)
    
    semantic_mi_scores = []
    for i in range(semantic_factors.shape[1]):
        # Discretize continuous factors for MI calculation
        factor_discrete = np.digitize(semantic_factors[:, i], 
                                    np.percentile(semantic_factors[:, i], [20, 40, 60, 80]))
        mi_score = mutual_info_score(labels, factor_discrete)
        semantic_mi_scores.append(mi_score)
        print(f"  Semantic Factor {i:2d}: {mi_score:.4f}")
    
    print()
    attribute_mi_scores = []
    for i in range(attribute_factors.shape[1]):
        factor_discrete = np.digitize(attribute_factors[:, i], 
                                    np.percentile(attribute_factors[:, i], [20, 40, 60, 80]))
        mi_score = mutual_info_score(labels, factor_discrete)
        attribute_mi_scores.append(mi_score)
        print(f"  Attribute Factor {i:2d}: {mi_score:.4f}")
    
    avg_semantic_mi = np.mean(semantic_mi_scores)
    avg_attribute_mi = np.mean(attribute_mi_scores)
    
    print(f"\nSummary:")
    print(f"  Average Semantic MI: {avg_semantic_mi:.4f}")
    print(f"  Average Attribute MI: {avg_attribute_mi:.4f}")
    print(f"  Ratio (Semantic/Attribute): {avg_semantic_mi/avg_attribute_mi:.2f}" if avg_attribute_mi > 0 else "  Ratio: N/A")
    
    # 2. Linear Classification Performance
    print(f"\n2. Linear Classification Performance:")
    print("-" * 40)
    
    # Split data for training/testing
    n_train = int(0.8 * len(semantic_factors))
    indices = np.random.permutation(len(semantic_factors))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    # Test semantic factors only
    clf_semantic = LogisticRegression(random_state=42, max_iter=1000)
    clf_semantic.fit(semantic_factors[train_idx], labels[train_idx])
    semantic_acc = accuracy_score(labels[test_idx], clf_semantic.predict(semantic_factors[test_idx]))
    
    # Test attribute factors only
    clf_attribute = LogisticRegression(random_state=42, max_iter=1000)
    clf_attribute.fit(attribute_factors[train_idx], labels[train_idx])
    attribute_acc = accuracy_score(labels[test_idx], clf_attribute.predict(attribute_factors[test_idx]))
    
    # Test combined factors
    combined_factors = np.concatenate([semantic_factors, attribute_factors], axis=1)
    clf_combined = LogisticRegression(random_state=42, max_iter=1000)
    clf_combined.fit(combined_factors[train_idx], labels[train_idx])
    combined_acc = accuracy_score(labels[test_idx], clf_combined.predict(combined_factors[test_idx]))
    
    print(f"  Semantic factors → Digit classification: {semantic_acc:.4f}")
    print(f"  Attribute factors → Digit classification: {attribute_acc:.4f}")
    print(f"  Combined factors → Digit classification: {combined_acc:.4f}")
    
    # 3. Factor Independence Analysis
    print(f"\n3. Factor Independence Analysis:")
    print("-" * 35)
    
    # Compute cross-correlations between semantic and attribute factors
    cross_correlations = []
    for i in range(semantic_factors.shape[1]):
        for j in range(attribute_factors.shape[1]):
            corr = np.corrcoef(semantic_factors[:, i], attribute_factors[:, j])[0, 1]
            if not np.isnan(corr):
                cross_correlations.append(abs(corr))
    
    if cross_correlations:
        max_cross_corr = max(cross_correlations)
        avg_cross_corr = np.mean(cross_correlations)
        print(f"  Max cross-correlation: {max_cross_corr:.4f}")
        print(f"  Average cross-correlation: {avg_cross_corr:.4f}")
        
        independence_score = 1.0 - avg_cross_corr
        print(f"  Independence score: {independence_score:.4f}")
    else:
        max_cross_corr = 0
        avg_cross_corr = 0
        independence_score = 1.0
        print(f"  Could not compute correlations")
    
    return {
        'semantic_mi': semantic_mi_scores,
        'attribute_mi': attribute_mi_scores,
        'avg_semantic_mi': avg_semantic_mi,
        'avg_attribute_mi': avg_attribute_mi,
        'semantic_acc': semantic_acc,
        'attribute_acc': attribute_acc,
        'combined_acc': combined_acc,
        'max_cross_corr': max_cross_corr,
        'avg_cross_corr': avg_cross_corr,
        'independence_score': independence_score
    }

def rq2_reconstruction_quality(original_data, reconstructions, labels):
    """RQ2: How well does the model reconstruct inputs?"""
    print("\n" + "="*60)
    print("RQ2: RECONSTRUCTION QUALITY ANALYSIS")
    print("="*60)
    
    # Compute reconstruction errors
    mse_errors = np.mean((original_data - reconstructions)**2, axis=(1,2,3))
    mae_errors = np.mean(np.abs(original_data - reconstructions), axis=(1,2,3))
    
    print(f"\nOverall Reconstruction Quality:")
    print("-" * 35)
    print(f"  Mean MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
    print(f"  Mean MAE: {np.mean(mae_errors):.6f} ± {np.std(mae_errors):.6f}")
    
    # Per-digit analysis
    print(f"\nPer-digit Reconstruction Quality (MSE):")
    print("-" * 40)
    digit_mse = {}
    for digit in range(10):
        digit_mask = labels == digit
        if np.any(digit_mask):
            digit_mse[digit] = np.mean(mse_errors[digit_mask])
            print(f"  Digit {digit}: {digit_mse[digit]:.6f}")
    
    # Quality assessment
    mean_mse = np.mean(mse_errors)
    if mean_mse < 0.01:
        quality = "Excellent"
    elif mean_mse < 0.05:
        quality = "Good"
    elif mean_mse < 0.1:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nOverall Quality Assessment: {quality}")
    
    return {
        'mse_errors': mse_errors,
        'mae_errors': mae_errors,
        'digit_mse': digit_mse,
        'mean_mse': mean_mse,
        'mean_mae': np.mean(mae_errors),
        'quality': quality
    }

def rq3_interpretability_analysis(semantic_factors, attribute_factors, labels):
    """RQ3: How interpretable are the learned factors?"""
    print("\n" + "="*60)
    print("RQ3: FACTOR INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    # 1. Factor Activation Patterns per Digit
    print(f"\n1. Factor Activation Patterns per Digit:")
    print("-" * 45)
    
    digit_patterns = {}
    for digit in range(10):
        digit_mask = labels == digit
        if np.any(digit_mask):
            digit_semantic = semantic_factors[digit_mask]
            digit_attribute = attribute_factors[digit_mask]
            
            # Compute means
            semantic_mean = np.mean(digit_semantic, axis=0)
            attribute_mean = np.mean(digit_attribute, axis=0)
            
            # Find most active factors for this digit
            top_semantic = np.argsort(semantic_mean)[-3:][::-1]
            top_attribute = np.argsort(attribute_mean)[-3:][::-1]
            
            digit_patterns[digit] = {
                'semantic_mean': semantic_mean,
                'attribute_mean': attribute_mean,
                'top_semantic': top_semantic,
                'top_attribute': top_attribute
            }
            
            print(f"  Digit {digit}:")
            print(f"    Top semantic factors: {top_semantic} (values: {semantic_mean[top_semantic]:.3f})")
            print(f"    Top attribute factors: {top_attribute} (values: {attribute_mean[top_attribute]:.3f})")
    
    # 2. Factor Specialization Analysis
    print(f"\n2. Factor Specialization Analysis:")
    print("-" * 40)
    
    # Measure how much each factor varies across digits
    semantic_specialization = []
    for i in range(semantic_factors.shape[1]):
        digit_means = []
        for digit in range(10):
            digit_mask = labels == digit
            if np.any(digit_mask):
                digit_means.append(np.mean(semantic_factors[digit_mask, i]))
        
        if len(digit_means) > 1:
            # Specialization = variance across digits / total variance
            specialization = np.var(digit_means) / (np.var(semantic_factors[:, i]) + 1e-8)
            semantic_specialization.append(specialization)
            print(f"  Semantic Factor {i}: Specialization = {specialization:.4f}")
    
    print()
    attribute_specialization = []
    for i in range(attribute_factors.shape[1]):
        digit_means = []
        for digit in range(10):
            digit_mask = labels == digit
            if np.any(digit_mask):
                digit_means.append(np.mean(attribute_factors[digit_mask, i]))
        
        if len(digit_means) > 1:
            specialization = np.var(digit_means) / (np.var(attribute_factors[:, i]) + 1e-8)
            attribute_specialization.append(specialization)
            print(f"  Attribute Factor {i}: Specialization = {specialization:.4f}")
    
    # Summary
    avg_semantic_spec = np.mean(semantic_specialization) if semantic_specialization else 0
    avg_attribute_spec = np.mean(attribute_specialization) if attribute_specialization else 0
    
    print(f"\nSpecialization Summary:")
    print(f"  Average semantic specialization: {avg_semantic_spec:.4f}")
    print(f"  Average attribute specialization: {avg_attribute_spec:.4f}")
    
    return {
        'digit_patterns': digit_patterns,
        'semantic_specialization': semantic_specialization,
        'attribute_specialization': attribute_specialization,
        'avg_semantic_spec': avg_semantic_spec,
        'avg_attribute_spec': avg_attribute_spec
    }

def create_visualizations(data, reconstructions, semantic_factors, attribute_factors, labels, rq1_results, rq2_results):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    # 1. Sample reconstructions
    plt.figure(figsize=(15, 8))
    
    # Show some good and some challenging reconstructions
    mse_errors = rq2_results['mse_errors']
    
    # Best reconstructions
    best_indices = np.argsort(mse_errors)[:4]
    # Worst reconstructions  
    worst_indices = np.argsort(mse_errors)[-4:]
    
    for i, idx in enumerate(best_indices):
        # Original
        plt.subplot(4, 4, i*2 + 1)
        plt.imshow(data[idx, 0] if len(data[idx].shape) > 2 else data[idx], cmap='gray')
        plt.title(f'Best Original {i+1}\nDigit {labels[idx]}')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(4, 4, i*2 + 2)
        plt.imshow(reconstructions[idx, 0] if len(reconstructions[idx].shape) > 2 else reconstructions[idx], cmap='gray')
        plt.title(f'Reconstruction\nMSE: {mse_errors[idx]:.4f}')
        plt.axis('off')
    
    for i, idx in enumerate(worst_indices):
        # Original
        plt.subplot(4, 4, 8 + i*2 + 1)
        plt.imshow(data[idx, 0] if len(data[idx].shape) > 2 else data[idx], cmap='gray')
        plt.title(f'Worst Original {i+1}\nDigit {labels[idx]}')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(4, 4, 8 + i*2 + 2)
        plt.imshow(reconstructions[idx, 0] if len(reconstructions[idx].shape) > 2 else reconstructions[idx], cmap='gray')
        plt.title(f'Reconstruction\nMSE: {mse_errors[idx]:.4f}')
        plt.axis('off')
    
    plt.suptitle('Reconstruction Quality: Best vs Worst Examples')
    plt.tight_layout()
    plt.savefig('reconstruction_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Factor analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MI scores
    axes[0, 0].bar(range(len(rq1_results['semantic_mi'])), rq1_results['semantic_mi'])
    axes[0, 0].set_title('Semantic Factors - MI with Digit Classes')
    axes[0, 0].set_xlabel('Factor Index')
    axes[0, 0].set_ylabel('Mutual Information')
    
    axes[0, 1].bar(range(len(rq1_results['attribute_mi'])), rq1_results['attribute_mi'])
    axes[0, 1].set_title('Attribute Factors - MI with Digit Classes')
    axes[0, 1].set_xlabel('Factor Index')
    axes[0, 1].set_ylabel('Mutual Information')
    
    # Classification accuracies
    accuracies = [rq1_results['semantic_acc'], rq1_results['attribute_acc'], rq1_results['combined_acc']]
    axes[0, 2].bar(['Semantic', 'Attribute', 'Combined'], accuracies)
    axes[0, 2].set_title('Linear Classification Accuracy')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_ylim(0, 1)
    
    # Factor spaces (first 2 dimensions)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels == digit
        if np.any(mask) and semantic_factors.shape[1] >= 2:
            axes[1, 0].scatter(semantic_factors[mask, 0], semantic_factors[mask, 1], 
                             c=[colors[digit]], label=f'{digit}', alpha=0.6, s=10)
    axes[1, 0].set_title('Semantic Factor Space (dims 0-1)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for digit in range(10):
        mask = labels == digit
        if np.any(mask) and attribute_factors.shape[1] >= 2:
            axes[1, 1].scatter(attribute_factors[mask, 0], attribute_factors[mask, 1], 
                             c=[colors[digit]], label=f'{digit}', alpha=0.6, s=10)
    axes[1, 1].set_title('Attribute Factor Space (dims 0-1)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # MSE distribution
    axes[1, 2].hist(mse_errors, bins=50, alpha=0.7)
    axes[1, 2].set_title('Reconstruction Error Distribution')
    axes[1, 2].set_xlabel('MSE')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved:")
    print("- reconstruction_analysis.png")
    print("- comprehensive_analysis.png")

def main():
    """Main analysis pipeline"""
    print("Disentangled VAE Model Analysis")
    print("=" * 40)
    
    # Configuration
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Model path: {model_path}")
    
    # Load data
    print("\nLoading MNIST test data...")
    train_data, test_data, train_labels, test_labels, n_classes = load_dataset('mnist', train_size=1000, test_size=1000)
    print(f"Loaded {len(test_data)} test samples with {n_classes} classes")
    
    # Load model
    print("\nLoading trained model...")
    model = load_model_weights_directly(model_path, device)
    
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Extract representations
    representations = extract_factors_simple(model, test_data, device)
    
    if representations is None:
        print("Failed to extract representations. Exiting...")
        return
    
    semantic_factors = representations['semantic']
    attribute_factors = representations['attribute']
    reconstructions = representations['reconstructions']
    
    print(f"\nExtracted representations:")
    print(f"  Semantic factors: {semantic_factors.shape}")
    print(f"  Attribute factors: {attribute_factors.shape}")
    print(f"  Reconstructions: {reconstructions.shape}")
    
    # Run research question analyses
    rq1_results = rq1_disentanglement_quality(semantic_factors, attribute_factors, test_labels)
    rq2_results = rq2_reconstruction_quality(test_data, reconstructions, test_labels)
    rq3_results = rq3_interpretability_analysis(semantic_factors, attribute_factors, test_labels)
    
    # Create visualizations
    create_visualizations(test_data, reconstructions, semantic_factors, attribute_factors, 
                         test_labels, rq1_results, rq2_results)
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nRQ1 - Disentanglement Quality:")
    print(f"  Semantic factors avg MI: {rq1_results['avg_semantic_mi']:.4f}")
    print(f"  Attribute factors avg MI: {rq1_results['avg_attribute_mi']:.4f}")
    print(f"  Semantic classification accuracy: {rq1_results['semantic_acc']:.4f}")
    print(f"  Attribute classification accuracy: {rq1_results['attribute_acc']:.4f}")
    print(f"  Combined classification accuracy: {rq1_results['combined_acc']:.4f}")
    print(f"  Factor independence score: {rq1_results['independence_score']:.4f}")
    
    print(f"\nRQ2 - Reconstruction Quality:")
    print(f"  Mean MSE: {rq2_results['mean_mse']:.6f}")
    print(f"  Mean MAE: {rq2_results['mean_mae']:.6f}")
    print(f"  Quality assessment: {rq2_results['quality']}")
    
    print(f"\nRQ3 - Interpretability:")
    print(f"  Avg semantic specialization: {rq3_results['avg_semantic_spec']:.4f}")
    print(f"  Avg attribute specialization: {rq3_results['avg_attribute_spec']:.4f}")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    disentanglement_score = (rq1_results['independence_score'] + 
                           (rq1_results['semantic_acc'] - rq1_results['attribute_acc']) / 2 + 0.5) / 2
    reconstruction_score = 1.0 - min(rq2_results['mean_mse'] * 10, 1.0)  # Scale MSE to 0-1
    interpretability_score = (rq3_results['avg_semantic_spec'] + rq3_results['avg_attribute_spec']) / 2
    
    overall_score = (disentanglement_score + reconstruction_score + interpretability_score) / 3
    
    print(f"  Disentanglement score: {disentanglement_score:.4f}")
    print(f"  Reconstruction score: {reconstruction_score:.4f}")
    print(f"  Interpretability score: {interpretability_score:.4f}")
    print(f"  Overall model score: {overall_score:.4f}")
    
    if overall_score > 0.7:
        assessment = "Excellent"
    elif overall_score > 0.5:
        assessment = "Good"
    elif overall_score > 0.3:
        assessment = "Fair"
    else:
        assessment = "Needs Improvement"
    
    print(f"  Final assessment: {assessment}")

if __name__ == "__main__":
    main() 