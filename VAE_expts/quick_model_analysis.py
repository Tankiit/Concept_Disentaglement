import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# Import from our main file
from disentangled_vae_pythae_integration import (
    FactorizedDisentangledVAE, DisentangledVAEConfig,
    FlexibleDisentangledEncoder, FlexibleDisentangledDecoder,
    load_mnist_data
)

def load_trained_model(model_path, device='mps'):
    """Load the trained model with proper error handling"""
    try:
        # Try loading directly first
        model = FactorizedDisentangledVAE.load_from_folder(model_path).to(device)
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Loading error: {e}")
        print("Creating new model with correct dimensions...")
        
        # Create model with MNIST dimensions
        config = DisentangledVAEConfig(
            input_shape=(1, 28, 28),
            semantic_dim=16,
            attribute_dim=8,
            n_classes=10
        )
        
        encoder = FlexibleDisentangledEncoder(config)
        decoder = FlexibleDisentangledDecoder(config)
        model = FactorizedDisentangledVAE(config, encoder, decoder).to(device)
        
        # Load weights manually, skipping problematic layers
        try:
            checkpoint = torch.load(os.path.join(model_path, 'model.pkl'), map_location=device)
            
            # Filter out auxiliary classifier weights (dimension mismatch)
            filtered_weights = {k: v for k, v in checkpoint.items() 
                              if not k.startswith('auxiliary_classifier')}
            
            model.load_state_dict(filtered_weights, strict=False)
            print("✓ Model loaded with corrections!")
            return model
        except Exception as e2:
            print(f"Manual loading also failed: {e2}")
            print("Using freshly initialized model for demonstration...")
            return model

def extract_factor_representations(model, data, device='mps', batch_size=100):
    """Extract semantic and attribute factor representations"""
    model.eval()
    
    semantic_factors = []
    attribute_factors = []
    reconstructions = []
    
    print("Extracting factor representations...")
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
            
            # Ensure correct input format
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(1)
            
            try:
                outputs = model(batch)
                semantic_factors.append(outputs.semantic_factors.cpu().numpy())
                attribute_factors.append(outputs.attribute_factors.cpu().numpy())
                reconstructions.append(outputs.recon_x.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                break
    
    if semantic_factors:
        return {
            'semantic_factors': np.concatenate(semantic_factors, axis=0),
            'attribute_factors': np.concatenate(attribute_factors, axis=0),
            'reconstructions': np.concatenate(reconstructions, axis=0)
        }
    else:
        return None

def analyze_factor_digit_correlation(semantic_factors, attribute_factors, labels):
    """Analyze how factors correlate with digit classes"""
    print("\n" + "="*50)
    print("FACTOR-DIGIT CORRELATION ANALYSIS")
    print("="*50)
    
    # Mutual Information Analysis
    print("\nMutual Information Scores:")
    print("-" * 30)
    
    semantic_mi_scores = []
    for i in range(semantic_factors.shape[1]):
        # Discretize factor for MI calculation
        factor_discrete = np.digitize(semantic_factors[:, i], 
                                    np.percentile(semantic_factors[:, i], [25, 50, 75]))
        mi_score = mutual_info_score(labels, factor_discrete)
        semantic_mi_scores.append(mi_score)
        print(f"Semantic Factor {i:2d}: {mi_score:.4f}")
    
    print()
    attribute_mi_scores = []
    for i in range(attribute_factors.shape[1]):
        factor_discrete = np.digitize(attribute_factors[:, i], 
                                    np.percentile(attribute_factors[:, i], [25, 50, 75]))
        mi_score = mutual_info_score(labels, factor_discrete)
        attribute_mi_scores.append(mi_score)
        print(f"Attribute Factor {i:2d}: {mi_score:.4f}")
    
    print(f"\nAverage Semantic MI: {np.mean(semantic_mi_scores):.4f}")
    print(f"Average Attribute MI: {np.mean(attribute_mi_scores):.4f}")
    
    return semantic_mi_scores, attribute_mi_scores

def test_linear_classification(semantic_factors, attribute_factors, labels):
    """Test linear classification performance using factors"""
    print("\n" + "="*50)
    print("LINEAR CLASSIFICATION TEST")
    print("="*50)
    
    # Split data
    n_train = int(0.8 * len(semantic_factors))
    indices = np.random.permutation(len(semantic_factors))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    # Test semantic factors
    clf_semantic = LogisticRegression(random_state=42, max_iter=1000)
    clf_semantic.fit(semantic_factors[train_idx], labels[train_idx])
    semantic_acc = accuracy_score(labels[test_idx], clf_semantic.predict(semantic_factors[test_idx]))
    
    # Test attribute factors
    clf_attribute = LogisticRegression(random_state=42, max_iter=1000)
    clf_attribute.fit(attribute_factors[train_idx], labels[train_idx])
    attribute_acc = accuracy_score(labels[test_idx], clf_attribute.predict(attribute_factors[test_idx]))
    
    # Test combined factors
    combined_factors = np.concatenate([semantic_factors, attribute_factors], axis=1)
    clf_combined = LogisticRegression(random_state=42, max_iter=1000)
    clf_combined.fit(combined_factors[train_idx], labels[train_idx])
    combined_acc = accuracy_score(labels[test_idx], clf_combined.predict(combined_factors[test_idx]))
    
    print(f"Semantic Factors → Digit Classification: {semantic_acc:.4f}")
    print(f"Attribute Factors → Digit Classification: {attribute_acc:.4f}")
    print(f"Combined Factors → Digit Classification: {combined_acc:.4f}")
    
    return semantic_acc, attribute_acc, combined_acc

def analyze_reconstruction_quality(original_data, reconstructions, labels):
    """Analyze reconstruction quality"""
    print("\n" + "="*50)
    print("RECONSTRUCTION QUALITY ANALYSIS")
    print("="*50)
    
    # Compute MSE and MAE
    mse_errors = np.mean((original_data - reconstructions)**2, axis=(1,2,3))
    mae_errors = np.mean(np.abs(original_data - reconstructions), axis=(1,2,3))
    
    print(f"Mean MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
    print(f"Mean MAE: {np.mean(mae_errors):.6f} ± {np.std(mae_errors):.6f}")
    
    # Per-digit analysis
    print(f"\nPer-digit reconstruction quality (MSE):")
    for digit in range(10):
        digit_mask = labels == digit
        if np.any(digit_mask):
            digit_mse = np.mean(mse_errors[digit_mask])
            print(f"  Digit {digit}: {digit_mse:.6f}")
    
    return mse_errors, mae_errors

def visualize_results(data, reconstructions, semantic_factors, attribute_factors, labels, 
                     semantic_mi_scores, attribute_mi_scores):
    """Create visualization plots"""
    
    # 1. Sample reconstructions
    plt.figure(figsize=(15, 6))
    sample_indices = np.random.choice(len(data), 8, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Original
        plt.subplot(2, 8, i + 1)
        plt.imshow(data[idx, 0] if len(data[idx].shape) > 2 else data[idx], cmap='gray')
        plt.title(f'Original\nDigit {labels[idx]}')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(2, 8, i + 9)
        plt.imshow(reconstructions[idx, 0] if len(reconstructions[idx].shape) > 2 else reconstructions[idx], cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Mutual Information scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(semantic_mi_scores)), semantic_mi_scores)
    plt.title('Semantic Factors - MI with Digit Classes')
    plt.xlabel('Factor Index')
    plt.ylabel('Mutual Information')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(attribute_mi_scores)), attribute_mi_scores)
    plt.title('Attribute Factors - MI with Digit Classes')
    plt.xlabel('Factor Index')
    plt.ylabel('Mutual Information')
    
    plt.tight_layout()
    plt.savefig('mutual_information_scores.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Factor space visualization (first 2 dimensions)
    plt.figure(figsize=(12, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    plt.subplot(1, 2, 1)
    for digit in range(10):
        mask = labels == digit
        if np.any(mask):
            plt.scatter(semantic_factors[mask, 0], semantic_factors[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=20)
    plt.title('Semantic Factor Space (dims 0-1)')
    plt.xlabel('Semantic Factor 0')
    plt.ylabel('Semantic Factor 1')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    for digit in range(10):
        mask = labels == digit
        if np.any(mask):
            plt.scatter(attribute_factors[mask, 0], attribute_factors[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=20)
    plt.title('Attribute Factor Space (dims 0-1)')
    plt.xlabel('Attribute Factor 0')
    plt.ylabel('Attribute Factor 1')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('factor_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def perform_factor_manipulation(model, data, labels, device='mps', n_examples=3):
    """Demonstrate factor manipulation"""
    print("\n" + "="*50)
    print("FACTOR MANIPULATION DEMONSTRATION")
    print("="*50)
    
    model.eval()
    
    # Select diverse examples
    selected_indices = []
    for digit in range(min(n_examples, 10)):
        digit_indices = np.where(labels == digit)[0]
        if len(digit_indices) > 0:
            selected_indices.append(np.random.choice(digit_indices))
    
    manipulation_results = []
    
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img = torch.tensor(data[idx:idx+1], dtype=torch.float32).to(device)
            if len(img.shape) == 3:
                img = img.unsqueeze(1)
            
            # Get original factors
            outputs = model(img)
            orig_semantic = outputs.semantic_factors
            orig_attribute = outputs.attribute_factors
            
            print(f"Example {i+1} (Digit {labels[idx]}):")
            print(f"  Original semantic factors: {orig_semantic[0].cpu().numpy()[:5]}...")
            print(f"  Original attribute factors: {orig_attribute[0].cpu().numpy()[:3]}...")
            
            # Try manipulating first semantic factor
            manipulated_images = []
            for value in [-2, -1, 0, 1, 2]:
                modified_semantic = orig_semantic.clone()
                modified_semantic[0, 0] = value  # Modify first semantic factor
                
                # Generate new image (if model has decode_factors method)
                try:
                    generated = model.decode_factors(modified_semantic, orig_attribute)
                    manipulated_images.append(generated.cpu().numpy()[0])
                except AttributeError:
                    # Fallback: use decoder directly
                    z_combined = torch.cat([modified_semantic, orig_attribute], dim=1)
                    generated = model.decoder(z_combined)
                    manipulated_images.append(generated.reconstruction.cpu().numpy()[0])
            
            manipulation_results.append({
                'original': data[idx],
                'label': labels[idx],
                'manipulated': manipulated_images
            })
    
    # Visualize manipulations
    if manipulation_results:
        fig, axes = plt.subplots(len(manipulation_results), 6, figsize=(18, 3*len(manipulation_results)))
        
        for i, result in enumerate(manipulation_results):
            # Original
            if len(manipulation_results) == 1:
                axes[0].imshow(result['original'][0] if len(result['original'].shape) > 2 else result['original'], cmap='gray')
                axes[0].set_title(f'Original\n(Digit {result["label"]})')
                axes[0].axis('off')
                
                # Manipulations
                for j, img in enumerate(result['manipulated']):
                    axes[j+1].imshow(img[0] if len(img.shape) > 2 else img, cmap='gray')
                    axes[j+1].set_title(f'Factor=\n{[-2,-1,0,1,2][j]}')
                    axes[j+1].axis('off')
            else:
                axes[i, 0].imshow(result['original'][0] if len(result['original'].shape) > 2 else result['original'], cmap='gray')
                axes[i, 0].set_title(f'Original\n(Digit {result["label"]})')
                axes[i, 0].axis('off')
                
                # Manipulations
                for j, img in enumerate(result['manipulated']):
                    axes[i, j+1].imshow(img[0] if len(img.shape) > 2 else img, cmap='gray')
                    axes[i, j+1].set_title(f'Factor=\n{[-2,-1,0,1,2][j]}')
                    axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig('factor_manipulations.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main analysis pipeline"""
    print("Quick Model Analysis for Disentangled VAE")
    print("="*50)
    
    # Load data
    print("Loading MNIST dataset...")
    _, test_data, _, test_labels = load_mnist_data(train_size=1000, test_size=1000)
    
    # Load model
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_trained_model(model_path, device)
    
    # Extract representations
    representations = extract_factor_representations(model, test_data, device)
    
    if representations is None:
        print("Failed to extract representations. Exiting...")
        return
    
    semantic_factors = representations['semantic_factors']
    attribute_factors = representations['attribute_factors']
    reconstructions = representations['reconstructions']
    
    print(f"Extracted representations:")
    print(f"  Semantic factors shape: {semantic_factors.shape}")
    print(f"  Attribute factors shape: {attribute_factors.shape}")
    print(f"  Reconstructions shape: {reconstructions.shape}")
    
    # Run analyses
    semantic_mi, attribute_mi = analyze_factor_digit_correlation(
        semantic_factors, attribute_factors, test_labels)
    
    semantic_acc, attribute_acc, combined_acc = test_linear_classification(
        semantic_factors, attribute_factors, test_labels)
    
    mse_errors, mae_errors = analyze_reconstruction_quality(
        test_data, reconstructions, test_labels)
    
    # Visualizations
    visualize_results(test_data, reconstructions, semantic_factors, attribute_factors, 
                     test_labels, semantic_mi, attribute_mi)
    
    # Factor manipulation
    perform_factor_manipulation(model, test_data, test_labels, device)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Average Semantic MI Score: {np.mean(semantic_mi):.4f}")
    print(f"Average Attribute MI Score: {np.mean(attribute_mi):.4f}")
    print(f"Semantic Classification Accuracy: {semantic_acc:.4f}")
    print(f"Attribute Classification Accuracy: {attribute_acc:.4f}")
    print(f"Combined Classification Accuracy: {combined_acc:.4f}")
    print(f"Mean Reconstruction MSE: {np.mean(mse_errors):.6f}")
    
    print("\nGenerated files:")
    print("- sample_reconstructions.png")
    print("- mutual_information_scores.png")
    print("- factor_space_visualization.png")
    print("- factor_manipulations.png")

if __name__ == "__main__":
    main() 