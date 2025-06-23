#!/usr/bin/env python3
"""
Advanced Dimension Interpreter for Disentangled VAE Models

This module provides post-hoc analysis to discover what each dimension learned,
leveraging TensorBoard logs and trained model outputs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json

# Import your model classes
from disentangled_vae_pythae_integration import FactorizedDisentangledVAE, load_dataset

class DimensionInterpreter:
    """
    Post-hoc analysis to discover what each dimension learned in a disentangled VAE.
    
    This class provides comprehensive analysis of learned representations including:
    - Correlation with known factors (digit class, style variations)
    - Prototype discovery (images that maximally activate each dimension)
    - Dimension manipulation effects
    - TensorBoard log analysis
    - Clustering and semantic grouping
    """
    
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for analysis results
        self.dimension_meanings = {}
        self.prototypes = {}
        self.correlations = {}
        self.manipulation_effects = {}
        
    def interpret_dimensions(self, labeled_data, max_samples=2000, batch_size=100):
        """
        Main method to discover semantic meaning of each dimension after training.
        
        Args:
            labeled_data: Dataset with labels (can be DataLoader or numpy arrays)
            max_samples: Maximum number of samples to analyze
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing interpretation results for each dimension
        """
        print("🔍 Starting dimension interpretation analysis...")
        
        # Extract all factors and ground truth information
        print("📊 Extracting factor representations...")
        factor_data = self._extract_factor_data(labeled_data, max_samples, batch_size)
        
        if factor_data is None:
            print("❌ Failed to extract factor data")
            return None
        
        # Analyze semantic dimensions
        print("🧠 Analyzing semantic dimensions...")
        semantic_meanings = self._analyze_semantic_dimensions(factor_data)
        
        # Analyze attribute dimensions  
        print("🎨 Analyzing attribute dimensions...")
        attribute_meanings = self._analyze_attribute_dimensions(factor_data)
        
        # Find prototypes for each dimension
        print("🔍 Finding dimension prototypes...")
        self._find_dimension_prototypes(factor_data)
        
        # Analyze manipulation effects
        print("⚙️ Analyzing manipulation effects...")
        self._analyze_manipulation_effects(factor_data)
        
        # Combine results
        self.dimension_meanings = {
            'semantic': semantic_meanings,
            'attribute': attribute_meanings,
            'prototypes': self.prototypes,
            'manipulation_effects': self.manipulation_effects
        }
        
        # Generate summary report
        print("📝 Generating interpretation report...")
        self._generate_interpretation_report()
        
        return self.dimension_meanings
    
    def _extract_factor_data(self, labeled_data, max_samples, batch_size):
        """Extract factor representations and associated data"""
        
        # Handle different input formats
        if hasattr(labeled_data, '__iter__') and not isinstance(labeled_data, np.ndarray):
            # It's a DataLoader or similar
            all_images = []
            all_labels = []
            all_semantic_factors = []
            all_attribute_factors = []
            all_reconstructions = []
            
            sample_count = 0
            with torch.no_grad():
                for batch in tqdm(labeled_data, desc="Processing batches"):
                    if sample_count >= max_samples:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        images = batch['data'].to(self.device)
                        labels = batch.get('label', batch.get('labels', None))
                    elif isinstance(batch, (list, tuple)):
                        images = batch[0].to(self.device)
                        labels = batch[1] if len(batch) > 1 else None
                    else:
                        images = batch.to(self.device)
                        labels = None
                    
                    # Ensure proper format
                    if images.dim() == 3:
                        images = images.unsqueeze(1)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Extract data
                    all_images.append(images.cpu())
                    if labels is not None:
                        all_labels.append(labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels))
                    
                    if hasattr(outputs, 'semantic_factors'):
                        all_semantic_factors.append(outputs.semantic_factors.cpu())
                    if hasattr(outputs, 'attribute_factors'):
                        all_attribute_factors.append(outputs.attribute_factors.cpu())
                    if hasattr(outputs, 'recon_x'):
                        all_reconstructions.append(outputs.recon_x.cpu())
                    
                    sample_count += images.shape[0]
            
            # Concatenate all data
            factor_data = {
                'images': torch.cat(all_images, dim=0),
                'labels': torch.cat(all_labels, dim=0) if all_labels else None,
                'semantic_factors': torch.cat(all_semantic_factors, dim=0) if all_semantic_factors else None,
                'attribute_factors': torch.cat(all_attribute_factors, dim=0) if all_attribute_factors else None,
                'reconstructions': torch.cat(all_reconstructions, dim=0) if all_reconstructions else None
            }
            
        else:
            # It's numpy arrays - convert and process
            if isinstance(labeled_data, tuple):
                images, labels = labeled_data[:2]
            else:
                images = labeled_data
                labels = None
            
            # Limit samples
            if len(images) > max_samples:
                indices = np.random.choice(len(images), max_samples, replace=False)
                images = images[indices]
                if labels is not None:
                    labels = labels[indices]
            
            # Process in batches
            all_semantic_factors = []
            all_attribute_factors = []
            all_reconstructions = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
                    batch_images = torch.tensor(images[i:i+batch_size], dtype=torch.float32).to(self.device)
                    
                    if batch_images.dim() == 3:
                        batch_images = batch_images.unsqueeze(1)
                    
                    outputs = self.model(batch_images)
                    
                    if hasattr(outputs, 'semantic_factors'):
                        all_semantic_factors.append(outputs.semantic_factors.cpu())
                    if hasattr(outputs, 'attribute_factors'):
                        all_attribute_factors.append(outputs.attribute_factors.cpu())
                    if hasattr(outputs, 'recon_x'):
                        all_reconstructions.append(outputs.recon_x.cpu())
            
            factor_data = {
                'images': torch.tensor(images, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                'semantic_factors': torch.cat(all_semantic_factors, dim=0) if all_semantic_factors else None,
                'attribute_factors': torch.cat(all_attribute_factors, dim=0) if all_attribute_factors else None,
                'reconstructions': torch.cat(all_reconstructions, dim=0) if all_reconstructions else None
            }
        
        return factor_data
    
    def _analyze_semantic_dimensions(self, factor_data):
        """Analyze what each semantic dimension captures"""
        semantic_factors = factor_data['semantic_factors']
        labels = factor_data['labels']
        
        if semantic_factors is None:
            return {}
        
        semantic_meanings = {}
        n_semantic_dims = semantic_factors.shape[1]
        
        print(f"  Analyzing {n_semantic_dims} semantic dimensions...")
        
        for dim_idx in range(n_semantic_dims):
            dim_values = semantic_factors[:, dim_idx].numpy()
            
            analysis = {
                'activation_stats': {
                    'mean': float(np.mean(dim_values)),
                    'std': float(np.std(dim_values)),
                    'min': float(np.min(dim_values)),
                    'max': float(np.max(dim_values)),
                    'sparsity': float(np.mean(dim_values < 0.1)),
                    'activation_rate': float(np.mean(dim_values > 0.5))
                }
            }
            
            # Correlation with labels if available
            if labels is not None:
                label_correlations = self._compute_label_correlations(dim_values, labels.numpy())
                analysis['label_correlations'] = label_correlations
                
                # Classification performance
                classification_acc = self._test_dimension_classification(dim_values, labels.numpy())
                analysis['classification_accuracy'] = classification_acc
            
            # Clustering analysis
            cluster_analysis = self._analyze_dimension_clustering(dim_values)
            analysis['clustering'] = cluster_analysis
            
            # Determine primary interpretation
            primary_meaning = self._determine_primary_meaning(analysis)
            analysis['primary_meaning'] = primary_meaning
            
            semantic_meanings[f'semantic_dim_{dim_idx}'] = analysis
        
        return semantic_meanings
    
    def _analyze_attribute_dimensions(self, factor_data):
        """Analyze what each attribute dimension captures"""
        attribute_factors = factor_data['attribute_factors']
        labels = factor_data['labels']
        
        if attribute_factors is None:
            return {}
        
        attribute_meanings = {}
        n_attribute_dims = attribute_factors.shape[1]
        
        print(f"  Analyzing {n_attribute_dims} attribute dimensions...")
        
        for dim_idx in range(n_attribute_dims):
            dim_values = attribute_factors[:, dim_idx].numpy()
            
            analysis = {
                'activation_stats': {
                    'mean': float(np.mean(dim_values)),
                    'std': float(np.std(dim_values)),
                    'min': float(np.min(dim_values)),
                    'max': float(np.max(dim_values)),
                    'sparsity': float(np.mean(dim_values < 0.1)),
                    'activation_rate': float(np.mean(dim_values > 0.5))
                }
            }
            
            # Correlation with labels if available
            if labels is not None:
                label_correlations = self._compute_label_correlations(dim_values, labels.numpy())
                analysis['label_correlations'] = label_correlations
                
                # Classification performance (should be lower for attribute dims)
                classification_acc = self._test_dimension_classification(dim_values, labels.numpy())
                analysis['classification_accuracy'] = classification_acc
            
            # Clustering analysis
            cluster_analysis = self._analyze_dimension_clustering(dim_values)
            analysis['clustering'] = cluster_analysis
            
            # Determine primary interpretation
            primary_meaning = self._determine_primary_meaning(analysis, is_attribute=True)
            analysis['primary_meaning'] = primary_meaning
            
            attribute_meanings[f'attribute_dim_{dim_idx}'] = analysis
        
        return attribute_meanings
    
    def _compute_label_correlations(self, dim_values, labels):
        """Compute correlations between dimension values and labels"""
        correlations = {}
        
        # For each unique label, compute correlation
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Create binary indicator for this label
            label_indicator = (labels == label).astype(float)
            
            # Compute correlation
            corr_pearson, p_pearson = pearsonr(dim_values, label_indicator)
            corr_spearman, p_spearman = spearmanr(dim_values, label_indicator)
            
            correlations[f'label_{int(label)}'] = {
                'pearson': float(corr_pearson),
                'pearson_p': float(p_pearson),
                'spearman': float(corr_spearman),
                'spearman_p': float(p_spearman)
            }
        
        # Find strongest correlation
        strongest_corr = max(correlations.items(), 
                           key=lambda x: abs(x[1]['pearson']))
        correlations['strongest'] = {
            'label': strongest_corr[0],
            'correlation': strongest_corr[1]['pearson']
        }
        
        return correlations
    
    def _test_dimension_classification(self, dim_values, labels):
        """Test how well a single dimension can classify labels"""
        try:
            # Split data
            n_train = int(0.8 * len(dim_values))
            indices = np.random.permutation(len(dim_values))
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            
            # Reshape for sklearn
            X_train = dim_values[train_idx].reshape(-1, 1)
            X_test = dim_values[test_idx].reshape(-1, 1)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            
            # Train classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            
            # Test accuracy
            accuracy = accuracy_score(y_test, clf.predict(X_test))
            
            return float(accuracy)
        except:
            return 0.0
    
    def _analyze_dimension_clustering(self, dim_values):
        """Analyze clustering properties of a dimension"""
        try:
            # Reshape for clustering
            X = dim_values.reshape(-1, 1)
            
            # Try different numbers of clusters
            cluster_results = {}
            for n_clusters in [2, 3, 5, 10]:
                if len(dim_values) >= n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X)
                    
                    # Compute silhouette-like metric
                    inertia = kmeans.inertia_
                    cluster_results[n_clusters] = {
                        'inertia': float(inertia),
                        'cluster_centers': kmeans.cluster_centers_.flatten().tolist()
                    }
            
            return cluster_results
        except:
            return {}
    
    def _determine_primary_meaning(self, analysis, is_attribute=False):
        """Determine the primary meaning/interpretation of a dimension"""
        meaning = {
            'type': 'attribute' if is_attribute else 'semantic',
            'confidence': 0.0,
            'description': 'Unknown'
        }
        
        # Check activation patterns
        stats = analysis['activation_stats']
        sparsity = stats['sparsity']
        activation_rate = stats['activation_rate']
        
        if sparsity > 0.8:
            meaning['description'] = 'Highly sparse - captures rare/specific features'
            meaning['confidence'] = 0.7
        elif activation_rate > 0.7:
            meaning['description'] = 'Highly active - captures common features'
            meaning['confidence'] = 0.6
        elif 0.3 < activation_rate < 0.7:
            meaning['description'] = 'Moderate activation - captures balanced features'
            meaning['confidence'] = 0.5
        
        # Check label correlations if available
        if 'label_correlations' in analysis and 'strongest' in analysis['label_correlations']:
            strongest_corr = abs(analysis['label_correlations']['strongest']['correlation'])
            if strongest_corr > 0.5:
                label = analysis['label_correlations']['strongest']['label']
                meaning['description'] = f'Strongly correlated with {label} (r={strongest_corr:.3f})'
                meaning['confidence'] = min(0.9, strongest_corr * 1.5)
            elif strongest_corr > 0.3:
                label = analysis['label_correlations']['strongest']['label']
                meaning['description'] = f'Moderately correlated with {label} (r={strongest_corr:.3f})'
                meaning['confidence'] = strongest_corr
        
        # Check classification performance
        if 'classification_accuracy' in analysis:
            acc = analysis['classification_accuracy']
            if acc > 0.7:
                meaning['description'] += f' - Good classification performance ({acc:.3f})'
                meaning['confidence'] = max(meaning['confidence'], acc)
            elif acc < 0.2 and is_attribute:
                meaning['description'] += ' - Low classification performance (good for attribute dim)'
                meaning['confidence'] = max(meaning['confidence'], 0.6)
        
        return meaning
    
    def _find_dimension_prototypes(self, factor_data, n_prototypes=5):
        """Find prototype images that maximally activate each dimension"""
        semantic_factors = factor_data['semantic_factors']
        attribute_factors = factor_data['attribute_factors']
        images = factor_data['images']
        
        self.prototypes = {}
        
        # Semantic dimension prototypes
        if semantic_factors is not None:
            self.prototypes['semantic'] = {}
            for dim_idx in range(semantic_factors.shape[1]):
                dim_values = semantic_factors[:, dim_idx]
                
                # Find top activating images
                top_indices = torch.argsort(dim_values, descending=True)[:n_prototypes]
                prototype_images = images[top_indices]
                prototype_values = dim_values[top_indices]
                
                self.prototypes['semantic'][f'dim_{dim_idx}'] = {
                    'images': prototype_images,
                    'values': prototype_values,
                    'indices': top_indices
                }
        
        # Attribute dimension prototypes
        if attribute_factors is not None:
            self.prototypes['attribute'] = {}
            for dim_idx in range(attribute_factors.shape[1]):
                dim_values = attribute_factors[:, dim_idx]
                
                # Find top activating images
                top_indices = torch.argsort(dim_values, descending=True)[:n_prototypes]
                prototype_images = images[top_indices]
                prototype_values = dim_values[top_indices]
                
                self.prototypes['attribute'][f'dim_{dim_idx}'] = {
                    'images': prototype_images,
                    'values': prototype_values,
                    'indices': top_indices
                }
    
    def _analyze_manipulation_effects(self, factor_data):
        """Analyze the effects of manipulating individual dimensions"""
        semantic_factors = factor_data['semantic_factors']
        attribute_factors = factor_data['attribute_factors']
        
        if semantic_factors is None or attribute_factors is None:
            return
        
        self.manipulation_effects = {}
        
        # Select a few representative samples for manipulation
        n_samples = min(5, semantic_factors.shape[0])
        sample_indices = torch.randperm(semantic_factors.shape[0])[:n_samples]
        
        sample_semantic = semantic_factors[sample_indices]
        sample_attribute = attribute_factors[sample_indices]
        
        # Test manipulation values
        manipulation_values = [-2, -1, 0, 1, 2]
        
        # Semantic dimension manipulations
        self.manipulation_effects['semantic'] = {}
        for dim_idx in range(min(8, semantic_factors.shape[1])):  # Limit to first 8 dims
            dim_effects = []
            
            for sample_idx in range(n_samples):
                sample_effects = []
                
                for value in manipulation_values:
                    # Create modified factors
                    modified_semantic = sample_semantic[sample_idx:sample_idx+1].clone()
                    modified_semantic[0, dim_idx] = value
                    
                    # Combine with original attribute factors
                    original_attribute = sample_attribute[sample_idx:sample_idx+1]
                    
                    # Generate reconstruction
                    with torch.no_grad():
                        combined_factors = torch.cat([modified_semantic, original_attribute], dim=1)
                        
                        # Use the decoder to generate image
                        try:
                            generated = self.model.decoder(combined_factors.to(self.device))
                            if hasattr(generated, 'reconstruction'):
                                sample_effects.append(generated.reconstruction.cpu())
                            else:
                                sample_effects.append(generated.cpu())
                        except:
                            # If direct decoding fails, skip this manipulation
                            continue
                
                if sample_effects:
                    dim_effects.append(torch.cat(sample_effects, dim=0))
            
            if dim_effects:
                self.manipulation_effects['semantic'][f'dim_{dim_idx}'] = dim_effects
        
        # Similar for attribute dimensions (simplified)
        self.manipulation_effects['attribute'] = {}
        for dim_idx in range(min(4, attribute_factors.shape[1])):  # Limit to first 4 dims
            # Similar manipulation logic for attribute dimensions
            # (Implementation similar to above but for attribute factors)
            pass
    
    def _generate_interpretation_report(self):
        """Generate a comprehensive interpretation report"""
        print("\n" + "="*80)
        print("🔍 DIMENSION INTERPRETATION REPORT")
        print("="*80)
        
        # Semantic dimensions summary
        if 'semantic' in self.dimension_meanings:
            print(f"\n📊 SEMANTIC DIMENSIONS ANALYSIS")
            print("-" * 50)
            
            semantic_dims = self.dimension_meanings['semantic']
            for dim_name, analysis in semantic_dims.items():
                print(f"\n{dim_name.upper()}:")
                print(f"  Primary meaning: {analysis['primary_meaning']['description']}")
                print(f"  Confidence: {analysis['primary_meaning']['confidence']:.3f}")
                print(f"  Activation rate: {analysis['activation_stats']['activation_rate']:.3f}")
                print(f"  Sparsity: {analysis['activation_stats']['sparsity']:.3f}")
                
                if 'classification_accuracy' in analysis:
                    print(f"  Classification accuracy: {analysis['classification_accuracy']:.3f}")
        
        # Attribute dimensions summary
        if 'attribute' in self.dimension_meanings:
            print(f"\n🎨 ATTRIBUTE DIMENSIONS ANALYSIS")
            print("-" * 50)
            
            attribute_dims = self.dimension_meanings['attribute']
            for dim_name, analysis in attribute_dims.items():
                print(f"\n{dim_name.upper()}:")
                print(f"  Primary meaning: {analysis['primary_meaning']['description']}")
                print(f"  Confidence: {analysis['primary_meaning']['confidence']:.3f}")
                print(f"  Activation rate: {analysis['activation_stats']['activation_rate']:.3f}")
                print(f"  Sparsity: {analysis['activation_stats']['sparsity']:.3f}")
                
                if 'classification_accuracy' in analysis:
                    print(f"  Classification accuracy: {analysis['classification_accuracy']:.3f}")
        
        # Overall summary
        print(f"\n📈 OVERALL SUMMARY")
        print("-" * 30)
        
        if 'semantic' in self.dimension_meanings:
            semantic_dims = self.dimension_meanings['semantic']
            high_conf_semantic = sum(1 for d in semantic_dims.values() 
                                   if d['primary_meaning']['confidence'] > 0.7)
            print(f"High-confidence semantic dimensions: {high_conf_semantic}/{len(semantic_dims)}")
        
        if 'attribute' in self.dimension_meanings:
            attribute_dims = self.dimension_meanings['attribute']
            high_conf_attribute = sum(1 for d in attribute_dims.values() 
                                    if d['primary_meaning']['confidence'] > 0.7)
            print(f"High-confidence attribute dimensions: {high_conf_attribute}/{len(attribute_dims)}")
    
    def visualize_dimension_analysis(self, save_prefix="dimension_analysis"):
        """Create comprehensive visualizations of dimension analysis"""
        
        # 1. Activation patterns heatmap
        self._plot_activation_heatmap(save_prefix)
        
        # 2. Prototype visualizations
        self._plot_dimension_prototypes(save_prefix)
        
        # 3. Correlation analysis
        self._plot_correlation_analysis(save_prefix)
        
        # 4. Manipulation effects (if available)
        if self.manipulation_effects:
            self._plot_manipulation_effects(save_prefix)
    
    def _plot_activation_heatmap(self, save_prefix):
        """Plot heatmap of dimension activation patterns"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Semantic dimensions
        if 'semantic' in self.dimension_meanings:
            semantic_data = []
            semantic_labels = []
            
            for dim_name, analysis in self.dimension_meanings['semantic'].items():
                stats = analysis['activation_stats']
                semantic_data.append([
                    stats['mean'], stats['std'], 
                    stats['activation_rate'], stats['sparsity']
                ])
                semantic_labels.append(dim_name.replace('semantic_dim_', 'S'))
            
            semantic_df = pd.DataFrame(semantic_data, 
                                     columns=['Mean', 'Std', 'Activation Rate', 'Sparsity'],
                                     index=semantic_labels)
            
            sns.heatmap(semantic_df.T, annot=True, fmt='.3f', ax=axes[0], cmap='viridis')
            axes[0].set_title('Semantic Dimensions')
        
        # Attribute dimensions
        if 'attribute' in self.dimension_meanings:
            attribute_data = []
            attribute_labels = []
            
            for dim_name, analysis in self.dimension_meanings['attribute'].items():
                stats = analysis['activation_stats']
                attribute_data.append([
                    stats['mean'], stats['std'], 
                    stats['activation_rate'], stats['sparsity']
                ])
                attribute_labels.append(dim_name.replace('attribute_dim_', 'A'))
            
            attribute_df = pd.DataFrame(attribute_data, 
                                      columns=['Mean', 'Std', 'Activation Rate', 'Sparsity'],
                                      index=attribute_labels)
            
            sns.heatmap(attribute_df.T, annot=True, fmt='.3f', ax=axes[1], cmap='plasma')
            axes[1].set_title('Attribute Dimensions')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_activation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_dimension_prototypes(self, save_prefix):
        """Plot prototype images for each dimension"""
        if not self.prototypes:
            return
        
        # Semantic prototypes
        if 'semantic' in self.prototypes:
            n_dims = min(8, len(self.prototypes['semantic']))
            n_prototypes = 3
            
            fig, axes = plt.subplots(n_dims, n_prototypes, figsize=(12, 3*n_dims))
            if n_dims == 1:
                axes = axes.reshape(1, -1)
            
            for i, (dim_name, proto_data) in enumerate(list(self.prototypes['semantic'].items())[:n_dims]):
                for j in range(n_prototypes):
                    if j < len(proto_data['images']):
                        img = proto_data['images'][j]
                        if img.shape[0] == 1:  # Grayscale
                            axes[i, j].imshow(img[0], cmap='gray')
                        else:  # RGB
                            axes[i, j].imshow(img.permute(1, 2, 0))
                        
                        axes[i, j].set_title(f'{dim_name}\nVal: {proto_data["values"][j]:.3f}')
                        axes[i, j].axis('off')
            
            plt.suptitle('Semantic Dimension Prototypes')
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_semantic_prototypes.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        # Similar for attribute prototypes
        if 'attribute' in self.prototypes:
            n_dims = min(4, len(self.prototypes['attribute']))
            n_prototypes = 3
            
            fig, axes = plt.subplots(n_dims, n_prototypes, figsize=(12, 3*n_dims))
            if n_dims == 1:
                axes = axes.reshape(1, -1)
            
            for i, (dim_name, proto_data) in enumerate(list(self.prototypes['attribute'].items())[:n_dims]):
                for j in range(n_prototypes):
                    if j < len(proto_data['images']):
                        img = proto_data['images'][j]
                        if img.shape[0] == 1:  # Grayscale
                            axes[i, j].imshow(img[0], cmap='gray')
                        else:  # RGB
                            axes[i, j].imshow(img.permute(1, 2, 0))
                        
                        axes[i, j].set_title(f'{dim_name}\nVal: {proto_data["values"][j]:.3f}')
                        axes[i, j].axis('off')
            
            plt.suptitle('Attribute Dimension Prototypes')
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_attribute_prototypes.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def _plot_correlation_analysis(self, save_prefix):
        """Plot correlation analysis results"""
        if 'semantic' not in self.dimension_meanings:
            return
        
        # Extract correlation data
        semantic_corrs = []
        semantic_labels = []
        
        for dim_name, analysis in self.dimension_meanings['semantic'].items():
            if 'label_correlations' in analysis and 'strongest' in analysis['label_correlations']:
                corr_val = analysis['label_correlations']['strongest']['correlation']
                semantic_corrs.append(abs(corr_val))
                semantic_labels.append(dim_name.replace('semantic_dim_', 'S'))
        
        if semantic_corrs:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(semantic_labels, semantic_corrs)
            plt.title('Strongest Label Correlations by Dimension')
            plt.xlabel('Dimension')
            plt.ylabel('Absolute Correlation')
            plt.xticks(rotation=45)
            
            # Color bars based on correlation strength
            for bar, corr in zip(bars, semantic_corrs):
                if corr > 0.7:
                    bar.set_color('green')
                elif corr > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_correlations.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def _plot_manipulation_effects(self, save_prefix):
        """Plot dimension manipulation effects"""
        # This would show the effects of manipulating individual dimensions
        # Implementation depends on the specific manipulation results stored
        pass
    
    def analyze_tensorboard_logs(self, log_dir):
        """
        Analyze TensorBoard logs to extract training dynamics and factor evolution.
        
        Args:
            log_dir: Path to TensorBoard log directory
        """
        print(f"📊 Analyzing TensorBoard logs from: {log_dir}")
        
        try:
            # Load TensorBoard logs
            ea = EventAccumulator(log_dir)
            ea.Reload()
            
            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']
            print(f"Available scalar metrics: {len(scalar_tags)}")
            
            # Analyze factor evolution over training
            factor_evolution = {}
            
            # Extract factor-related metrics
            for tag in scalar_tags:
                if 'factors/' in tag:
                    scalar_events = ea.Scalars(tag)
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    
                    factor_evolution[tag] = {
                        'steps': steps,
                        'values': values
                    }
            
            # Plot factor evolution
            if factor_evolution:
                self._plot_factor_evolution(factor_evolution, save_prefix="tensorboard_analysis")
            
            return factor_evolution
            
        except Exception as e:
            print(f"❌ Error analyzing TensorBoard logs: {e}")
            return None
    
    def _plot_factor_evolution(self, factor_evolution, save_prefix):
        """Plot how factors evolved during training"""
        
        # Separate semantic and attribute metrics
        semantic_metrics = {k: v for k, v in factor_evolution.items() if 'semantic' in k}
        attribute_metrics = {k: v for k, v in factor_evolution.items() if 'attribute' in k}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Semantic factor means
        for tag, data in semantic_metrics.items():
            if 'mean' in tag and 'semantic_' not in tag.split('/')[-1]:  # Overall mean, not individual factors
                axes[0, 0].plot(data['steps'], data['values'], label=tag.split('/')[-1])
        axes[0, 0].set_title('Semantic Factor Activation (Training)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].legend()
        
        # Semantic factor sparsity
        for tag, data in semantic_metrics.items():
            if 'sparsity' in tag:
                axes[0, 1].plot(data['steps'], data['values'], label=tag.split('/')[-1])
        axes[0, 1].set_title('Semantic Factor Sparsity (Training)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Sparsity')
        axes[0, 1].legend()
        
        # Attribute factor means
        for tag, data in attribute_metrics.items():
            if 'mean' in tag and 'attribute_' not in tag.split('/')[-1]:  # Overall mean
                axes[1, 0].plot(data['steps'], data['values'], label=tag.split('/')[-1])
        axes[1, 0].set_title('Attribute Factor Activation (Training)')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Mean Activation')
        axes[1, 0].legend()
        
        # Attribute factor sparsity
        for tag, data in attribute_metrics.items():
            if 'sparsity' in tag:
                axes[1, 1].plot(data['steps'], data['values'], label=tag.split('/')[-1])
        axes[1, 1].set_title('Attribute Factor Sparsity (Training)')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Sparsity')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_factor_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_interpretation_results(self, filepath="dimension_interpretation_results.json"):
        """Save interpretation results to JSON file"""
        # Convert torch tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Prepare data for saving (exclude large image data)
        save_data = {
            'semantic_meanings': self.dimension_meanings.get('semantic', {}),
            'attribute_meanings': self.dimension_meanings.get('attribute', {}),
            # Skip prototypes (too large) and manipulation effects
        }
        
        save_data = convert_for_json(save_data)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Interpretation results saved to: {filepath}")


# Example usage function
def run_dimension_interpretation_example():
    """Example of how to use the DimensionInterpreter"""
    
    # Load your trained model
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    
    # This would be replaced with your actual model loading code
    # model = load_trained_model(model_path)
    
    # Load test data
    train_data, test_data, train_labels, test_labels, n_classes = load_dataset(
        'mnist', input_size=28, train_size=1000, test_size=1000
    )
    
    # Create labeled dataset
    labeled_data = (test_data, test_labels)
    
    # Initialize interpreter
    # interpreter = DimensionInterpreter(model, device='mps')
    
    # Run interpretation
    # results = interpreter.interpret_dimensions(labeled_data, max_samples=1000)
    
    # Create visualizations
    # interpreter.visualize_dimension_analysis()
    
    # Analyze TensorBoard logs
    # log_dir = "disentangled_vae_output/tensorboard_logs"
    # interpreter.analyze_tensorboard_logs(log_dir)
    
    # Save results
    # interpreter.save_interpretation_results()
    
    print("Example setup complete. Uncomment the lines above to run with your trained model.")

if __name__ == "__main__":
    run_dimension_interpretation_example() 