#!/usr/bin/env python3
"""
Comprehensive Analysis Example: Combining Dimension Interpretation with TensorBoard Analysis

This script demonstrates how to use the DimensionInterpreter class with your trained 
disentangled VAE model and leverage TensorBoard logs for complete analysis.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from dimension_interpreter import DimensionInterpreter
from tensorboard_analyzer import TensorBoardAnalyzer, TENSORBOARD_AVAILABLE
from simple_model_analysis import load_model_weights_directly
from disentangled_vae_pythae_integration import load_dataset

def main():
    """
    Complete analysis workflow combining:
    1. Model loading and factor extraction
    2. Dimension interpretation analysis
    3. TensorBoard log analysis
    4. Combined insights and recommendations
    """
    
    print("🚀 COMPREHENSIVE DISENTANGLED VAE ANALYSIS")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_path': "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model",
        'tensorboard_logs': "disentangled_vae_output/tensorboard_logs",
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'max_samples': 1000,
        'dataset': 'mnist'
    }
    
    print_config(config)
    
    # Step 1: Load the trained model
    print("\n" + "="*50)
    print("STEP 1: LOADING TRAINED MODEL")
    print("="*50)
    
    model = load_model_weights_directly(config['model_path'], config['device'])
    if model is None:
        print("❌ Failed to load model. Please check the model path.")
        return
    
    print("✅ Model loaded successfully!")
    
    # Step 2: Load test data
    print("\n" + "="*50)
    print("STEP 2: LOADING TEST DATA")
    print("="*50)
    
    train_data, test_data, train_labels, test_labels, n_classes = load_dataset(
        config['dataset'], 
        input_size=28, 
        train_size=config['max_samples'], 
        test_size=config['max_samples']
    )
    
    print(f"✅ Loaded {len(test_data)} test samples with {n_classes} classes")
    
    # Step 3: Run Dimension Interpretation Analysis
    print("\n" + "="*50)
    print("STEP 3: DIMENSION INTERPRETATION ANALYSIS")
    print("="*50)
    
    interpreter = DimensionInterpreter(model, device=config['device'])
    
    # Run interpretation
    labeled_data = (test_data, test_labels)
    interpretation_results = interpreter.interpret_dimensions(
        labeled_data, 
        max_samples=config['max_samples']
    )
    
    if interpretation_results is None:
        print("❌ Failed to extract dimension interpretations")
        return
    
    print("✅ Dimension interpretation completed!")
    
    # Step 4: TensorBoard Log Analysis
    print("\n" + "="*50)
    print("STEP 4: TENSORBOARD LOG ANALYSIS")
    print("="*50)
    
    tensorboard_results = None
    if os.path.exists(config['tensorboard_logs']) and TENSORBOARD_AVAILABLE:
        try:
            tb_analyzer = TensorBoardAnalyzer(config['tensorboard_logs'])
            if tb_analyzer.load_logs():
                tensorboard_results = tb_analyzer.analyze_factors()
                tb_analyzer.plot_evolution()
                print("✅ TensorBoard analysis completed!")
            else:
                print("⚠️  Could not load TensorBoard logs")
        except Exception as e:
            print(f"⚠️  TensorBoard analysis failed: {e}")
    else:
        print("⚠️  TensorBoard logs not available or TensorBoard not installed")
    
    # Step 5: Create Comprehensive Visualizations
    print("\n" + "="*50)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*50)
    
    create_comprehensive_visualizations(
        interpreter, 
        interpretation_results, 
        tensorboard_results,
        config
    )
    
    # Step 6: Generate Combined Analysis Report
    print("\n" + "="*50)
    print("STEP 6: GENERATING ANALYSIS REPORT")
    print("="*50)
    
    generate_combined_report(
        interpretation_results, 
        tensorboard_results, 
        config
    )
    
    # Step 7: Provide Actionable Recommendations
    print("\n" + "="*50)
    print("STEP 7: ACTIONABLE RECOMMENDATIONS")
    print("="*50)
    
    provide_actionable_recommendations(
        interpretation_results, 
        tensorboard_results, 
        config
    )
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("Generated files:")
    print("  - comprehensive_dimension_analysis.png")
    print("  - combined_analysis_report.txt")
    print("  - actionable_recommendations.txt")

def print_config(config):
    """Print configuration details"""
    print(f"📁 Model path: {config['model_path']}")
    print(f"📊 TensorBoard logs: {config['tensorboard_logs']}")
    print(f"💻 Device: {config['device']}")
    print(f"📈 Max samples: {config['max_samples']}")
    print(f"🗂️  Dataset: {config['dataset']}")

def create_comprehensive_visualizations(interpreter, interpretation_results, tensorboard_results, config):
    """Create comprehensive visualizations combining all analyses"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Dimension interpretation heatmap (top left)
    ax1 = plt.subplot(3, 3, (1, 2))
    plot_dimension_interpretation_heatmap(interpretation_results, ax1)
    
    # 2. Factor correlation analysis (top right)
    ax2 = plt.subplot(3, 3, 3)
    plot_factor_correlations(interpretation_results, ax2)
    
    # 3. TensorBoard factor evolution (middle row)
    if tensorboard_results:
        ax3 = plt.subplot(3, 3, (4, 6))
        plot_tensorboard_evolution(tensorboard_results, ax3)
    
    # 4. Dimension confidence scores (bottom left)
    ax4 = plt.subplot(3, 3, 7)
    plot_confidence_scores(interpretation_results, ax4)
    
    # 5. Sparsity analysis (bottom middle)
    ax5 = plt.subplot(3, 3, 8)
    plot_sparsity_analysis(interpretation_results, ax5)
    
    # 6. Summary statistics (bottom right)
    ax6 = plt.subplot(3, 3, 9)
    plot_summary_statistics(interpretation_results, ax6)
    
    plt.suptitle(f'Comprehensive Disentangled VAE Analysis - {config["dataset"].upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comprehensive_dimension_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization saved!")

def plot_dimension_interpretation_heatmap(results, ax):
    """Plot dimension interpretation as heatmap"""
    if 'semantic' not in results:
        ax.text(0.5, 0.5, 'No semantic data', ha='center', va='center')
        return
    
    semantic_data = []
    labels = []
    
    for dim_name, analysis in results['semantic'].items():
        stats = analysis['activation_stats']
        semantic_data.append([
            stats['mean'], 
            stats['activation_rate'], 
            stats['sparsity'],
            analysis['interpretation']['confidence']
        ])
        labels.append(dim_name.replace('semantic_dim_', 'S'))
    
    if semantic_data:
        data_array = np.array(semantic_data).T
        im = ax.imshow(data_array, aspect='auto', cmap='viridis')
        ax.set_title('Semantic Dimensions Analysis')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['Mean', 'Activation Rate', 'Sparsity', 'Confidence'])
        plt.colorbar(im, ax=ax, shrink=0.8)

def plot_factor_correlations(results, ax):
    """Plot strongest correlations"""
    if 'semantic' not in results:
        ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
        return
    
    correlations = []
    labels = []
    
    for dim_name, analysis in results['semantic'].items():
        if 'strongest_correlation' in analysis:
            corr = abs(analysis['strongest_correlation']['correlation'])
            correlations.append(corr)
            labels.append(dim_name.replace('semantic_dim_', 'S'))
    
    if correlations:
        bars = ax.bar(labels, correlations)
        ax.set_title('Strongest Label Correlations')
        ax.set_ylabel('Absolute Correlation')
        ax.tick_params(axis='x', rotation=45)
        
        # Color by strength
        for bar, corr in zip(bars, correlations):
            if corr > 0.7:
                bar.set_color('green')
            elif corr > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')

def plot_tensorboard_evolution(tb_results, ax):
    """Plot TensorBoard factor evolution"""
    ax.text(0.5, 0.5, 'TensorBoard Evolution\n(Implementation depends on log format)', 
            ha='center', va='center', fontsize=12)
    ax.set_title('Factor Evolution During Training')

def plot_confidence_scores(results, ax):
    """Plot interpretation confidence scores"""
    if 'semantic' not in results:
        ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
        return
    
    confidences = []
    labels = []
    
    for dim_name, analysis in results['semantic'].items():
        conf = analysis['interpretation']['confidence']
        confidences.append(conf)
        labels.append(dim_name.replace('semantic_dim_', 'S'))
    
    if confidences:
        bars = ax.bar(labels, confidences)
        ax.set_title('Interpretation Confidence')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Color by confidence level
        for bar, conf in zip(bars, confidences):
            if conf > 0.8:
                bar.set_color('green')
            elif conf > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

def plot_sparsity_analysis(results, ax):
    """Plot sparsity analysis"""
    semantic_sparsity = []
    attribute_sparsity = []
    
    if 'semantic' in results:
        for analysis in results['semantic'].values():
            semantic_sparsity.append(analysis['activation_stats']['sparsity'])
    
    if 'attribute' in results:
        for analysis in results['attribute'].values():
            attribute_sparsity.append(analysis['activation_stats']['sparsity'])
    
    data = []
    labels = []
    
    if semantic_sparsity:
        data.append(semantic_sparsity)
        labels.append('Semantic')
    
    if attribute_sparsity:
        data.append(attribute_sparsity)
        labels.append('Attribute')
    
    if data:
        ax.boxplot(data, labels=labels)
        ax.set_title('Sparsity Distribution')
        ax.set_ylabel('Sparsity')

def plot_summary_statistics(results, ax):
    """Plot summary statistics"""
    stats_text = []
    
    if 'semantic' in results:
        semantic_dims = results['semantic']
        high_conf = sum(1 for d in semantic_dims.values() 
                       if d['interpretation']['confidence'] > 0.7)
        stats_text.append(f"Semantic Dimensions: {len(semantic_dims)}")
        stats_text.append(f"High Confidence: {high_conf}")
        stats_text.append(f"Success Rate: {high_conf/len(semantic_dims)*100:.1f}%")
    
    if 'attribute' in results:
        attribute_dims = results['attribute']
        high_conf = sum(1 for d in attribute_dims.values() 
                       if d['interpretation']['confidence'] > 0.7)
        stats_text.append(f"")
        stats_text.append(f"Attribute Dimensions: {len(attribute_dims)}")
        stats_text.append(f"High Confidence: {high_conf}")
        stats_text.append(f"Success Rate: {high_conf/len(attribute_dims)*100:.1f}%")
    
    ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Summary Statistics')
    ax.axis('off')

def generate_combined_report(interpretation_results, tensorboard_results, config):
    """Generate comprehensive analysis report"""
    
    report_lines = []
    report_lines.append("COMPREHENSIVE DISENTANGLED VAE ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Dataset: {config['dataset'].upper()}")
    report_lines.append(f"Model: {config['model_path']}")
    report_lines.append(f"Analysis Date: {np.datetime64('now', 'D')}")
    report_lines.append("")
    
    # Dimension interpretation summary
    if interpretation_results:
        report_lines.append("DIMENSION INTERPRETATION SUMMARY")
        report_lines.append("-" * 40)
        
        if 'semantic' in interpretation_results:
            semantic_dims = interpretation_results['semantic']
            high_conf_semantic = sum(1 for d in semantic_dims.values() 
                                   if d['interpretation']['confidence'] > 0.7)
            
            report_lines.append(f"Semantic Dimensions:")
            report_lines.append(f"  Total: {len(semantic_dims)}")
            report_lines.append(f"  High Confidence: {high_conf_semantic}")
            report_lines.append(f"  Success Rate: {high_conf_semantic/len(semantic_dims)*100:.1f}%")
            report_lines.append("")
            
            # Top interpretations
            top_dims = sorted(semantic_dims.items(), 
                            key=lambda x: x[1]['interpretation']['confidence'], 
                            reverse=True)[:5]
            
            report_lines.append("Top 5 Semantic Dimension Interpretations:")
            for dim_name, analysis in top_dims:
                conf = analysis['interpretation']['confidence']
                desc = analysis['interpretation']['description']
                report_lines.append(f"  {dim_name}: {desc} (confidence: {conf:.3f})")
            report_lines.append("")
        
        if 'attribute' in interpretation_results:
            attribute_dims = interpretation_results['attribute']
            high_conf_attribute = sum(1 for d in attribute_dims.values() 
                                    if d['interpretation']['confidence'] > 0.7)
            
            report_lines.append(f"Attribute Dimensions:")
            report_lines.append(f"  Total: {len(attribute_dims)}")
            report_lines.append(f"  High Confidence: {high_conf_attribute}")
            report_lines.append(f"  Success Rate: {high_conf_attribute/len(attribute_dims)*100:.1f}%")
            report_lines.append("")
    
    # TensorBoard analysis summary
    if tensorboard_results:
        report_lines.append("TENSORBOARD TRAINING ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append("Training dynamics analysis completed.")
        report_lines.append("See tensorboard_analysis.png for evolution plots.")
        report_lines.append("")
    
    # Overall assessment
    report_lines.append("OVERALL ASSESSMENT")
    report_lines.append("-" * 20)
    
    if interpretation_results and 'semantic' in interpretation_results:
        semantic_success = high_conf_semantic / len(semantic_dims) if 'semantic_dims' in locals() else 0
        
        if semantic_success > 0.7:
            report_lines.append("✅ Excellent disentanglement quality")
        elif semantic_success > 0.5:
            report_lines.append("✅ Good disentanglement quality")
        elif semantic_success > 0.3:
            report_lines.append("⚠️  Fair disentanglement quality - room for improvement")
        else:
            report_lines.append("❌ Poor disentanglement quality - significant improvements needed")
    
    # Save report
    with open("combined_analysis_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    print("✅ Combined analysis report saved!")

def provide_actionable_recommendations(interpretation_results, tensorboard_results, config):
    """Provide specific, actionable recommendations"""
    
    recommendations = []
    recommendations.append("ACTIONABLE RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    recommendations.append("=" * 60)
    recommendations.append("")
    
    # Based on dimension interpretation
    if interpretation_results and 'semantic' in interpretation_results:
        semantic_dims = interpretation_results['semantic']
        
        # Check sparsity levels
        avg_sparsity = np.mean([d['activation_stats']['sparsity'] 
                               for d in semantic_dims.values()])
        
        if avg_sparsity < 0.3:
            recommendations.append("🔧 INCREASE SPARSITY:")
            recommendations.append("  - Increase sparsity_weight from current value to 0.01-0.05")
            recommendations.append("  - Consider adjusting target_sparsity to 0.2-0.4")
            recommendations.append("")
        elif avg_sparsity > 0.8:
            recommendations.append("🔧 REDUCE EXCESSIVE SPARSITY:")
            recommendations.append("  - Decrease sparsity_weight")
            recommendations.append("  - Check if too many dimensions are unused")
            recommendations.append("")
        
        # Check interpretation confidence
        high_conf_count = sum(1 for d in semantic_dims.values() 
                             if d['interpretation']['confidence'] > 0.7)
        
        if high_conf_count < len(semantic_dims) * 0.5:
            recommendations.append("🔧 IMPROVE DIMENSION INTERPRETABILITY:")
            recommendations.append("  - Increase semantic_dim to give more capacity")
            recommendations.append("  - Adjust factorization_weight to encourage cleaner factors")
            recommendations.append("  - Consider auxiliary classification loss")
            recommendations.append("")
        
        # Check for unused dimensions
        unused_dims = sum(1 for d in semantic_dims.values() 
                         if d['activation_stats']['activation_rate'] < 0.1)
        
        if unused_dims > len(semantic_dims) * 0.3:
            recommendations.append("🔧 REDUCE UNUSED DIMENSIONS:")
            recommendations.append(f"  - {unused_dims}/{len(semantic_dims)} dimensions are rarely active")
            recommendations.append("  - Consider reducing semantic_dim")
            recommendations.append("  - Or adjust initialization/learning rates")
            recommendations.append("")
    
    # Training-specific recommendations
    recommendations.append("🔧 TRAINING IMPROVEMENTS:")
    recommendations.append("  - Monitor factor evolution in TensorBoard during training")
    recommendations.append("  - Use early stopping based on dimension interpretation quality")
    recommendations.append("  - Consider curriculum learning: start with higher reconstruction weight")
    recommendations.append("  - Experiment with different optimizers (AdamW vs Adam)")
    recommendations.append("")
    
    # Architecture recommendations
    recommendations.append("🔧 ARCHITECTURE CONSIDERATIONS:")
    recommendations.append("  - If semantic dims have low confidence, try different encoder architectures")
    recommendations.append("  - Consider batch normalization in factorization layers")
    recommendations.append("  - Experiment with different activation functions in factor layers")
    recommendations.append("")
    
    # Evaluation recommendations
    recommendations.append("🔧 EVALUATION IMPROVEMENTS:")
    recommendations.append("  - Implement factor manipulation experiments")
    recommendations.append("  - Add prototype-based evaluation metrics")
    recommendations.append("  - Consider human evaluation of dimension interpretability")
    recommendations.append("  - Test on out-of-distribution data")
    
    # Save recommendations
    with open("actionable_recommendations.txt", "w") as f:
        f.write("\n".join(recommendations))
    
    print("✅ Actionable recommendations saved!")
    
    # Print key recommendations
    print("\n💡 KEY RECOMMENDATIONS:")
    print("-" * 30)
    if interpretation_results and 'semantic' in interpretation_results:
        semantic_dims = interpretation_results['semantic']
        avg_sparsity = np.mean([d['activation_stats']['sparsity'] 
                               for d in semantic_dims.values()])
        
        if avg_sparsity < 0.3:
            print("  🔧 Increase sparsity_weight to improve factor selectivity")
        
        high_conf_count = sum(1 for d in semantic_dims.values() 
                             if d['interpretation']['confidence'] > 0.7)
        
        if high_conf_count < len(semantic_dims) * 0.5:
            print("  🔧 Improve interpretability with auxiliary classification loss")
            print("  🔧 Consider increasing semantic dimension capacity")

if __name__ == "__main__":
    main() 