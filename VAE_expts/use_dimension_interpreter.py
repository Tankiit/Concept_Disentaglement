#!/usr/bin/env python3
"""
Example script showing how to use the DimensionInterpreter with your trained model
and TensorBoard logs for comprehensive dimension analysis.
"""

import torch
import numpy as np
import os
from dimension_interpreter import DimensionInterpreter
from simple_model_analysis import load_model_weights_directly
from disentangled_vae_pythae_integration import load_dataset

def main():
    """Main function demonstrating dimension interpretation workflow"""
    
    print("🚀 Starting Dimension Interpretation Analysis")
    print("=" * 60)
    
    # Configuration
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    tensorboard_log_dir = "disentangled_vae_output/tensorboard_logs"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"📁 Model path: {model_path}")
    print(f"📊 TensorBoard logs: {tensorboard_log_dir}")
    print(f"💻 Device: {device}")
    
    # Step 1: Load the trained model
    print(f"\n🔄 Loading trained model...")
    model = load_model_weights_directly(model_path, device)
    
    if model is None:
        print("❌ Failed to load model. Please check the model path.")
        return
    
    print("✅ Model loaded successfully!")
    
    # Step 2: Load test data
    print(f"\n📊 Loading test data...")
    train_data, test_data, train_labels, test_labels, n_classes = load_dataset(
        'mnist', input_size=28, train_size=1000, test_size=1000
    )
    
    print(f"✅ Loaded {len(test_data)} test samples with {n_classes} classes")
    
    # Step 3: Initialize the DimensionInterpreter
    print(f"\n🔍 Initializing DimensionInterpreter...")
    interpreter = DimensionInterpreter(model, device=device)
    
    # Step 4: Run dimension interpretation analysis
    print(f"\n🧠 Running dimension interpretation analysis...")
    
    # Create labeled dataset tuple
    labeled_data = (test_data, test_labels)
    
    # Run the main interpretation
    results = interpreter.interpret_dimensions(labeled_data, max_samples=1000)
    
    if results is None:
        print("❌ Failed to extract interpretations")
        return
    
    print("✅ Dimension interpretation completed!")
    
    # Step 5: Create visualizations
    print(f"\n📈 Creating visualizations...")
    interpreter.visualize_analysis(save_prefix="mnist_dimension_analysis")
    
    # Step 6: Analyze TensorBoard logs (if available)
    if os.path.exists(tensorboard_log_dir):
        print(f"\n📊 Analyzing TensorBoard logs...")
        try:
            # You would need to implement this in the DimensionInterpreter class
            # For now, we'll show what could be analyzed
            analyze_tensorboard_trends(tensorboard_log_dir)
        except Exception as e:
            print(f"⚠️  Could not analyze TensorBoard logs: {e}")
    else:
        print(f"⚠️  TensorBoard log directory not found: {tensorboard_log_dir}")
    
    # Step 7: Generate detailed analysis report
    print(f"\n📝 Generating detailed analysis report...")
    generate_detailed_report(interpreter, results)
    
    # Step 8: Save results
    print(f"\n💾 Saving results...")
    interpreter.save_results("mnist_dimension_interpretation.json")
    
    # Step 9: Provide recommendations
    print(f"\n💡 Generating recommendations...")
    provide_recommendations(results)
    
    print(f"\n🎉 Analysis complete! Check the generated files:")
    print(f"  - mnist_dimension_analysis_patterns.png")
    print(f"  - mnist_dimension_analysis_correlations.png") 
    print(f"  - mnist_dimension_interpretation.json")
    print(f"  - detailed_analysis_report.txt")

def analyze_tensorboard_trends(log_dir):
    """
    Analyze TensorBoard logs to understand training dynamics.
    This function shows what could be analyzed from the logs.
    """
    print(f"  📊 TensorBoard Analysis (Conceptual):")
    print(f"    - Factor activation evolution during training")
    print(f"    - Sparsity trends over epochs")
    print(f"    - Loss component contributions")
    print(f"    - Prototype stability across training")
    
    # Example of what you could extract:
    trends_to_analyze = [
        "factors/semantic_mean",
        "factors/semantic_sparsity", 
        "factors/attribute_mean",
        "factors/attribute_sparsity",
        "loss/factor_sparsity",
        "loss/orthogonality",
        "loss/total_correlation"
    ]
    
    print(f"    - Available metrics to analyze: {len(trends_to_analyze)}")
    
    # You could implement actual TensorBoard log parsing here
    # using tensorboard.backend.event_processing.event_accumulator

def generate_detailed_report(interpreter, results):
    """Generate a detailed analysis report"""
    
    report_lines = []
    report_lines.append("DETAILED DIMENSION INTERPRETATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Summary statistics
    if 'semantic' in results:
        semantic_dims = results['semantic']
        high_conf_semantic = sum(1 for d in semantic_dims.values() 
                               if d['interpretation']['confidence'] > 0.7)
        
        report_lines.append(f"SEMANTIC DIMENSIONS SUMMARY:")
        report_lines.append(f"  Total dimensions: {len(semantic_dims)}")
        report_lines.append(f"  High-confidence interpretations: {high_conf_semantic}")
        report_lines.append(f"  Success rate: {high_conf_semantic/len(semantic_dims)*100:.1f}%")
        report_lines.append("")
        
        # Detailed breakdown
        for dim_name, analysis in semantic_dims.items():
            report_lines.append(f"{dim_name.upper()}:")
            report_lines.append(f"  Interpretation: {analysis['interpretation']['description']}")
            report_lines.append(f"  Confidence: {analysis['interpretation']['confidence']:.3f}")
            report_lines.append(f"  Activation Statistics:")
            report_lines.append(f"    Mean: {analysis['activation_stats']['mean']:.4f}")
            report_lines.append(f"    Std: {analysis['activation_stats']['std']:.4f}")
            report_lines.append(f"    Sparsity: {analysis['activation_stats']['sparsity']:.4f}")
            report_lines.append(f"    Activation Rate: {analysis['activation_stats']['activation_rate']:.4f}")
            
            if 'classification_accuracy' in analysis:
                report_lines.append(f"  Classification Accuracy: {analysis['classification_accuracy']:.4f}")
            
            if 'strongest_correlation' in analysis:
                corr_info = analysis['strongest_correlation']
                report_lines.append(f"  Strongest Correlation: {corr_info['label']} (r={corr_info['correlation']:.3f})")
            
            report_lines.append("")
    
    if 'attribute' in results:
        attribute_dims = results['attribute']
        high_conf_attribute = sum(1 for d in attribute_dims.values() 
                                if d['interpretation']['confidence'] > 0.7)
        
        report_lines.append(f"ATTRIBUTE DIMENSIONS SUMMARY:")
        report_lines.append(f"  Total dimensions: {len(attribute_dims)}")
        report_lines.append(f"  High-confidence interpretations: {high_conf_attribute}")
        report_lines.append(f"  Success rate: {high_conf_attribute/len(attribute_dims)*100:.1f}%")
        report_lines.append("")
        
        # Detailed breakdown (similar to semantic)
        for dim_name, analysis in attribute_dims.items():
            report_lines.append(f"{dim_name.upper()}:")
            report_lines.append(f"  Interpretation: {analysis['interpretation']['description']}")
            report_lines.append(f"  Confidence: {analysis['interpretation']['confidence']:.3f}")
            report_lines.append(f"  Activation Statistics:")
            report_lines.append(f"    Mean: {analysis['activation_stats']['mean']:.4f}")
            report_lines.append(f"    Std: {analysis['activation_stats']['std']:.4f}")
            report_lines.append(f"    Sparsity: {analysis['activation_stats']['sparsity']:.4f}")
            report_lines.append(f"    Activation Rate: {analysis['activation_stats']['activation_rate']:.4f}")
            
            if 'classification_accuracy' in analysis:
                report_lines.append(f"  Classification Accuracy: {analysis['classification_accuracy']:.4f}")
            
            report_lines.append("")
    
    # Save report
    with open("detailed_analysis_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    
    print("✅ Detailed report saved to: detailed_analysis_report.txt")

def provide_recommendations(results):
    """Provide recommendations based on the analysis"""
    
    print("💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    # Analyze semantic dimensions
    if 'semantic' in results:
        semantic_dims = results['semantic']
        
        # Check for highly correlated dimensions
        high_corr_dims = []
        for dim_name, analysis in semantic_dims.items():
            if 'strongest_correlation' in analysis:
                corr = abs(analysis['strongest_correlation']['correlation'])
                if corr > 0.7:
                    high_corr_dims.append((dim_name, corr))
        
        if high_corr_dims:
            print(f"✅ Strong semantic-label correlations found in {len(high_corr_dims)} dimensions")
            print(f"   This suggests good disentanglement of semantic content")
        else:
            print(f"⚠️  Few strong semantic-label correlations found")
            print(f"   Consider: Increasing semantic_dim or adjusting loss weights")
        
        # Check sparsity patterns
        sparse_dims = []
        for dim_name, analysis in semantic_dims.items():
            sparsity = analysis['activation_stats']['sparsity']
            if sparsity > 0.8:
                sparse_dims.append(dim_name)
        
        if len(sparse_dims) > len(semantic_dims) * 0.5:
            print(f"⚠️  Many dimensions are highly sparse ({len(sparse_dims)}/{len(semantic_dims)})")
            print(f"   Consider: Reducing sparsity_weight or increasing target_sparsity")
        
    # Analyze attribute dimensions
    if 'attribute' in results:
        attribute_dims = results['attribute']
        
        # Check if attribute dimensions have low classification accuracy (good!)
        low_class_dims = []
        for dim_name, analysis in attribute_dims.items():
            if 'classification_accuracy' in analysis:
                acc = analysis['classification_accuracy']
                if acc < 0.3:  # Low classification accuracy is good for attribute dims
                    low_class_dims.append(dim_name)
        
        if len(low_class_dims) > len(attribute_dims) * 0.5:
            print(f"✅ Attribute dimensions show low classification accuracy")
            print(f"   This suggests good separation from semantic content")
        else:
            print(f"⚠️  Some attribute dimensions correlate with labels")
            print(f"   Consider: Increasing orthogonality_weight")
    
    # Overall recommendations
    print(f"\n🔧 TRAINING RECOMMENDATIONS:")
    print(f"  - Monitor factor evolution in TensorBoard during training")
    print(f"  - Adjust loss weights based on dimension interpretations")
    print(f"  - Consider prototype-based evaluation for dimension quality")
    print(f"  - Use manipulation experiments to validate interpretations")

def run_quick_dimension_check():
    """Quick check to see if dimension interpretation is working"""
    
    print("🔍 Quick Dimension Interpretation Check")
    print("-" * 40)
    
    # This is a simplified version for testing
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load model
    model = load_model_weights_directly(model_path, device)
    if model is None:
        print("❌ Cannot load model for quick check")
        return
    
    # Load small amount of data
    _, test_data, _, test_labels, _ = load_dataset('mnist', input_size=28, test_size=100)
    
    # Quick interpretation
    interpreter = DimensionInterpreter(model, device=device)
    results = interpreter.interpret_dimensions((test_data, test_labels), max_samples=100)
    
    if results:
        print("✅ Quick check successful!")
        print(f"   Semantic dimensions: {len(results.get('semantic', {}))}")
        print(f"   Attribute dimensions: {len(results.get('attribute', {}))}")
    else:
        print("❌ Quick check failed")

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Full analysis (recommended)
    main()
    
    # Quick check only
    # run_quick_dimension_check() 