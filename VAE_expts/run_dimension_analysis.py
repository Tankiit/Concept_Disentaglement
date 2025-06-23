#!/usr/bin/env python3
"""
Simple script to run dimension analysis using the DimensionInterpreter
"""

import torch
import numpy as np
from dimension_interpreter import DimensionInterpreter
from simple_model_analysis import load_model_weights_directly
from disentangled_vae_pythae_integration import load_dataset

def run_analysis():
    """Run dimension interpretation analysis"""
    
    print("🔍 Starting Dimension Analysis")
    print("=" * 40)
    
    # Configuration
    model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Load model
    print("\n🔄 Loading model...")
    model = load_model_weights_directly(model_path, device)
    if model is None:
        print("❌ Failed to load model")
        return
    print("✅ Model loaded successfully!")
    
    # Load test data
    print("\n📊 Loading test data...")
    _, test_data, _, test_labels, n_classes = load_dataset('mnist', input_size=28, test_size=1000)
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # Initialize interpreter
    print("\n🧠 Initializing DimensionInterpreter...")
    interpreter = DimensionInterpreter(model, device=device)
    
    # Run interpretation
    print("\n🔍 Running dimension interpretation...")
    labeled_data = (test_data, test_labels)
    results = interpreter.interpret_dimensions(labeled_data, max_samples=1000)
    
    if results is None:
        print("❌ Failed to extract interpretations")
        return
    
    print("✅ Dimension interpretation completed!")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    interpreter.visualize_analysis(save_prefix="dimension_analysis")
    
    # Save results
    print("\n💾 Saving results...")
    interpreter.save_results("dimension_interpretation_results.json")
    
    # Print summary
    print_summary(results)
    
    print("\n🎉 Analysis complete!")
    print("Generated files:")
    print("  - dimension_analysis_patterns.png")
    print("  - dimension_analysis_correlations.png")
    print("  - dimension_interpretation_results.json")

def print_summary(results):
    """Print analysis summary"""
    
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 25)
    
    if 'semantic' in results:
        semantic_dims = results['semantic']
        high_conf_semantic = sum(1 for d in semantic_dims.values() 
                               if d['interpretation']['confidence'] > 0.7)
        
        print(f"Semantic Dimensions: {len(semantic_dims)}")
        print(f"High Confidence: {high_conf_semantic}")
        print(f"Success Rate: {high_conf_semantic/len(semantic_dims)*100:.1f}%")
        
        # Show top interpretations
        top_dims = sorted(semantic_dims.items(), 
                         key=lambda x: x[1]['interpretation']['confidence'], 
                         reverse=True)[:3]
        
        print(f"\nTop 3 Interpretations:")
        for dim_name, analysis in top_dims:
            conf = analysis['interpretation']['confidence']
            desc = analysis['interpretation']['description']
            print(f"  {dim_name}: {desc} (confidence: {conf:.3f})")
    
    if 'attribute' in results:
        attribute_dims = results['attribute']
        high_conf_attribute = sum(1 for d in attribute_dims.values() 
                                if d['interpretation']['confidence'] > 0.7)
        
        print(f"\nAttribute Dimensions: {len(attribute_dims)}")
        print(f"High Confidence: {high_conf_attribute}")
        print(f"Success Rate: {high_conf_attribute/len(attribute_dims)*100:.1f}%")

if __name__ == "__main__":
    run_analysis() 