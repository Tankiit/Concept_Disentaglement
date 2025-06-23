# Dimension Interpretation for Disentangled VAE

This guide shows you how to use the `DimensionInterpreter` class to analyze what each dimension in your disentangled VAE learned, leveraging your TensorBoard logs for comprehensive insights.

## Overview

The `DimensionInterpreter` provides post-hoc analysis to discover the semantic meaning of each dimension after training. It combines:

1. **Factor extraction** from your trained model
2. **Correlation analysis** with ground truth labels  
3. **Activation pattern analysis** (sparsity, specialization)
4. **Classification performance** testing
5. **TensorBoard log integration** for training dynamics

## Quick Start

### 1. Basic Dimension Analysis

```python
from dimension_interpreter import DimensionInterpreter
from simple_model_analysis import load_model_weights_directly
from disentangled_vae_pythae_integration import load_dataset

# Load your trained model
model_path = "disentangled_vae_output/VAE_training_2025-06-22_18-21-20/final_model"
model = load_model_weights_directly(model_path, device='mps')

# Load test data
_, test_data, _, test_labels, _ = load_dataset('mnist', input_size=28, test_size=1000)

# Run interpretation
interpreter = DimensionInterpreter(model, device='mps')
results = interpreter.interpret_dimensions((test_data, test_labels), max_samples=1000)

# Create visualizations
interpreter.visualize_analysis(save_prefix="my_analysis")
```

### 2. Using the Simple Script

Run the pre-built analysis script:

```bash
python run_dimension_analysis.py
```

This will:
- Load your trained model
- Extract factor representations
- Analyze dimension meanings
- Create visualizations
- Save results to JSON

## What the Analysis Provides

### For Each Dimension, You Get:

1. **Activation Statistics**:
   - Mean activation level
   - Sparsity (how often it's inactive)
   - Activation rate (how often it's highly active)

2. **Label Correlations**:
   - Correlation with each digit class (for MNIST)
   - Strongest correlation identified
   - Statistical significance

3. **Classification Performance**:
   - How well this single dimension can classify digits
   - Indicates semantic vs. attribute role

4. **Interpretation**:
   - Primary meaning (e.g., "Strongly correlated with digit 3")
   - Confidence score (0-1)
   - Semantic vs. attribute classification

### Example Output:

```
SEMANTIC_DIM_5:
  Interpretation: Strongly correlated with label_3 (r=0.847)
  Confidence: 0.847
  Activation rate: 0.234
  Sparsity: 0.678
  Classification accuracy: 0.892
```

## Understanding Your Results

### Good Disentanglement Indicators:

✅ **High correlation with specific labels** (r > 0.7)
✅ **Good classification accuracy** for semantic dimensions (> 0.7)
✅ **Low classification accuracy** for attribute dimensions (< 0.3)
✅ **Moderate sparsity** (0.3 - 0.7)
✅ **High interpretation confidence** (> 0.7)

### Warning Signs:

⚠️ **Very high sparsity** (> 0.8) - dimensions may be unused
⚠️ **Very low sparsity** (< 0.1) - dimensions may be too dense
⚠️ **Low confidence scores** (< 0.5) - unclear interpretations
⚠️ **Many unused dimensions** - consider reducing latent dimensionality

## TensorBoard Integration

### Analyzing Training Logs

```python
from tensorboard_analyzer import TensorBoardAnalyzer

# Analyze your training logs
analyzer = TensorBoardAnalyzer("disentangled_vae_output/tensorboard_logs")
analyzer.load_logs()
factor_evolution = analyzer.analyze_factors()
analyzer.plot_evolution()
```

### What TensorBoard Analysis Shows:

1. **Factor Evolution**: How semantic/attribute factors changed during training
2. **Sparsity Trends**: Whether sparsity increased/decreased over time
3. **Stability**: Whether factors converged or kept oscillating
4. **Loss Balance**: Relative contribution of different loss components

## Improving Your Model

Based on the analysis results, you can:

### If Dimensions Have Low Confidence:
```python
# In your training config:
disentangle_config.factorization_weight = 0.05  # Increase from 0.01
disentangle_config.n_semantic_factors = 64      # Increase capacity
```

### If Too Many Sparse Dimensions:
```python
disentangle_config.sparsity_weight = 0.005      # Reduce from 0.001
disentangle_config.target_sparsity = 0.4        # Increase from 0.3
```

### If Poor Semantic-Label Correlation:
```python
# Add auxiliary classification loss
disentangle_config.classification_weight = 0.1
```

## Generated Files

After running the analysis, you'll get:

- `dimension_analysis_patterns.png` - Heatmap of activation patterns
- `dimension_analysis_correlations.png` - Bar chart of label correlations  
- `dimension_interpretation_results.json` - Detailed results in JSON format
- `tensorboard_analysis.png` - Training evolution plots (if TensorBoard available)

## Advanced Usage

### Custom Analysis

```python
# Analyze specific aspects
interpreter = DimensionInterpreter(model)
results = interpreter.interpret_dimensions(data)

# Extract specific insights
semantic_results = results['semantic']
for dim_name, analysis in semantic_results.items():
    if analysis['interpretation']['confidence'] > 0.8:
        print(f"High-confidence dimension: {dim_name}")
        print(f"  Meaning: {analysis['interpretation']['description']}")
```

### Integration with Your Training Loop

```python
# During training, periodically check dimension quality
if epoch % 10 == 0:
    interpreter = DimensionInterpreter(model)
    results = interpreter.interpret_dimensions(val_data, max_samples=500)
    
    # Early stopping based on interpretation quality
    avg_confidence = np.mean([d['interpretation']['confidence'] 
                             for d in results['semantic'].values()])
    
    if avg_confidence > 0.8:
        print("Good disentanglement achieved!")
```

## Troubleshooting

### Common Issues:

1. **"Model output doesn't have expected factor attributes"**
   - Check that you're using the correct model architecture
   - Ensure the model was trained with disentanglement losses

2. **"No correlation data"**
   - Make sure you're providing labels with your data
   - Check that labels are in the correct format (torch.LongTensor)

3. **"TensorBoard not available"**
   - Install TensorBoard: `pip install tensorboard`
   - Or skip TensorBoard analysis and use dimension interpretation only

### Getting Help:

If you encounter issues:
1. Check that your model path is correct
2. Verify your data format matches the expected input
3. Ensure all dependencies are installed
4. Try with a smaller sample size first (`max_samples=100`)

## Next Steps

After analyzing your dimensions:

1. **Experiment with hyperparameters** based on recommendations
2. **Implement factor manipulation** to validate interpretations
3. **Add human evaluation** for dimension quality
4. **Test on different datasets** to verify generalization

The dimension interpreter gives you insights into what your model learned, helping you improve disentanglement quality and interpretability! 