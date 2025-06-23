# Concept Disentanglement with VAE

A comprehensive framework for training and analyzing disentangled Variational Autoencoders (VAEs) with interpretable latent representations. This repository focuses on separating semantic and attribute factors in learned representations, with advanced post-hoc analysis tools.

## 🚀 Features

- **Disentangled VAE Implementation**: Separates semantic and attribute factors using sparse factorization layers
- **Dimension Interpretation**: Post-hoc analysis to discover what each latent dimension learned
- **TensorBoard Integration**: Real-time monitoring of factor evolution during training
- **Comprehensive Analysis Tools**: Multiple analysis scripts for different aspects of disentanglement
- **Flexible Architecture**: Supports different input sizes and datasets (MNIST, MedMNIST)

## 📁 Repository Structure

### Core Implementation
- `disentangled_vae_pythae_integration.py` - Main VAE implementation with Pythae integration
- `dimension_interpreter.py` - Post-hoc dimension interpretation analysis
- `tensorboard_analyzer.py` - TensorBoard log analysis for training dynamics

### Analysis Scripts
- `run_dimension_analysis.py` - Simple script to run dimension interpretation
- `simple_model_analysis.py` - Basic model analysis and factor extraction
- `model_analysis_rqs.py` - Research question-focused analysis
- `quick_model_analysis.py` - Quick validation of model performance
- `comprehensive_analysis_example.py` - Complete analysis workflow example
- `use_dimension_interpreter.py` - Advanced dimension interpretation usage
- `use_original_for_analysis.py` - Analysis using original model checkpoints

### Utilities
- `XYSquares.py` - Synthetic dataset generation for testing
- `validation_report.txt` - Model validation results

### Documentation
- `DIMENSION_ANALYSIS_README.md` - Detailed guide for dimension interpretation
- `README.md` - This file

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Tankiit/Concept_Disentaglement.git
cd Concept_Disentaglement
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
pip install pythae  # For VAE framework
pip install tensorboard  # For training monitoring (optional)
pip install medmnist  # For medical imaging datasets (optional)
```

## 🚀 Quick Start

### 1. Train a Disentangled VAE

```python
from disentangled_vae_pythae_integration import main

# Train with default MNIST configuration
main()
```

### 2. Analyze Learned Dimensions

```python
# Run dimension interpretation analysis
python run_dimension_analysis.py
```

This will:
- Load your trained model
- Extract factor representations from test data
- Analyze what each dimension learned
- Generate visualizations and reports

### 3. View Results

The analysis generates several files:
- `dimension_analysis_patterns.png` - Heatmap of activation patterns
- `dimension_analysis_correlations.png` - Label correlation analysis
- `dimension_interpretation_results.json` - Detailed results

## 📊 Understanding Your Results

### Good Disentanglement Indicators:
✅ High correlation with specific labels (r > 0.7)  
✅ Good classification accuracy for semantic dimensions (> 0.7)  
✅ Low classification accuracy for attribute dimensions (< 0.3)  
✅ Moderate sparsity (0.3 - 0.7)  
✅ High interpretation confidence (> 0.7)  

### Example Output:
```
SEMANTIC_DIM_5:
  Interpretation: Strongly correlated with label_3 (r=0.847)
  Confidence: 0.847
  Activation rate: 0.234
  Sparsity: 0.678
  Classification accuracy: 0.892
```

## 🔧 Configuration

Key hyperparameters in `DisentangledVAEConfig`:

```python
@dataclass
class DisentangledVAEConfig:
    semantic_dim: int = 16          # Semantic latent dimension
    attribute_dim: int = 8          # Attribute latent dimension
    n_semantic_factors: int = 32    # Number of semantic factors
    n_attribute_factors: int = 16   # Number of attribute factors
    sparsity_weight: float = 0.001  # Sparsity regularization
    orthogonality_weight: float = 0.01  # Orthogonality between factors
```

## 📈 Monitoring Training

The framework includes comprehensive TensorBoard logging:

```bash
tensorboard --logdir=disentangled_vae_output/tensorboard_logs/
```

Logged metrics include:
- Factor activation statistics
- Sparsity evolution
- Loss component contributions
- Sample reconstructions
- Factor prototypes

## 🔍 Advanced Analysis

### Custom Dimension Interpretation

```python
from dimension_interpreter import DimensionInterpreter

interpreter = DimensionInterpreter(model, device='cuda')
results = interpreter.interpret_dimensions(labeled_data, max_samples=1000)

# Extract high-confidence interpretations
semantic_results = results['semantic']
for dim_name, analysis in semantic_results.items():
    if analysis['interpretation']['confidence'] > 0.8:
        print(f"High-confidence: {dim_name}")
        print(f"Meaning: {analysis['interpretation']['description']}")
```

### TensorBoard Log Analysis

```python
from tensorboard_analyzer import TensorBoardAnalyzer

analyzer = TensorBoardAnalyzer("path/to/tensorboard/logs")
analyzer.load_logs()
factor_evolution = analyzer.analyze_factors()
analyzer.plot_evolution()
```

## 📚 Research Applications

This framework is designed for research in:
- **Disentangled Representation Learning**
- **Interpretable AI**
- **Factor Analysis in Deep Learning**
- **Concept Discovery**
- **Representation Quality Assessment**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

## 🙏 Acknowledgments

- Built on the [Pythae](https://github.com/clementchadebec/pythae) VAE framework
- Inspired by recent advances in disentangled representation learning
- TensorBoard integration for comprehensive training monitoring

---

**Note**: This repository focuses on concept disentanglement and interpretability. For production use, consider additional validation and testing on your specific datasets. 