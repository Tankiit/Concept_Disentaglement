from pythae.models import BaseTrainerConfig
from pythae.models import VAEConfig
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_architecture import BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from config import DisentangledVAEConfig
import torch
from dataset_specific import FactorizedDisentangledVAEWithAnnealing, DisentangledVAETrainingPipelineWithAnnealing, DisentangledVAETrainerWithAnnealing

import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Disentangled VAE Training with Beta Annealing")
    parser.add_argument("--dataset", type=str, default="dsprites", choices=["dsprites", "xyobject", "shapes3d", "mnist", "pathmnist", "chestmnist", "dermamnist"], help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="disentangled_vae_output", help="Output directory")
    parser.add_argument("--train_size", type=int, default=None, help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=None, help="Number of test samples")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--analyze", action="store_true", help="Run disentanglement analysis")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--max_analysis_samples", type=int, default=1000, help="Maximum number of samples for analysis")
    return parser.parse_args()

def main():
    """Main execution function with dataset-specific configurations"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Validate dataset
    supported_datasets = ['dsprites', 'xyobject', 'shapes3d', 'mnist', 'pathmnist', 'chestmnist', 'dermamnist']
    
    if args.dataset in ['dsprites', 'xyobject', 'shapes3d']:
        # Use our custom dataset configuration
        dataset_config_name = args.dataset
    else:
        # Use default configuration for other datasets
        dataset_config_name = 'xyobject'  # default
        print(f"Using default XYObject configuration for dataset: {args.dataset}")
    
    # Create dataset-specific configuration
    disentangle_config = DisentangledVAEConfig(
        dataset_name=dataset_config_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Override epochs if specified
    if args.epochs:
        disentangle_config.epochs = args.epochs
    
    print("=" * 80)
    print("DISENTANGLED VAE TRAINING WITH BETA ANNEALING")
    print("=" * 80)
    print(f"Dataset: {args.dataset} (config: {disentangle_config.dataset_name})")
    print(f"Input: {disentangle_config.input_channels}×{disentangle_config.input_size}×{disentangle_config.input_size}")
    print(f"Architecture:")
    print(f"  - Encoder: {4 if disentangle_config.dataset_name != 'shapes3d' else 6} conv layers")
    print(f"  - Bottleneck: {disentangle_config.bottleneck_dim} units")
    print(f"  - Dropout: {disentangle_config.encoder_dropout}")
    print(f"  - Decoder type: {disentangle_config.decoder_type}")
    print(f"Latent dimensions:")
    print(f"  - Content (semantic): {disentangle_config.semantic_dim}")
    print(f"  - Style (attribute): {disentangle_config.attribute_dim}")
    print(f"  - Total: {disentangle_config.semantic_dim + disentangle_config.attribute_dim}")
    print(f"Training:")
    print(f"  - Epochs: {disentangle_config.epochs}")
    print(f"  - Beta schedule: {disentangle_config.beta_start} → {disentangle_config.beta_end}")
    print(f"  - Batch size: {disentangle_config.batch_size}")
    print(f"  - Learning rate: {disentangle_config.learning_rate}")
    print("=" * 80)
    
    # Device configuration
    device = torch.device(disentangle_config.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    try:
        # For the custom datasets, we need to determine the correct number of channels
        if args.dataset == 'dsprites':
            input_channels = 1
        elif args.dataset in ['xyobject', 'shapes3d']:
            input_channels = 3
        else:
            # Use the existing load_dataset function which will handle this
            input_channels = None
        
        train_data, test_data, num_classes = load_dataset(
            args.dataset,
            input_size=disentangle_config.input_size,
            train_size=args.train_size,
            test_size=args.test_size
        )
        
        # Update config with actual number of classes
        disentangle_config.num_classes = num_classes
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Number of classes: {num_classes}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        if args.dataset in ['dsprites', 'xyobject', 'shapes3d']:
            print(f"\nNote: You need to implement the dataset loader for {args.dataset}")
            print("The loader should return (train_data, test_data, num_classes)")
        return
    
    # Configure Pythae trainer
    training_config = BaseTrainerConfig(
        output_dir=args.output_dir or f'disentangled_vae_{args.dataset}_output',
        num_epochs=disentangle_config.epochs,
        learning_rate=disentangle_config.learning_rate,
        per_device_train_batch_size=disentangle_config.batch_size,
        per_device_eval_batch_size=disentangle_config.batch_size,
        train_dataloader_num_workers=args.num_workers,
        eval_dataloader_num_workers=args.num_workers,
        steps_saving=5,
        steps_predict=10,  # Log every 10 steps
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.01, "betas": (0.9, 0.999)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 3, "factor": 0.5, "mode": "min"}
    )
    
    # Configure VAE model
    model_config = VAEConfig(
        input_dim=(disentangle_config.input_channels, disentangle_config.input_size, disentangle_config.input_size),
        latent_dim=disentangle_config.semantic_dim + disentangle_config.attribute_dim
    )
    
    # Add dataset-specific config
    model_config.dataset_name = disentangle_config.dataset_name
    model_config.input_channels = disentangle_config.input_channels
    model_config.input_size = disentangle_config.input_size
    model_config.semantic_dim = disentangle_config.semantic_dim
    model_config.attribute_dim = disentangle_config.attribute_dim
    model_config.bottleneck_dim = disentangle_config.bottleneck_dim
    model_config.encoder_dropout = disentangle_config.encoder_dropout
    model_config.decoder_type = disentangle_config.decoder_type
    
    # Create model with dataset-specific architecture
    print("\nCreating Factorized Disentangled VAE with Beta Annealing...")
    model = FactorizedDisentangledVAEWithAnnealing(
        model_config=model_config,
        encoder=DatasetSpecificEncoder(model_config),
        decoder=DatasetSpecificDecoder(model_config),
        disentangle_config=disentangle_config
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    factor_params = (sum(p.numel() for p in model.semantic_factorization.parameters() if p.requires_grad) +
                    sum(p.numel() for p in model.attribute_factorization.parameters() if p.requires_grad))
    
    print(f"\nModel architecture summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  - Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"  - Factorization: {factor_params:,} ({factor_params/total_params*100:.1f}%)")
    
    # Create training pipeline with beta annealing
    print("\nSetting up training pipeline with TensorBoard logging...")
    pipeline = DisentangledVAETrainingPipelineWithAnnealing(
        training_config=training_config,
        model=model
    )
    
    # Train the model
    print(f"\nStarting training for {disentangle_config.epochs} epochs...")
    print(f"Beta will anneal from {disentangle_config.beta_start} to {disentangle_config.beta_end}")
    
    start_time = time.time()
    try:
        pipeline(
            train_data=train_data,
            eval_data=test_data
        )
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training completed successfully in {elapsed_time/60:.1f} minutes!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save the model
    save_path = f'{training_config.output_dir}/final_model'
    print(f"\nSaving trained model to {save_path}...")
    try:
        model.save(save_path)
        print("✓ Model saved successfully!")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    # Analyze if requested
    if args.analyze:
        print("\n" + "="*80)
        print("RUNNING DISENTANGLEMENT ANALYSIS")
        print("="*80)
        
        try:
            # Prepare data for analysis
            if hasattr(test_data, '__getitem__'):
                # It's a Dataset object
                test_samples = []
                test_labels = []
                for i in range(min(len(test_data), args.max_analysis_samples)):
                    sample = test_data[i]
                    if isinstance(sample, dict):
                        test_samples.append(sample['data'].numpy())
                        test_labels.append(sample.get('label', i))
                    else:
                        test_samples.append(sample[0].numpy())
                        test_labels.append(sample[1] if len(sample) > 1 else i)
                test_samples = np.array(test_samples)
                test_labels = np.array(test_labels)
            else:
                # It's already numpy arrays
                test_samples = test_data[:args.max_analysis_samples]
                test_labels = np.arange(len(test_samples))  # Dummy labels if not available
            
            # Run analysis
            results = analyze_disentanglement(
                model, test_samples, test_labels, 
                device=device, max_samples=args.max_analysis_samples
            )
            
            print("\nDisentanglement Analysis Results:")
            print("-" * 50)
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        print(f"{key}: mean={np.mean(value):.3f}, std={np.std(value):.3f}")
                    else:
                        print(f"{key}: {len(value)} entries")
                elif isinstance(value, (int, float)):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize if requested
    if args.visualize:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        try:
            # Prepare data for visualization
            if hasattr(test_data, '__getitem__'):
                viz_samples = []
                viz_labels = []
                for i in range(min(len(test_data), 100)):
                    sample = test_data[i]
                    if isinstance(sample, dict):
                        viz_samples.append(sample['data'].numpy())
                        viz_labels.append(sample.get('label', i))
                    else:
                        viz_samples.append(sample[0].numpy())
                        viz_labels.append(sample[1] if len(sample) > 1 else i)
                viz_samples = np.array(viz_samples)
                viz_labels = np.array(viz_labels)
            else:
                viz_samples = test_data[:100]
                viz_labels = np.arange(len(viz_samples))
            
            visualize_disentanglement_results(
                model, viz_samples, viz_labels, 
                device=device, 
                save_prefix=f"{training_config.output_dir}/{args.dataset}_disentangled"
            )
            print("✓ Visualizations saved!")
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model configuration: {disentangle_config.dataset_name}")
    print(f"Training time: {elapsed_time/60:.1f} minutes")
    print(f"Final beta value: {disentangle_config.beta_end}")
    print(f"\nOutput directory: {training_config.output_dir}")
    print(f"Model saved at: {save_path}")
    print(f"\nTensorBoard logs: {training_config.output_dir}/tensorboard_logs/")
    print(f"To view: tensorboard --logdir={training_config.output_dir}/tensorboard_logs/")
    print("=" * 80)

if __name__ == "__main__":
    main()