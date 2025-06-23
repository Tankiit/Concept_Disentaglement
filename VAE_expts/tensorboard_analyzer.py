#!/usr/bin/env python3
"""
TensorBoard Log Analyzer for Disentangled VAE

This module analyzes TensorBoard logs to understand how factors evolved during training
and provides insights into the learning dynamics of disentangled representations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

class TensorBoardAnalyzer:
    """Analyze TensorBoard logs for disentangled VAE training insights"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data = {}
        self.analysis_results = {}
        
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is required for log analysis")
    
    def load_logs(self):
        """Load TensorBoard logs"""
        print(f"📊 Loading TensorBoard logs from: {self.log_dir}")
        
        if not TENSORBOARD_AVAILABLE:
            print("TensorBoard not available")
            return False
            
        try:
            ea = EventAccumulator(self.log_dir)
            ea.Reload()
            
            scalar_tags = ea.Tags()['scalars']
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                self.data[tag] = {'steps': steps, 'values': values}
            
            print(f"✅ Successfully loaded {len(self.data)} metrics")
            return True
        except Exception as e:
            print(f"❌ Error loading logs: {e}")
            return False
    
    def analyze_factors(self):
        """Analyze how factors evolved during training"""
        print("🔍 Analyzing factor evolution...")
        
        factor_data = {}
        
        for tag, data in self.data.items():
            if 'factors/' in tag:
                parts = tag.split('/')
                category = parts[1]  # semantic or attribute
                metric = parts[-1]   # mean, sparsity, etc.
                
                if category not in factor_data:
                    factor_data[category] = {}
                factor_data[category][metric] = data
        
        self.analysis_results['factor_evolution'] = factor_data
        
        # Analyze trends
        trends = {}
        for category, metrics in factor_data.items():
            trends[category] = {}
            
            for metric_name, metric_data in metrics.items():
                values = np.array(metric_data['values'])
                
                # Calculate trend statistics
                if len(values) > 1:
                    # Linear trend
                    x = np.arange(len(values))
                    trend_coeff = np.polyfit(x, values, 1)[0]
                    
                    # Stability (coefficient of variation in last 20% of training)
                    last_20_percent = max(1, len(values) // 5)
                    recent_values = values[-last_20_percent:]
                    stability = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
                    
                    trends[category][metric_name] = {
                        'trend': float(trend_coeff),
                        'stability': float(stability),
                        'final_value': float(values[-1]),
                        'initial_value': float(values[0]),
                        'change': float(values[-1] - values[0])
                    }
        
        self.analysis_results['factor_trends'] = trends
        return trends
    
    def analyze_loss_components(self):
        """Analyze loss component evolution"""
        print("📈 Analyzing loss components...")
        
        loss_metrics = {}
        
        for tag, data in self.data.items():
            if 'loss/' in tag:
                loss_name = tag.split('/')[-1]
                loss_metrics[loss_name] = data
        
        self.analysis_results['loss_evolution'] = loss_metrics
        
        # Analyze loss balance
        loss_balance = {}
        if loss_metrics:
            # Get final values for each loss component
            for loss_name, loss_data in loss_metrics.items():
                if loss_data['values']:
                    final_value = loss_data['values'][-1]
                    loss_balance[loss_name] = final_value
            
            # Calculate relative contributions
            total_loss = loss_balance.get('total', sum(loss_balance.values()))
            if total_loss > 0:
                loss_proportions = {k: v/total_loss for k, v in loss_balance.items()}
                loss_balance['proportions'] = loss_proportions
        
        self.analysis_results['loss_balance'] = loss_balance
        return loss_balance
    
    def analyze_training_stability(self):
        """Analyze training stability and convergence"""
        print("📊 Analyzing training stability...")
        
        stability_metrics = {}
        
        # Check for oscillations, convergence, etc.
        for tag, data in self.data.items():
            values = np.array(data['values'])
            
            if len(values) > 10:
                # Calculate stability metrics
                
                # 1. Coefficient of variation in last 25% of training
                last_quarter = max(1, len(values) // 4)
                recent_values = values[-last_quarter:]
                cv = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
                
                # 2. Trend in last 25%
                x = np.arange(len(recent_values))
                if len(recent_values) > 1:
                    trend = np.polyfit(x, recent_values, 1)[0]
                else:
                    trend = 0
                
                # 3. Oscillation detection (count sign changes in differences)
                diffs = np.diff(values)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                oscillation_rate = sign_changes / len(diffs) if len(diffs) > 0 else 0
                
                stability_metrics[tag] = {
                    'coefficient_of_variation': float(cv),
                    'recent_trend': float(trend),
                    'oscillation_rate': float(oscillation_rate),
                    'converged': cv < 0.1 and abs(trend) < 1e-6
                }
        
        self.analysis_results['stability'] = stability_metrics
        return stability_metrics
    
    def generate_insights(self):
        """Generate high-level insights from the analysis"""
        print("💡 Generating insights...")
        
        insights = []
        
        # Factor evolution insights
        if 'factor_trends' in self.analysis_results:
            trends = self.analysis_results['factor_trends']
            
            for category, metrics in trends.items():
                if 'sparsity' in metrics:
                    sparsity_trend = metrics['sparsity']['trend']
                    sparsity_final = metrics['sparsity']['final_value']
                    
                    if sparsity_trend > 0.001:
                        insights.append(f"✅ {category.title()} factors became sparser during training (final: {sparsity_final:.3f})")
                    elif sparsity_final > 0.7:
                        insights.append(f"✅ {category.title()} factors achieved good sparsity ({sparsity_final:.3f})")
                    else:
                        insights.append(f"⚠️  {category.title()} factors may not be sparse enough ({sparsity_final:.3f})")
                
                if 'mean' in metrics:
                    activation_stability = metrics['mean']['stability']
                    if activation_stability < 0.1:
                        insights.append(f"✅ {category.title()} factor activations are stable")
                    else:
                        insights.append(f"⚠️  {category.title()} factor activations are unstable")
        
        # Loss balance insights
        if 'loss_balance' in self.analysis_results:
            balance = self.analysis_results['loss_balance']
            
            if 'proportions' in balance:
                props = balance['proportions']
                
                if 'reconstruction' in props and props['reconstruction'] > 0.8:
                    insights.append("⚠️  Reconstruction loss dominates - consider increasing regularization weights")
                
                if 'factor_sparsity' in props and props['factor_sparsity'] < 0.05:
                    insights.append("⚠️  Sparsity loss contribution is very low - consider increasing sparsity_weight")
                
                if 'orthogonality' in props and props['orthogonality'] > 0.3:
                    insights.append("⚠️  Orthogonality loss is high - factors may be too entangled")
        
        # Stability insights
        if 'stability' in self.analysis_results:
            stability = self.analysis_results['stability']
            
            unstable_metrics = [tag for tag, metrics in stability.items() 
                              if metrics['coefficient_of_variation'] > 0.2]
            
            if len(unstable_metrics) > len(stability) * 0.3:
                insights.append("⚠️  Many metrics are unstable - consider reducing learning rate")
            
            converged_metrics = [tag for tag, metrics in stability.items() 
                               if metrics['converged']]
            
            convergence_rate = len(converged_metrics) / len(stability) if stability else 0
            if convergence_rate > 0.8:
                insights.append("✅ Training appears to have converged well")
            elif convergence_rate < 0.3:
                insights.append("⚠️  Training may need more epochs to converge")
        
        self.analysis_results['insights'] = insights
        return insights
    
    def create_visualizations(self, save_prefix="tensorboard_analysis"):
        """Create comprehensive visualizations"""
        
        # 1. Factor evolution over time
        self._plot_factor_evolution(save_prefix)
        
        # 2. Loss component evolution
        self._plot_loss_evolution(save_prefix)
        
        # 3. Training stability analysis
        self._plot_stability_analysis(save_prefix)
        
        # 4. Summary dashboard
        self._create_summary_dashboard(save_prefix)
    
    def _plot_factor_evolution(self, save_prefix):
        """Plot factor evolution over training"""
        if 'factor_evolution' not in self.analysis_results:
            return
        
        factor_data = self.analysis_results['factor_evolution']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Semantic factors
        if 'semantic' in factor_data:
            semantic_data = factor_data['semantic']
            
            # Mean activation
            if 'mean' in semantic_data:
                steps = semantic_data['mean']['steps']
                values = semantic_data['mean']['values']
                axes[0, 0].plot(steps, values, label='Semantic Mean', color='blue')
            
            # Sparsity
            if 'sparsity' in semantic_data:
                steps = semantic_data['sparsity']['steps']
                values = semantic_data['sparsity']['values']
                axes[0, 1].plot(steps, values, label='Semantic Sparsity', color='blue')
        
        # Attribute factors
        if 'attribute' in factor_data:
            attribute_data = factor_data['attribute']
            
            # Mean activation
            if 'mean' in attribute_data:
                steps = attribute_data['mean']['steps']
                values = attribute_data['mean']['values']
                axes[1, 0].plot(steps, values, label='Attribute Mean', color='red')
            
            # Sparsity
            if 'sparsity' in attribute_data:
                steps = attribute_data['sparsity']['steps']
                values = attribute_data['sparsity']['values']
                axes[1, 1].plot(steps, values, label='Attribute Sparsity', color='red')
        
        # Set titles and labels
        axes[0, 0].set_title('Factor Mean Activation')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Factor Sparsity')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Sparsity')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Factor Mean Activation')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Mean Activation')
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Factor Sparsity')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Sparsity')
        axes[1, 1].legend()
        
        plt.suptitle('Factor Evolution During Training')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_factor_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_loss_evolution(self, save_prefix):
        """Plot loss component evolution"""
        if 'loss_evolution' not in self.analysis_results:
            return
        
        loss_data = self.analysis_results['loss_evolution']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        loss_components = ['total', 'reconstruction', 'factor_sparsity', 'orthogonality']
        colors = ['black', 'blue', 'green', 'red']
        
        for i, (loss_name, color) in enumerate(zip(loss_components, colors)):
            if loss_name in loss_data and i < len(axes):
                steps = loss_data[loss_name]['steps']
                values = loss_data[loss_name]['values']
                
                axes[i].plot(steps, values, color=color, linewidth=2)
                axes[i].set_title(f'{loss_name.title()} Loss')
                axes[i].set_xlabel('Training Step')
                axes[i].set_ylabel('Loss Value')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Loss Component Evolution')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_loss_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_stability_analysis(self, save_prefix):
        """Plot stability analysis"""
        if 'stability' not in self.analysis_results:
            return
        
        stability_data = self.analysis_results['stability']
        
        # Extract stability metrics
        metrics = []
        cv_values = []
        oscillation_rates = []
        
        for tag, data in stability_data.items():
            if 'factors/' in tag:
                metrics.append(tag.split('/')[-1])
                cv_values.append(data['coefficient_of_variation'])
                oscillation_rates.append(data['oscillation_rate'])
        
        if metrics:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Coefficient of variation
            bars1 = axes[0].bar(metrics, cv_values)
            axes[0].set_title('Training Stability (Coefficient of Variation)')
            axes[0].set_xlabel('Metric')
            axes[0].set_ylabel('CV (lower = more stable)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Color bars by stability
            for bar, cv in zip(bars1, cv_values):
                if cv < 0.1:
                    bar.set_color('green')
                elif cv < 0.2:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Oscillation rates
            bars2 = axes[1].bar(metrics, oscillation_rates)
            axes[1].set_title('Oscillation Rates')
            axes[1].set_xlabel('Metric')
            axes[1].set_ylabel('Oscillation Rate')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_stability.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def _create_summary_dashboard(self, save_prefix):
        """Create a summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a text summary of key insights
        if 'insights' in self.analysis_results:
            insights = self.analysis_results['insights']
            
            plt.figtext(0.1, 0.9, "TensorBoard Analysis Summary", fontsize=20, weight='bold')
            
            y_pos = 0.85
            for insight in insights[:10]:  # Show top 10 insights
                plt.figtext(0.1, y_pos, f"• {insight}", fontsize=12)
                y_pos -= 0.05
        
        # Add some key metrics as text
        if 'factor_trends' in self.analysis_results:
            trends = self.analysis_results['factor_trends']
            
            plt.figtext(0.6, 0.9, "Key Metrics", fontsize=16, weight='bold')
            
            y_pos = 0.85
            for category, metrics in trends.items():
                plt.figtext(0.6, y_pos, f"{category.title()} Factors:", fontsize=14, weight='bold')
                y_pos -= 0.03
                
                for metric_name, metric_data in metrics.items():
                    final_val = metric_data['final_value']
                    plt.figtext(0.62, y_pos, f"  {metric_name}: {final_val:.4f}", fontsize=12)
                    y_pos -= 0.025
                
                y_pos -= 0.02
        
        plt.axis('off')
        plt.savefig(f'{save_prefix}_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_analysis_report(self, filepath="tensorboard_analysis_report.txt"):
        """Save a comprehensive analysis report"""
        
        report_lines = []
        report_lines.append("TENSORBOARD ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Add insights
        if 'insights' in self.analysis_results:
            report_lines.append("KEY INSIGHTS:")
            report_lines.append("-" * 20)
            for insight in self.analysis_results['insights']:
                report_lines.append(f"  {insight}")
            report_lines.append("")
        
        # Add factor trends
        if 'factor_trends' in self.analysis_results:
            report_lines.append("FACTOR TRENDS:")
            report_lines.append("-" * 20)
            
            for category, metrics in self.analysis_results['factor_trends'].items():
                report_lines.append(f"{category.upper()} FACTORS:")
                for metric_name, metric_data in metrics.items():
                    report_lines.append(f"  {metric_name}:")
                    report_lines.append(f"    Final value: {metric_data['final_value']:.6f}")
                    report_lines.append(f"    Trend: {metric_data['trend']:.6f}")
                    report_lines.append(f"    Stability: {metric_data['stability']:.6f}")
                    report_lines.append(f"    Change: {metric_data['change']:.6f}")
                report_lines.append("")
        
        # Save report
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Analysis report saved to: {filepath}")


def run_tensorboard_analysis(log_dir="disentangled_vae_output/tensorboard_logs"):
    """Run complete TensorBoard analysis"""
    
    if not TENSORBOARD_AVAILABLE:
        print("❌ TensorBoard not available. Install with: pip install tensorboard")
        return
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    print("🔍 Starting TensorBoard Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = TensorBoardAnalyzer(log_dir)
    
    # Load logs
    if not analyzer.load_logs():
        return
    
    # Run analyses
    analyzer.analyze_factors()
    analyzer.analyze_loss_components()
    analyzer.analyze_training_stability()
    
    # Generate insights
    insights = analyzer.generate_insights()
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 20)
    for insight in insights:
        print(f"  {insight}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    analyzer.create_visualizations()
    
    # Save report
    analyzer.save_analysis_report()
    
    print("\n✅ TensorBoard analysis complete!")
    print("Generated files:")
    print("  - tensorboard_analysis_factor_evolution.png")
    print("  - tensorboard_analysis_loss_evolution.png")
    print("  - tensorboard_analysis_stability.png")
    print("  - tensorboard_analysis_summary.png")
    print("  - tensorboard_analysis_report.txt")


if __name__ == "__main__":
    run_tensorboard_analysis() 