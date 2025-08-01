import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Model evaluation utilities for AML fraud detection."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_binary_classification(self, y_true, y_pred, y_proba=None, model_name="model"):
        """Comprehensive evaluation for binary classification."""
        
        results = {
            'model_name': model_name,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_proba)
            results['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Calculate key AML metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results['aml_metrics'] = {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        self.results[model_name] = results
        return results
    
    def plot_roc_curves(self, models_data, figsize=(10, 8)):
        """Plot ROC curves for multiple models."""
        plt.figure(figsize=figsize)
        
        for model_name, (y_true, y_proba) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, models_data, figsize=(10, 8)):
        """Plot Precision-Recall curves for multiple models."""
        plt.figure(figsize=figsize)
        
        for model_name, (y_true, y_proba) in models_data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_confusion_matrices(self, models_results, figsize=(15, 5)):
        """Plot confusion matrices for multiple models."""
        n_models = len(models_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare all evaluated models."""
        if not self.results:
            print("No models evaluated yet.")
            return
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            aml_metrics = results['aml_metrics']
            
            row = {
                'Model': model_name,
                'Precision': aml_metrics['precision'],
                'Recall': aml_metrics['recall'],
                'F1-Score': results['classification_report']['1']['f1-score'],
                'False Positive Rate': aml_metrics['false_positive_rate'],
                'Detection Rate': aml_metrics['detection_rate']
            }
            
            if 'roc_auc' in results:
                row['ROC-AUC'] = results['roc_auc']
            if 'pr_auc' in results:
                row['PR-AUC'] = results['pr_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.round(4)
    
    def calculate_cost_analysis(self, y_true, y_pred, cost_fp=1, cost_fn=10):
        """Calculate cost analysis for AML detection."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        total_alerts = tp + fp
        alert_precision = tp / total_alerts if total_alerts > 0 else 0
        
        return {
            'total_cost': total_cost,
            'false_positive_cost': fp * cost_fp,
            'false_negative_cost': fn * cost_fn,
            'total_alerts': total_alerts,
            'alert_precision': alert_precision,
            'cost_per_transaction': total_cost / len(y_true)
        }
