import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding # Good practice to explicitly use data collator
)
import numpy as np
import evaluate # Import the evaluate library
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from typing import Dict, Optional, List

class BaseEmailClassifier:
    def __init__(
        self,
        token: Optional[str] = None,  # Hugging Face token for login
    ):
        if token:
            login(token=token)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_and_plot_confusion_matrix(
        self, preds: torch.Tensor, labels: torch.Tensor, class_names: List[str], filename: str
    ) -> np.ndarray:
        """
        Computes and plots the confusion matrix.

        Args:
            preds (torch.Tensor): Predicted labels.
            labels (torch.Tensor): True labels.
            class_names (List[str]): List of class names.
            filename (str): Filename to save the plot.

        Returns:
            np.ndarray: The confusion matrix.
        """

        cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return cm

    def _print_metrics_from_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> None:
        """
        Prints evaluation metrics (accuracy, precision, recall, F1) from the confusion matrix.

        Args:
            cm (np.ndarray): The confusion matrix.
            class_names (List[str]): List of class names.
        """

        num_classes = len(class_names)
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nPer-class metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 45)
        for i in range(num_classes):
            tp = cm[i, i]
            all_pred_as_i = np.sum(cm[:, i])
            all_actual_i = np.sum(cm[i, :])
            precision = tp / all_pred_as_i if all_pred_as_i > 0 else 0
            recall = tp / all_actual_i if all_actual_i > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{class_names[i]:<15} {precision:.4f} {recall:.4f} {f1:.4f}")

    def _plot_roc_curve(
        self, labels_bin: np.ndarray, probs: np.ndarray, class_names: List[str], filename: str
    ) -> None:
        """
        Plots the ROC curve.

        Args:
            labels_bin (np.ndarray): Binarized labels.
            probs (np.ndarray): Predicted probabilities.
            class_names (List[str]): List of class names.
            filename (str): Filename to save the plot.
        """

        plt.figure(figsize=(8, 6))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print("\nPer-class ROC AUC:")
        for i in range(num_classes):
            print(f"{class_names[i]:<15}: {roc_auc[i]:.4f}")
        print(f"\nMacro-average ROC AUC: {np.nanmean(list(roc_auc.values())):.4f}")
    