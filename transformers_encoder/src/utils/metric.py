from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int], 
                         classes: List[str],
                         save_path: str = None) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()