import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import auc, precision_recall_curve, roc_curve, classification_report, roc_auc_score, average_precision_score, f1_score

def save_curve_data(y_true, y_score, method, curve_type) -> None:
    """
    Save curve data to disk for later visualization. The supported curve types are ROC and PRC.

    The curves are stored in a plots folder that is created if it does not exist.
    """

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    if curve_type == 'roc':
        fpr, tpr, _ = roc_curve(y_true, y_score)
        np.save(plots_dir / f"{method}_{curve_type}_fpr.npy", fpr)
        np.save(plots_dir / f"{method}_{curve_type}_tpr.npy", tpr)
    elif curve_type == 'prc':
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        np.save(plots_dir / f"{method}_{curve_type}_precision.npy", precision)
        np.save(plots_dir / f"{method}_{curve_type}_recall.npy", recall)

def plot_curves(methods, curve_type, title=None) -> None:
    """
    Plots the curves that are stored in the `plots` folder.

    This is just a helper function to plot ROC and PR curves for multiple methods at once to compare them.
    """
    plt.figure(figsize=(10, 8))

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    for method in methods:
        if curve_type == 'roc':
            fpr = np.load(plots_dir / f"{method}_{curve_type}_fpr.npy", allow_pickle=True)
            tpr = np.load(plots_dir / f"{method}_{curve_type}_tpr.npy", allow_pickle=True)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.3f})')

            plt.xlabel('False Positive Rate', fontsize=15)
            plt.ylabel('True Positive Rate', fontsize=15)
            plt.title('ROC Curve' if title is None else title)
            plt.legend(loc='lower right')
        elif curve_type == 'prc':
            precision = np.load(plots_dir / f"{method}_{curve_type}_precision.npy", allow_pickle=True)
            recall = np.load(plots_dir / f"{method}_{curve_type}_recall.npy", allow_pickle=True)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{method} (AUC = {pr_auc:.3f})')

            plt.xlabel('Recall', fontsize=15)
            plt.ylabel('Precision', fontsize=15)
            plt.title('Precision-Recall Curve' if title is None else title)
            plt.legend(loc='lower left')
    
    
    plt.savefig(plots_dir / f"{curve_type}_comparison.png")
    plt.close()

def compute_metrics(y_true, y_pred, y_binarized, probabilities, method, save_curve=False) -> dict:
    """
    Computes many classification metrics for the given task. This function returns a dictionary containing:

    This function expects:

    y_true: true binary labels
    y_pred: predicted binary labels
    y_binarized: binarized version of the true labels for multi-class classification
    probabilities: logits
    method: the name of the method used 

    And returns a dictionary of:

    - classification report of per-class accuracies, precision, recall, f1-score, and support
    - micro averaged auroc
    - micro averaged auprc
    - macro averaged auprc
    - micro averaged f1-score
    - macro averaged f1-score

    It will also save ROC and PRC curve values for later visualization in a `plots` folder if the `save_curve` parameter is set.
    """

    # Calculate micro F1 score and overall accuracy
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Compute micro-averaged AUROC and AUPR
    auroc = roc_auc_score(y_binarized, probabilities, average='micro')
    micro_aupr = average_precision_score(y_binarized, probabilities, average='micro')
    macro_aupr = average_precision_score(y_binarized, probabilities, average='macro')

    # Save the roc and prc curves for viz later
    if save_curve:
        save_curve_data(y_binarized.ravel(), probabilities.ravel(), method, 'roc')
        save_curve_data(y_binarized.ravel(), probabilities.ravel(), method, 'prc')

    return {
        'classification_report': df_report,
        'micro_auroc': auroc,
        'micro_aupr': micro_aupr,
        'macro_aupr': macro_aupr,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

