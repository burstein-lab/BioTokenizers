import numpy as np
import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import evaluate
f1, precision, recall, accuracy, roc_auc, mc_roc_auc = evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall"), evaluate.load("accuracy"), evaluate.load("roc_auc"), evaluate.load("roc_auc", "multiclass")
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import label_binarize


COLORS = {'ProtBERTa_2': '#E6AA61', 'ProtBERTa_4': '#e67961', 'ProtBERTa_8': '#ce4763', 'ProtBERTa_12': '#a3386f', 'ProtBERTa_20': '#672a6b'}


def calc_mcc(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    if np.all(pred == pred[0]):  # if all prediction in the same
        return 0
    return matthews_corrcoef(labels, pred)


def calc_f1_vec(prec, recall):
    numerator = 2 * recall * prec
    denominator = recall + prec
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0))
    return f1_scores


def calc_metrics(labels, scores, n_labels=2, metric='weighted'):
    if n_labels == 2:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = calc_f1_vec(precision, recall)
        best_f1_ind = np.argmax(f1)
        scores = np.array(scores)
        mcc = [calc_mcc(labels, scores, t) for t in thresholds]
        best_mcc_ind = np.argmax(mcc)
        rocauc_score = roc_auc_score(labels, scores)
        prauc_score = auc(recall, precision)
    else:
        rocauc_score = roc_auc_score(labels, scores, multi_class='ovr')
        labels_binarized = label_binarize(labels, classes=range(n_labels))
        prauc_score = average_precision_score(labels_binarized, scores, average=metric)
        precision = recall = f1 = mcc = [np.nan]  # not relevant for multiclass
        best_f1_ind = best_mcc_ind = 0  # not relevant for multiclass
    return rocauc_score, prauc_score, precision[best_f1_ind], recall[best_f1_ind], f1[best_f1_ind], precision[best_mcc_ind], recall[best_mcc_ind], mcc[best_mcc_ind]


def return_computed_metrics(probs, labels): # TODO change name to return_all_eval_metrics_dict
    preds = torch.argmax(probs, dim=1).cpu().numpy()
    preds = np.asarray(preds).ravel()
    labels = np.array(labels).ravel()
    probs = probs.cpu().numpy()
    precision_macro = precision.compute(predictions=preds, references=labels, average="macro")["precision"]
    precision_weighted = precision.compute(predictions=preds, references=labels, average="weighted")["precision"]
    recall_macro = recall.compute(predictions=preds, references=labels, average="macro")["recall"]
    recall_weighted = recall.compute(predictions=preds, references=labels, average="weighted")["recall"]
    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    f1_weighted = f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = accuracy.compute(predictions=preds, references=labels)['accuracy']

    if probs.shape[1] == 2:  # binary classification
        auroc = roc_auc.compute(references=labels, prediction_scores=probs[:, 1])['roc_auc']
        aupr = average_precision_score(labels, probs[:, 1])
    else:
        auroc = mc_roc_auc.compute(references=labels, prediction_scores=probs, average='weighted', multi_class='ovr')['roc_auc']
        labels_binarized = label_binarize(labels, classes=range(probs.shape[1]))
        aupr = average_precision_score(labels_binarized, probs, average='weighted')
    return {
        "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
        "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted,
        'accuracy': acc, 'auroc': auroc, 'aupr': aupr
    }


def plot_all_metric_results(df, output_path, metric='weighted', colors=COLORS):
    # Cleaning the df columns
    df = df[[col for col in df.columns if col in ('Model', 'accuracy', 'AUPR', 'AUROC') or metric in col]]
    df = df.rename(columns={col: col.replace(f'_{metric}', '').title() for col in df.columns if col not in ('AUPR', 'AUROC')})
    df = df.rename(columns={col: col.upper() for col in df.columns if col in ('mae', 'mse', 'rmse', 'Aupr', 'Auroc')}).T

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot.bar(color=colors, alpha=0.9, xlabel='Metric', rot=0, ax=ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()