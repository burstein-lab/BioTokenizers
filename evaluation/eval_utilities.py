import numpy as np
import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import evaluate
f1, precision, recall, accuracy, roc_auc, mc_roc_auc = evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall"), evaluate.load("accuracy"), evaluate.load("roc_auc"), evaluate.load("roc_auc", "multiclass")
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, matthews_corrcoef, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize


COLORS = {'ProtBERTa_2': '#E6AA61', 'ProtBERTa_4': '#e67961', 'ProtBERTa_8': '#ce4763', 'ProtBERTa_12': '#a3386f', 'ProtBERTa_20': '#672a6b'}


def clean_col_name(col):
    clean_col = col.replace('_', ' ').title()
    clean_col = clean_col.upper() if clean_col in ('mae', 'mse', 'rmse', 'AUROC', 'AUPR', 'auroc', 'aupr') else clean_col.title()
    return clean_col


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


def get_results_dict(scores, labels, n_labels=2, metric='weighted', is_probs=False):
    probs = scores if is_probs else torch.nn.functional.softmax(scores.float(), dim=-1)
    res_dict = return_computed_metrics(probs, labels)
    rocauc_score, prauc_score, chosen_precision, chosen_recall, chosen_f1, mcc_prec, mcc_rec, mcc = calc_metrics(labels, probs[:, 1] if n_labels == 2 else probs, n_labels, metric=metric)
    res_dict.update({'AUROC': rocauc_score, 'AUPR': prauc_score, 'Precision of Best F1': chosen_precision,
                'Recall of Best F1': chosen_recall, 'Best F1': chosen_f1, 'Precision of Best MCC': mcc_prec,
                'Recall of Best MCC': mcc_rec, 'Best MCC': mcc})
    return res_dict


def plot_binary_AUPR_AUROC(labels, probs, model_label, color, axes):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    rocauc_score = roc_auc_score(labels, probs[:, 1])
    prauc_score = auc(recall, precision)

    axes[0].plot(fpr, tpr, lw=2, label=f'{model_label} (ROC AUC: {round(rocauc_score, 2)})', color=color)
    axes[1].plot(recall, precision, lw=2, label=f'{model_label} (PR AUC: {round(prauc_score, 2)})', color=color)


def finish_AUPR_AUROC_figure(f, axes, img_path):
    axes[0].set_xlabel('FPR')
    axes[1].set_xlabel('Recall')
    axes[0].set_ylabel('TPR')
    axes[1].set_ylabel('Precision')
    axes[0].legend(loc="best", fontsize='small')
    axes[1].legend(loc="best", fontsize='small')
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set_title(f"ROC curve")
    axes[1].set_title(f"Precision Recall curve")
    f.tight_layout()
    plt.savefig(img_path)
    plt.close()
