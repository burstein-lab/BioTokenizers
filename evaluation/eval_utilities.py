import numpy as np
import torch
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import evaluate
f1, precision, recall, accuracy, roc_auc, mc_roc_auc = evaluate.load("f1"), evaluate.load("precision"), evaluate.load("recall"), evaluate.load("accuracy"), evaluate.load("roc_auc"), evaluate.load("roc_auc", "multiclass")
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, matthews_corrcoef, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from model_training.train_roberta_regression import compute_metrics as compute_metrics_regression
from collections import defaultdict


COLORS = {'ProtBERTa_2': '#E6AA61', 'ProtBERTa_4': '#e67961', 'ProtBERTa_8': '#ce4763', 'ProtBERTa_12': '#a3386f', 'ProtBERTa_20': '#672a6b'}


def clean_col_name(col):
    clean_col = col.replace('_', ' ').title()
    clean_col = clean_col.upper() if clean_col.lower() in ('mae', 'mse', 'rmse', 'auroc', 'aupr') else clean_col.title()
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


def get_bootstrap_se(labels, probs, n_labels, n_bootstrap=100, is_regression=False, is_probs=False, metric='macro'):
    """
    Calculates confidence intervals for classification metrics using bootstrapping.
    """
    bootstrapped_stats = []
    labels = np.array(labels)
    se_results = {}

    if is_regression:
        orig_stats = compute_metrics_regression((probs.flatten(), labels.flatten()))
    else:
        orig_stats = get_results_dict(probs, labels, n_labels, is_probs=is_probs, metric=metric)
    se_results.update(orig_stats)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(len(labels)), random_state=i)
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]

        # Calculate metrics for this bootstrap sample
        if is_regression:
            stats = compute_metrics_regression((resampled_probs.flatten(), resampled_labels.flatten()))
        else:
            stats = get_results_dict(resampled_probs, resampled_labels, n_labels, is_probs=is_probs, metric=metric)
        bootstrapped_stats.append(stats)

    # Aggregate results into Confidence Intervals
    metrics = bootstrapped_stats[0].keys()

    for metric in metrics:
        values = [s[metric] for s in bootstrapped_stats]
        se = np.std(values) / np.sqrt(len(values))
        se_results[f"{metric}_se"] = se
        se_results[f"{metric}_bootstrap"] = np.array(values)  # Store all bootstrap values for potential further analysis

    return se_results


def calc_metrics(labels, scores, n_labels=2, metric='macro'):
    if n_labels == 2:
        scores = np.array(scores)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = calc_f1_vec(precision, recall)
        best_f1_ind = np.argmax(f1)
        mcc = [calc_mcc(labels, scores, t) for t in thresholds]
        best_mcc_ind = np.argmax(mcc)
        rocauc_score = roc_auc_score(labels, scores)
        prauc_score = auc(recall, precision)
    else:
        rocauc_score = roc_auc_score(labels, scores, multi_class='ovr', average=metric)
        labels_binarized = label_binarize(labels, classes=range(n_labels))
        prauc_score = average_precision_score(labels_binarized, scores, average=metric)
        precision = recall = f1 = mcc = [np.nan]  # not relevant for multiclass
        best_f1_ind = best_mcc_ind = 0  # not relevant for multiclass
    return rocauc_score, prauc_score, precision[best_f1_ind], recall[best_f1_ind], f1[best_f1_ind], precision[best_mcc_ind], recall[best_mcc_ind], mcc[best_mcc_ind]


def return_all_eval_metrics_dict(probs, labels):
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


def plot_all_metric_results_with_SE(df, output_path, metric='weighted', colors=COLORS):
    # Cleaning the df columns
    df = df[[col for col in df.columns if (col.replace('_se', '') in ('Model', 'accuracy', 'AUPR', 'AUROC') or metric in col) and '_bootstrap' not in col]]
    val_cols = [col for col in df.columns if col != 'Model' and '_se' not in col]
    se_cols = [f"{col}_se" for col in val_cols if f"{col}_se" in df.columns]

    # 2. Create the Values DataFrame (Transposed for plotting)
    plot_df = df[['Model'] + val_cols].copy()
    plot_df = plot_df.rename(columns={col: col.replace(f'_{metric}', '').title() for col in plot_df.columns if col not in ('AUPR', 'AUROC')})
    plot_df = plot_df.rename(columns={col: col.upper() for col in plot_df.columns if col.lower() in ('mae', 'mse', 'rmse', 'aupr', 'auroc')})
    plot_df = plot_df.set_index('Model').T

    # 3. Create the Error DataFrame (Must match shape of plot_df)
    err_df = df[['Model'] + se_cols].copy()
    err_df = err_df.rename(columns={col: col.replace('_se', '').replace(f'_{metric}', '').title() for col in err_df.columns if col not in ('AUPR', 'AUROC')})
    err_df = err_df.rename(columns={col: col.upper() for col in err_df.columns if col.lower() in ('mae', 'mse', 'rmse', 'aupr', 'auroc')})
    err_df = err_df.set_index('Model').T

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot.bar(ax=ax, yerr=err_df, color=colors, alpha=0.9, capsize=4, error_kw={'elinewidth': 1.5, 'ecolor': 'black'}, xlabel='Metric', rot=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_results_dict(scores, labels, n_labels=2, metric='macro', is_probs=False):
    probs = scores if is_probs else torch.nn.functional.softmax(scores.float(), dim=-1)
    res_dict = return_all_eval_metrics_dict(probs, labels)
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


def plot_binary_AUPR_AUROC_with_SE(labels, probs, model_label, color, axes, n_bootstrap=1000):
    labels = np.array(labels)
    bootstrapped_stats = defaultdict(list)

    o_fpr, o_tpr, _ = roc_curve(labels, probs[:, 1])
    o_precision, o_recall, _ = precision_recall_curve(labels, probs[:, 1])
    o_rocauc_score = roc_auc_score(labels, probs[:, 1])
    o_prauc_score = auc(o_recall, o_precision)

    fpr_mean = np.linspace(0, 1, len(o_fpr))
    interp_tprs = []

    rec_mean = np.linspace(0, 1, len(o_recall))
    interp_pres = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(len(labels)), random_state=i)
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]

        # Calculate metrics for this bootstrap sample
        fpr, tpr, _ = roc_curve(resampled_labels, resampled_probs[:, 1])
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interp_tprs.append(interp_tpr)

        precision, recall, _ = precision_recall_curve(resampled_labels, resampled_probs[:, 1])
        interp_pre = np.interp(rec_mean, recall, precision)
        interp_pre[0] = 1.0
        interp_pre[-1] = 0.0
        interp_pres.append(interp_pre)

        rocauc_score = roc_auc_score(resampled_labels, resampled_probs[:, 1])
        prauc_score = auc(recall, precision)
        bootstrapped_stats['rocauc_score'].append(rocauc_score)
        bootstrapped_stats['prauc_score'].append(prauc_score)

    tpr_se = np.std(interp_tprs, axis=0) / np.sqrt(n_bootstrap)
    tpr_upper = np.clip(o_tpr + tpr_se, 0, 1)
    tpr_lower = o_tpr - tpr_se
    roc_auc_se = np.std(bootstrapped_stats['rocauc_score']) / np.sqrt(n_bootstrap)

    axes[0].plot(o_fpr, o_tpr, lw=2, label=f'{model_label} (ROC AUC: {round(o_rocauc_score, 2)}±{round(roc_auc_se, 4)})', color=color)
    axes[0].fill_between(o_fpr, tpr_lower, tpr_upper, alpha=0.4, color=color)

    precision_se = np.std(interp_pres, axis=0) / np.sqrt(n_bootstrap)
    precision_upper = np.clip(o_precision + precision_se, 0, 1)
    precision_lower = o_precision - precision_se
    aupr_se = np.std(bootstrapped_stats['prauc_score']) / np.sqrt(n_bootstrap)
    axes[1].plot(o_recall, o_precision, lw=2, label=f'{model_label} (PR AUC: {round(o_prauc_score, 2)}±{round(aupr_se, 4)})', color=color)
    axes[1].fill_between(o_recall, precision_lower, precision_upper, alpha=0.3, color=color)


def finish_AUPR_AUROC_figure(f, axes, img_path):
    axes[0].set_xlabel('FPR', fontsize=12)
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('TPR', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[0].legend(loc="best", fontsize='medium')
    axes[1].legend(loc="best", fontsize='medium')
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set_title(f"ROC curve", fontsize=14)
    axes[1].set_title(f"Precision Recall curve", fontsize=14)
    f.tight_layout()
    plt.savefig(img_path)
    plt.close()
