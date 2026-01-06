import pandas as pd
from torch.nn import Softmax
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import re
from utilities import load_model_and_tokenizer, clear_cache, run_model_in_batches
from data_processing.get_encoded_dataset import get_downstream_train_test
from model_training.finetune_model import prepare_pairwise_dataset
from model_training.train_roberta_regression import compute_metrics as compute_metrics_regression
from eval_utilities import plot_all_metric_results, COLORS, return_computed_metrics,  calc_metrics
import torch
torch.backends.cudnn.benchmark = True


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


def eval_model_on_test(model_path, tokenizer_file, dataset, output_file, device_num, metric='weighted', n_labels=2, is_regression=False, is_pairwise=False):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    all_res = []
    for aa_mapping in [2, 4, 8, 12, 20]:
        clear_cache()
        print(f'aa_mapping: {aa_mapping}')
        tokenizer_path = tokenizer_file + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path, device_num, max_length=1026, model_type='regression' if is_regression else 'SeqClass', n_labels=n_labels)
        _, test_dataset, _ = get_downstream_train_test(dataset, mapping_code=20 if is_pairwise else aa_mapping, train_file_num=0, proc=10)

        if is_pairwise:
            test_dataset = prepare_pairwise_dataset(test_dataset, aa_mapping, tokenizer, 1026, 10)

        model_res = run_model_in_batches(model, tokenizer, test_dataset, device, batch_size=64)
        if is_regression:
            metric = ''
            res_dict = compute_metrics_regression((model_res, test_dataset.to_pandas()['label'].tolist()))
        else:
            res_dict = get_results_dict(model_res, test_dataset.to_pandas()['label'].tolist(), n_labels=n_labels, metric=metric)
            test_dataset.set_format(type="numpy", columns=['label'])
            sm = Softmax(dim=1)
            probs = sm(model_res).numpy()
            plot_binary_AUPR_AUROC(test_dataset['label'], probs, f'ProtBERTa_{aa_mapping}', COLORS[f'ProtBERTa_{aa_mapping}'], axes)
        print(res_dict)
        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'
        all_res.append(res_dict)

    if not is_regression:
        auroc_aupr_file = output_file.replace('.pdf', '_AUPR_AUROC.pdf').replace('.svg', '_AUPR_AUROC.svg').replace('.png', '_AUPR_AUROC.png')
        finish_AUPR_AUROC_figure(f, axes, auroc_aupr_file)

    df = pd.DataFrame(all_res).set_index('Model').dropna(how='all', axis=1)  # remove columns with all NaN values
    df.to_pickle(output_file.replace('.pdf', '.pkl').replace('.svg', '.pkl').replace('.png', '.pkl'))
    plot_all_metric_results(df, output_file, metric)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Evaluating the ProtBERTa models')
    parser.add_argument('--model_path', help='path to a the finetuned model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--dataset', help='path to a directory with .csv train and test dataset')
    parser.add_argument('--output_file', type=str, help='Path to output file, should end with .pdf or .svg or .png')
    parser.add_argument('--metric', type=str, help='Metric to calculate (micro, macro, weighted). default: weighted', default='weighted')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels predicted by the model. default: 2')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    parser.add_argument('--is_pairwise', action='store_true', help='Choose this to evaluate a pairwise classification model. Default: False')
    parser.add_argument('--is_regression', action='store_true', help='Choose this to evaluate a regression model. Default: False')
    parser.set_defaults(is_pairwise=False)
    parser.set_defaults(is_regression=False)
    args = parser.parse_args()

    eval_model_on_test(args.model_path, args.tokenizer_prefix, args.dataset, args.output_file, args.device, metric=args.metric, n_labels=args.n_labels, is_regression=args.is_regression, is_pairwise=args.is_pairwise)
