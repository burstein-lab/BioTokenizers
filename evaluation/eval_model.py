import pandas as pd
from torch.nn import Softmax
import matplotlib.pyplot as plt
import re
from utilities import load_model_and_tokenizer, clear_cache, run_model_in_batches
from data_processing.get_encoded_dataset import get_downstream_train_test
from model_training.finetune_model import prepare_pairwise_dataset
from model_training.train_roberta_regression import compute_metrics as compute_metrics_regression
from evaluation.eval_utilities import plot_all_metric_results, COLORS, get_results_dict, plot_binary_AUPR_AUROC, finish_AUPR_AUROC_figure
import torch
torch.backends.cudnn.benchmark = True


def create_regression_cleveland_plot(file_lst, title_lst, metric, output_file):
    """
    Create Cleveland dot plot for regression results across multiple tasks.
    :param file_lst: List of results pickle files, one per task - output of the eval_model_on_test function.
    :param title_lst: List of titles of the tasks corresponding to the files in file_lst.
    :param metric: The metric to plot. Should be one of the columns in the results dataframes.
    :param output_file: Path to save the output figure (should end with .pdf or .svg or .png)
    """
    metric_clean = metric.upper()
    fig, axes = plt.subplots(1, len(file_lst), figsize=(3 * len(file_lst), len(file_lst)), sharey=True)
    for ind in range(len(file_lst)):
        df = pd.read_pickle(file_lst[ind]).reset_index()
        x_data = df[metric]
        y_data = df['Model']
        colors = [COLORS[model] for model in y_data]
        ax = axes[ind]
        ax.hlines(y=y_data, xmin=min(x_data) * 0.9, xmax=x_data, color='lightgray', linestyle='--', alpha=0.7)
        ax.set_ylim(-0.5, 5 - 0.5)
        # Draw the dots
        ax.scatter(x_data, y_data, color=colors, s=120, zorder=3, linewidth=0.5)

        ax.set_title(title_lst[ind], fontsize=14, pad=12)
        ax.grid(axis='x', linestyle=':', alpha=0.5)

    # Labels and layout
    axes[0].set_ylabel('Model', fontsize=14)
    axes[1].set_xlabel(metric_clean, fontsize=14)
    plt.suptitle(f'Model Performance ({metric_clean}) Across Regression Tasks', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def eval_model_on_test(model_path, tokenizer_file, dataset, output_file, device_num, metric='weighted', n_labels=2, is_regression=False, is_pairwise=False, proc=10, max_len=1026, batch_size=64, col='prot'):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    all_res = []
    for aa_mapping in [2, 4, 8, 12, 20]:
        clear_cache()
        print(f'aa_mapping: {aa_mapping}')
        tokenizer_path = tokenizer_file + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path, device_num, max_length=max_len, model_type='regression' if is_regression else 'SeqClass', n_labels=n_labels)
        _, test_dataset, _ = get_downstream_train_test(dataset, mapping_code=20 if is_pairwise else aa_mapping, proc=proc)

        if is_pairwise:
            test_dataset = prepare_pairwise_dataset(test_dataset, aa_mapping, tokenizer, max_len, proc)

        model_res = run_model_in_batches(model, tokenizer, test_dataset, device, batch_size=batch_size, col=col)
        if is_regression:
            metric = ''
            res_dict = compute_metrics_regression((model_res, test_dataset.to_pandas()['label'].tolist()))
        else:
            res_dict = get_results_dict(model_res, test_dataset.to_pandas()['label'].tolist(), n_labels=n_labels, metric=metric)
            if n_labels == 2:
                test_dataset.set_format(type="numpy", columns=['label'])
                sm = Softmax(dim=1)
                probs = sm(model_res).numpy()
                plot_binary_AUPR_AUROC(test_dataset['label'], probs, f'ProtBERTa_{aa_mapping}', COLORS[f'ProtBERTa_{aa_mapping}'], axes)
        print(res_dict)
        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'
        all_res.append(res_dict)

    if not is_regression and n_labels == 2:
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
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels predicted by the model. default: 2')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default: 64')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    parser.add_argument('--is_pairwise', action='store_true', help='Choose this to evaluate a pairwise classification model. Default: False')
    parser.add_argument('--is_regression', action='store_true', help='Choose this to evaluate a regression model. Default: False')
    parser.add_argument('--file_lst', nargs='*', default=[], help='List of paths to regression model result pickle files to plot cleveland regression plot from existing results')
    parser.add_argument('--titles', nargs='*', default=[], help='List of tasks titles for the cleveland regression plot, should correspond to the order of file_lst')
    parser.set_defaults(is_pairwise=False)
    parser.set_defaults(is_regression=False)
    args = parser.parse_args()

    if args.file_lst and args.titles:  # plotting cleveland regression plot from existing files
        create_regression_cleveland_plot(args.file_lst, args.titles, args.metric, args.output_file)
    else:
        eval_model_on_test(args.model_path, args.tokenizer_prefix, args.dataset, args.output_file, args.device, metric=args.metric, n_labels=args.n_labels, is_regression=args.is_regression, is_pairwise=args.is_pairwise, proc=args.ncpu, max_len=args.max_length, batch_size=args.batch_size, col=args.col_name)
