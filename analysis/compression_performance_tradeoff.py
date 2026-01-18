import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import torch
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import precision_recall_curve
from utilities import load_model_and_tokenizer, clear_cache, run_model_in_batches, load_tokenizer
from data_processing.get_encoded_dataset import get_downstream_train_test
from model_training.finetune_model import prepare_pairwise_dataset
from evaluation.eval_utilities import get_results_dict, calc_f1_vec, clean_col_name
from model_training.train_roberta_regression import compute_metrics as compute_metrics_regression
from evaluation.ProtBERTa_zeroshot import EmbeddingClassifier, get_train_test_embeddings
from evaluation.protberta_pairwise_similarity import get_pairwise_similarity, get_all_metrics


TASKS_COLORS = ['#9e0142', '#f46d43', '#66c2a5', '#2b83ba', '#542788']
MULTICLASS_METRICS_MAPPING = {'Best F1': 'f1_weighted', 'Precision of Best F1': 'precision_weighted', 'Recall of Best F1': 'recall_weighted', 'Precision of Best MCC': 'precision_weighted', 'Recall of Best MCC': 'recall_weighted'}


def plot_multi_performance_compression_tradeoff(file_lst, tasks, output_path, metric, is_regression=False):
    plt.figure(figsize=(12, 5.5))
    min_value = None
    colors = TASKS_COLORS if len(file_lst) <= len(TASKS_COLORS) else sns.color_palette("hls", len(file_lst)).as_hex()
    lowest_y_values = [None, None, None, None, None]
    for index in range(len(file_lst)):
        df = pd.read_pickle(file_lst[index]).set_index('Model')
        rel_metric = metric if metric in df.columns else MULTICLASS_METRICS_MAPPING.get(metric, metric)
        min_value = df[rel_metric].min() if min_value is None else min(min_value, df[rel_metric].min())
        x_values = df['Compression_Ratio']
        if is_regression:  # For regression lower is better
            y_values = df.loc['ProtBERTa_20', rel_metric] / df[rel_metric]
        else:
            y_values = df[rel_metric] / df.loc['ProtBERTa_20', rel_metric]

        # Updating the lowest values for annotations
        for i, yv in enumerate(y_values):
            if lowest_y_values[i] is None or yv < lowest_y_values[i]:
                lowest_y_values[i] = yv

        plt.plot(x_values, y_values, marker='o', markersize=6, color=colors[index], label=tasks[index], linewidth=2, alpha=0.8)

        # Add annotations for the model to the Last task only to act as a header for each vertical column of points
        if index == len(file_lst) - 1:
            for i, (xi, yi) in enumerate(zip(x_values, lowest_y_values)):
                plt.annotate(
                    df.index[i],
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, -17),  # Position the text 12pts above the point
                    ha='center',
                    fontsize=12,
                    fontweight='normal',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none')
                )

    # 5. Formatting and Styling
    plt.xlabel('Compression Ratio', fontsize=15)
    plt.ylabel(f'Relative Performance ({clean_col_name(metric)})', fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Performance vs. Compression Ratio', fontsize=17, pad=15)

    y_limit_bottom = min_value - 0.02
    plt.ylim(bottom=y_limit_bottom)
    plt.legend(title='Tasks', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=14, title_fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_compression_performance_tradeoff(df, output_dir, task):
    for col in df.columns:
        if col in ['Model', 'Compression_Ratio', 'success']:
            continue
        clean_col = clean_col_name(col)
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Compression_Ratio'], df[col], s=100, c='#0c2c84', alpha=0.6)

        for i, txt in enumerate(df['Model']):
            plt.annotate(txt, (df['Compression_Ratio'][i], df[col][i]), xytext=(5, 5), textcoords='offset points')

        plt.axhline(y=df[col].iloc[0], color='gray', linestyle='--', alpha=0.3, label=f'Baseline {clean_col}')
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3, label='Baseline Size')

        plt.title('Compression Efficiency vs. Task Performance Trade-off')
        plt.xlabel('Compression Ratio')
        plt.ylabel(clean_col)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        sorted_indices = np.argsort(df['Compression_Ratio'])
        pareto_x = [df['Compression_Ratio'][i] for i in sorted_indices]
        pareto_y = [df[col][i] for i in sorted_indices]
        plt.plot(pareto_x, pareto_y, color='#7fcdbb', alpha=0.3, linestyle='-', label='Trend Line')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task}_compression_vs_performance_{col}.svg'), dpi=300, bbox_inches="tight")
        plt.close()


def get_model_success(n_labels, probs, labels, res_dict):
    if n_labels == 2:
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1 = calc_f1_vec(precision, recall)
        thresh = thresholds[np.argmax(f1)]
        preds = (probs >= thresh).astype(int)
        res = preds == np.array(labels)
        res_dict['success'] = res
    else:
        preds = torch.argmax(torch.Tensor(probs), dim=1).cpu().numpy()
        res = preds == np.array(labels)
        res_dict['success'] = res


def run_mcnemar_test(df):
    baseline = df[df['Model'] == 'ProtBERTa_20'].iloc[0]['success']
    for aa_mapping in [2, 4, 8, 12]:
        alternative = df[df['Model'] == f'ProtBERTa_{aa_mapping}'].iloc[0]['success']
        both_correct = np.sum((baseline == True) & (alternative == True))
        base_only = np.sum((baseline == True) & (alternative == False))
        comp_only = np.sum((baseline == False) & (alternative == True))
        both_wrong = np.sum((baseline == False) & (alternative == False))
        contingency_table = [[both_correct, base_only], [comp_only, both_wrong]]
        result = mcnemar(contingency_table, exact=True)

        print(f"Comparison: Baseline vs  ProtBERTa_{aa_mapping}")
        print(f"p-value: {result.pvalue:.5f}")

        if result.pvalue < 0.05:
            print(">> RESULT: The drop in performance is Statistically Significant.")
        else:
            print(">> RESULT: The difference is NOT statistically significant (Effective Tie).")


def finetuned_models_compression_tradeoff(model_path, tokenizer_file, dataset, task, output_dir, device_num, n_labels=2, is_pairwise=False, is_regression=False, col='prot', max_length=1026, proc=10, batch_size=64):
    all_res = []
    baseline_len = 0
    for aa_mapping in [20, 12, 8, 4, 2]:
        clear_cache()
        print(f'aa_mapping: {aa_mapping}')
        tokenizer_path = tokenizer_file + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path, device_num, max_length=max_length, model_type='regression' if is_regression else'SeqClass', n_labels=n_labels)
        _, test_dataset, _ = get_downstream_train_test(dataset, mapping_code=20 if is_pairwise else aa_mapping, proc=proc)

        if is_pairwise:
            test_dataset = prepare_pairwise_dataset(test_dataset, aa_mapping, tokenizer, max_length, proc)
        else:
            test_dataset = test_dataset.map(lambda e: tokenizer(e[col], truncation=True), batched=True, keep_in_memory=False, num_proc=proc)

        avg_len = np.mean([len(s['input_ids']) for s in test_dataset])
        if aa_mapping == 20:
            baseline_len = avg_len

        compression = baseline_len / avg_len
        model_res = run_model_in_batches(model, tokenizer, test_dataset, device, batch_size=batch_size)
        labels = test_dataset.to_pandas()['label'].tolist()
        if is_regression:
            res_dict = compute_metrics_regression((model_res, labels))
        else:
            res_dict = get_results_dict(model_res, labels, n_labels=n_labels)
            probs = torch.nn.functional.softmax(model_res.float(), dim=-1).cpu().numpy()
            probs = probs[:, 1] if n_labels == 2 else probs
            print(res_dict)
            get_model_success(n_labels, probs, labels, res_dict)

        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'
        res_dict['Compression_Ratio'] = compression
        all_res.append(res_dict)

    df = pd.DataFrame(all_res).dropna(how='all', axis=1)  # remove columns with all NaN values
    plot_compression_performance_tradeoff(df, output_dir, task)
    if not is_regression:
        run_mcnemar_test(df)
        df = df.drop(columns=['success'])

    df.to_pickle(os.path.join(output_dir, f'{task}_compression_performance_tradeoff.pkl'))


def run_zeroshot_classification_compression_tradeoff(model_path, tokenizer_file, dataset, emb_dir, task, output_dir, k=5, distance_metric='cosine', proc=10, col='prot', max_length=1026, device=-1, batch_size=32):
    all_res = []
    baseline_len = 0
    for aa_mapping in [20, 12, 8, 4, 2]:
        clear_cache()
        print(f'aa_mapping: {aa_mapping}')
        tokenizer_path = tokenizer_file + str(aa_mapping)
        tokenizer = load_tokenizer(tokenizer_path, max_length=max_length)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        train_embeddings, train_labels, test_embeddings, test_labels = get_train_test_embeddings(dataset, emb_dir, tokenizer_path, model_path, task, aa_mapping, proc=proc, col=col, max_length=max_length, device=device, batch_size=batch_size)
        _, test_dataset, _ = get_downstream_train_test(dataset, mapping_code=aa_mapping, proc=proc)

        test_dataset = test_dataset.map(lambda e: tokenizer(e['prot'], truncation=True), batched=True, keep_in_memory=False, num_proc=proc)
        avg_len = np.mean([len(s['input_ids']) for s in test_dataset])
        if aa_mapping == 20:
            baseline_len = avg_len

        compression = baseline_len / avg_len
        classifier = EmbeddingClassifier(k=k, distance_metric=distance_metric, chunk_size=batch_size)
        classifier.fit(train_embeddings, train_labels)
        n_labels = len(classifier.classes)

        probs = classifier.predict_proba(test_embeddings).cpu()
        res_dict = get_results_dict(probs, test_labels, n_labels=n_labels, is_probs=True)
        probs = probs[:, 1] if n_labels == 2 else probs
        print(res_dict)
        get_model_success(n_labels, probs.numpy(), test_labels, res_dict)
        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'
        res_dict['Compression_Ratio'] = compression
        all_res.append(res_dict)

    df = pd.DataFrame(all_res).dropna(how='all', axis=1)  # remove columns with all NaN values
    plot_compression_performance_tradeoff(df, output_dir, task)
    run_mcnemar_test(df)
    df.drop('success', axis=1).to_pickle(os.path.join(output_dir, f'{task}_compression_zeroshot_performance_tradeoff.pkl'))


def run_pairwise_zeroshot_compression_tradeoff(model_path, tokenizer_prefix, dataset, output_dir, proc=10, device=-1, batch_size=16, max_length=1026):
    all_res = []
    baseline_len = 0
    for aa_mapping in [20, 12, 8, 4, 2]:
        clear_cache()
        print(f'aa_mapping: {aa_mapping}')
        tokenizer_path = tokenizer_prefix + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        df = get_pairwise_similarity(dataset, model_path, tokenizer_path, aa_mapping, proc=proc, device_num=device, batch_size=batch_size, max_len=max_length)
        res_dict = get_all_metrics(df)  # Overall Metrics
        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'

        len_1 = df['attention_mask_1'].apply(lambda x: x.sum())
        len_2 = df['attention_mask_2'].apply(lambda x: x.sum())
        avg_len = (len_1.sum() + len_2.sum()) / len(len_1)
        if aa_mapping == 20:
            baseline_len = avg_len
        compression = baseline_len / avg_len

        probs = df['similarity'].values
        get_model_success(2, probs, df['label'].tolist(), res_dict)
        print(res_dict)
        res_dict['Model'] = f'ProtBERTa_{aa_mapping}'
        res_dict['Compression_Ratio'] = compression
        all_res.append(res_dict)

    df = pd.DataFrame(all_res).dropna(how='all', axis=1)  # remove columns with all NaN values
    plot_compression_performance_tradeoff(df, output_dir, 'homology')
    run_mcnemar_test(df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Analysis of the tradeoff between compression and performance of the finetuned ProtBERTa models')
    parser.add_argument('--model_path', help='path to a the finetuned model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--dataset', help='path to a directory with .csv files')
    parser.add_argument('--out_dir', help='path to directory to save the results')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels predicted by the model. default: 2')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default: 64')
    parser.add_argument('--zeroshot', action='store_true', help='Choose this to evaluate Zeroshot classification. Default: False')
    parser.add_argument('--is_pairwise', action='store_true', help='Choose this to evaluate a pairwise classification model. If zeroshot is true, zeroshot pairwise classification is evaluated. Default: False')
    parser.add_argument('--is_regression', action='store_true', help='Choose this to evaluate a regression model. Default: False')
    parser.add_argument('--file_lst', nargs='*', default=[], help='List of paths to runtime result pickle files to plot subplots from existing results')
    parser.add_argument('--tasks', nargs='+', help='List of tasks names to use. If file_lst, it should match the order of file_lst. A single task name otherwise.')
    parser.add_argument('--metric', type=str, help='Metric to plot when plotting multiple tasks. Default: Best F1', default='Best F1')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN for zeroshot classification (default: 5)')
    parser.add_argument('--distance_metric', help='Distance metric to use (cosine or euclidean) for zeroshot classification. default: cosine', default='cosine')
    parser.add_argument('--emb_dir', help='path to a directory to save pre-trained embeddings files for zeroshot classification')
    parser.set_defaults(zeroshot=False)
    parser.set_defaults(is_pairwise=False)
    parser.set_defaults(is_regression=False)
    args = parser.parse_args()

    if args.file_lst:
        plot_multi_performance_compression_tradeoff(args.file_lst, args.tasks, os.path.join(args.out_dir, f'{"regression_" if args.is_regression else ""}multi_task_{args.metric}_tradeoff.svg'), args.metric, is_regression=args.is_regression)
    elif args.zeroshot:
        if args.is_pairwise:
            run_pairwise_zeroshot_compression_tradeoff(args.model_path, args.tokenizer_prefix, args.dataset, args.out_dir, args.ncpu, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
        else:
            run_zeroshot_classification_compression_tradeoff(args.model_path, args.tokenizer_prefix, args.dataset, args.emb_dir,
                                                                 args.tasks[0], args.out_dir, k=args.k,
                                                                 distance_metric=args.distance_metric, proc=args.ncpu, col=args.col_name,
                                                                 max_length=args.max_length, device=args.device, batch_size=args.batch_size)
    else:
        finetuned_models_compression_tradeoff(args.model_path, args.tokenizer_prefix, args.dataset, args.tasks[0],
                                              args.out_dir, args.device, n_labels=args.n_labels, col=args.col_name,
                                              is_pairwise=args.is_pairwise, is_regression=args.is_regression,
                                              max_length=args.max_length, proc=args.ncpu, batch_size=args.batch_size)






