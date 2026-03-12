import re
import math
import torch
torch.backends.cudnn.benchmark = True
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from utilities import load_model_and_tokenizer, run_model_in_batches
from evaluation.eval_utilities import COLORS
from data_processing.get_encoded_dataset import get_downstream_train_test
from model_training.finetune_model import prepare_pairwise_dataset

SIZES = [1, 5, 10, 25, 50]


def plot_ProtBerta_runtime_subplots(file_lst, tasks, output_file):
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))

    for ind in range(len(file_lst)):
        x_ind = math.floor(int(ind / 4))
        y_ind = ind % 4
        summary = pd.read_pickle(file_lst[ind])
        for model in summary['model'].unique():
            model_data = summary[summary['model'] == model]
            ax[x_ind][y_ind].errorbar(
                model_data['size'],
                model_data['mean'],
                yerr=model_data['se'],
                fmt='o-',
                capsize=4,
                label=model,
                color=COLORS[model],
                alpha=0.9
            )

        # Add grid
        ax[x_ind][y_ind].grid(True, which='both', linestyle='--', linewidth=0.5)
        ax[x_ind][y_ind].set_title(f"{tasks[ind]}", fontsize=14)

    handles, labels = ax[x_ind][y_ind].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.15, 0.1), ncol=len(labels), fontsize=14,frameon=False)

    fig.text(0.5, 0.02, 'Number of Sequences (in Thousands)', ha='center', fontsize=18)
    fig.text(0.0000000001, 0.5, 'Runtime (seconds)', va='center', rotation='vertical', fontsize=18)
    plt.tight_layout(rect=[0.015, 0.07, 1, 0.95])  # [0, 0.09, 1, 0.95]
    plt.suptitle('ProtBERTa Inference Time Comparison', fontsize=16, y=1.01)
    plt.subplots_adjust(top=0.91)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ProtBerta_runtime(df, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax.errorbar(
            model_data['size'],
            model_data['mean'],
            yerr=model_data['se'],
            fmt='o-',
            capsize=4,
            label=model,
            color=COLORS[model],
            alpha=0.9
        )

    ax.set_xlabel(f'Number of Sequences (in Thousands)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('ProtBERTa Runtime Comparison', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def measure_model_runtime_on_data(eval_dataloader, model, start, end, device):
    total_time = 0.0
    for batch in eval_dataloader:
        input_ids, attention_mask = torch.as_tensor(batch['input_ids']).detach().to(device), torch.as_tensor(batch['attention_mask']).detach().to(device)
        with torch.no_grad():
            start.record()
            _ = model(input_ids, attention_mask=attention_mask)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end) / 1000.0  # convert to seconds
    return total_time


def create_ProtBerta_runtime_plot(model_path, tokenizer_file, dataset, output_file, repeats=10, device_num=-1, size_factor=1000, n_labels=2, warmup=100, is_pairwise=False, is_regression=False, col='prot', proc=10, max_len=1026, batch_size=128):
    all_res = []

    torch.backends.cudnn.benchmark = True
    for aa_mapping in [2, 4, 8, 12, 20]:
        tokenizer_path = tokenizer_file + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path, device_num,max_length=max_len, model_type='regression' if is_regression else 'SeqClass', n_labels=n_labels)
        data_collator = DataCollatorWithPadding(tokenizer)
        _, test_dataset, _ = get_downstream_train_test(dataset, mapping_code=20 if is_pairwise else aa_mapping, proc=proc)
        size_factor = size_factor if size_factor > 0 else math.floor(len(test_dataset) / SIZES[-1])  # if zero, adjust size factor based on dataset size

        if is_pairwise:
            test_dataset = prepare_pairwise_dataset(test_dataset, aa_mapping, tokenizer, max_len, proc)

        else:
            test_dataset = test_dataset.map(lambda e: tokenizer(e[col], truncation=True), batched=True, keep_in_memory=False, num_proc=proc)

        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        columns_to_remove = [col for col in test_dataset.column_names if col not in ['label', 'input_ids', 'attention_mask']]
        test_dataset = test_dataset.remove_columns(columns_to_remove)

        # Warmup
        curr_size = warmup*batch_size
        if curr_size > len(test_dataset):
            nrepeats = int(curr_size / len(test_dataset)) + 1
            for repeat in range(nrepeats):
                warmup_set = test_dataset.shuffle(seed=repeat)
                run_model_in_batches(model, tokenizer, warmup_set, device, batch_size=batch_size)
        else:
            warmup_set = test_dataset.shuffle(seed=42).select(range(curr_size))
            run_model_in_batches(model, tokenizer, warmup_set, device, batch_size=batch_size)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for size in SIZES:
            full_size = size * size_factor
            # We show the number in thousands
            size_to_save = size if size_factor == 1000 else full_size / 1000.0
            if full_size > len(test_dataset):
                break
            for repeat in range(repeats):
                curr_set = test_dataset.shuffle(seed=full_size+repeat).select(range(full_size))
                eval_dataloader = DataLoader(curr_set, batch_size=batch_size, collate_fn=data_collator, shuffle=False, pin_memory=True)
                elapsed_time = measure_model_runtime_on_data(eval_dataloader, model, start, end, device)
                all_res.append({'model': f'ProtBERTa_{aa_mapping}', 'size': size_to_save, 'runtime': elapsed_time})

    df = pd.DataFrame(all_res)
    summary = df.groupby(['model', 'size'])['runtime'].agg(['mean', 'std']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(repeats)
    summary['num'] = summary['model'].apply(lambda x: int(x.split('_')[-1]))
    summary = summary.sort_values(['num', 'size'], ascending=[False, True]).drop(columns=['num'])
    summary.to_pickle(output_file.replace('.svg', '.pkl').replace('.pdf', '.pkl').replace('.png', '.pkl'))
    plot_ProtBerta_runtime(summary, output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Analysis of the inference times of the finetuned ProtBERTa models')
    parser.add_argument('--model_path', help='path to a the finetuned model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--dataset', help='path to a directory with .csv files')
    parser.add_argument('--out_file', help='path to result figure (svg, pdf or png format)')
    parser.add_argument('--n_repeats', type=int, default=10, help='number of repeats to measure for each size')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    parser.add_argument('--size_factor', type=int, default=1000, help='Number to skip over when increasing dataset size. If zero, a factor is chosen based on the dataset size. default: 1000')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default: 64')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of labels predicted by the model. default: 2')
    parser.add_argument('--warm_up', type=int, default=100, help='number of warmup batches before starting timing. default: 100')
    parser.add_argument('--is_pairwise', action='store_true', help='Choose this to evaluate a pairwise classification model. Default: False')
    parser.add_argument('--is_regression', action='store_true', help='Choose this to evaluate a regression model. Default: False')
    parser.add_argument('--file_lst', nargs='*', default=[], help='List of paths to runtime result pickle files to plot subplots from existing results')
    parser.add_argument('--tasks', nargs='*', default=[], help='List of tasks names to use as subplot titles when plotting from existing result files. Should match the order of file_lst')
    parser.set_defaults(is_pairwise=False)
    parser.set_defaults(is_regression=False)
    args = parser.parse_args()

    if args.file_lst and args.tasks:  # plotting multiple subplots from existing files
        plot_ProtBerta_runtime_subplots(args.file_lst, args.tasks, args.out_file)
    else:
        create_ProtBerta_runtime_plot(args.model_path, args.tokenizer_prefix, args.dataset, args.out_file, repeats=args.n_repeats, device_num=args.device, size_factor=args.size_factor, n_labels=args.n_labels, warmup=args.warm_up, is_pairwise=args.is_pairwise, is_regression=args.is_regression, max_len=args.max_length, proc=args.ncpu, batch_size=args.batch_size)
