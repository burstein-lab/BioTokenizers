import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from evaluation.eval_utilities import COLORS
from data_processing.get_encoded_dataset import get_tokenizer_dataset, map_amino_acids
from utilities import load_tokenizer


SIZES = [5, 10, 25, 50, 75, 100]


def plot_tokenization_time(df, output_file):
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
    ax.set_ylabel('Tokenization Time (seconds)', fontsize=12)
    ax.set_title('ProtBERTa Tokenization Time Comparison', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def create_tokenizer_training_time_plot(dataset, tokenizer_file, output_dir, repeats=10, size_factor=1000, col='prot', max_length=1026):
    output_file = os.path.join(output_dir, 'tokenization_time.svg')
    all_res = []

    test_dataset = get_tokenizer_dataset(dataset, mapping_code=20, col=col)['train']  # using 20 so that we measure mapping time in loop
    for aa_mapping in [2, 4, 8, 12, 20]:
        tokenizer_path = tokenizer_file + str(aa_mapping)
        tokenizer = load_tokenizer(tokenizer_path, max_length=max_length)
        for size in SIZES:
            full_size = size * size_factor
            size_to_save = size if size_factor == 1000 else full_size / 1000.0
            for repeat in range(repeats):
                curr_set = test_dataset.shuffle(seed=full_size+repeat).select(range(full_size))
                before = time.perf_counter()
                curr_set = curr_set.map(lambda x: map_amino_acids(x, aa_mapping, col=col))
                curr_set = curr_set.map(lambda e: tokenizer(e[col], truncation=True), batched=True, keep_in_memory=False, num_proc=1)
                after = time.perf_counter()
                elapsed_time = (after-before)
                all_res.append({'model': f'ProtBERTa_{aa_mapping}', 'size': size_to_save, 'runtime': elapsed_time})

    df = pd.DataFrame(all_res)
    summary = df.groupby(['model', 'size'])['runtime'].agg(['mean', 'std']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(repeats)
    summary['num'] = summary['model'].apply(lambda x: int(x.split('_')[-1]))
    summary = summary.sort_values(['num', 'size'], ascending=[False, True]).drop(columns=['num'])
    summary.to_pickle(output_file.replace('.svg', '.pkl').replace('.pdf', '.pkl').replace('.png', '.pkl'))
    plot_tokenization_time(summary, output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Analysis of the tokenizer training times of the reduced alphabets')
    parser.add_argument('--dataset', help='path to a directory with .csv files')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--out_dir', help='path to result directory')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--n_repeats', type=int, default=10, help='number of repeats to measure for each size')
    parser.add_argument('--size_factor', type=int, default=1000, help='Number to skip over when increasing dataset size. default: 1000')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    args = parser.parse_args()
    create_tokenizer_training_time_plot(args.dataset, args.tokenizer_prefix, args.out_dir, repeats=args.n_repeats, size_factor=args.size_factor, col=args.col_name, max_length=args.max_length)
