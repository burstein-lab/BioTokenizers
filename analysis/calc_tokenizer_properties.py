import os
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast
from data_processing.get_encoded_dataset import get_tokenizer_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


COLORS = {2: '#E6AA61', 4: '#e67961', 8: '#ce4763', 12: '#a3386f', None: '#85316d', 20: '#672a6b'}
COLORS_FULL = {'ProtBERTa_2': '#E6AA61', 'ProtBERTa_4': '#e67961', 'ProtBERTa_8': '#ce4763', 'ProtBERTa_12': '#a3386f', None: '#85316d', 'ProtBERTa_20': '#672a6b'}
CODING_LABELS = [f"ProtBERTa_{c}" for c in [2, 4, 8, 12, 20]]


def plot_sentence_len_dist(data, outdir, quantile=0.99):
    for aa_mapping in [2, 4, 8, 12, 20]:
        lens = data[f'ProtBERTa_{aa_mapping}']['seq_lens']
        size = np.quantile(lens, quantile)
        sns.kdeplot(data[f'ProtBERTa_{aa_mapping}']['seq_lens'], color=COLORS[aa_mapping], linewidth=2, fill=True, bw_adjust=2, clip=(1, size))

    plt.title('Distribution of Sentence Length per Amino Acid Mapping')
    plt.xlabel('Sentence Length')
    plt.ylabel('Density')
    plt.legend(labels=CODING_LABELS, prop={'size': 12})
    plt.savefig(os.path.join(outdir, f'sentence_{quantile}_quantile_len_dist.svg'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_avg_token_len_violin(data, outdir):
    d = {f'ProtBERTa_{aa_mapping}': data[f'ProtBERTa_{aa_mapping}']['avg_token_len_per_sent'] for aa_mapping in [2, 4, 8, 12, 20]}
    sns.violinplot(d, palette=COLORS_FULL, fill=True)

    plt.xlabel('Model', fontsize=12)
    plt.xticks(fontsize=9)
    plt.ylabel('Average Token Length', fontsize=12)
    plt.title('Average Token Length per Protein', fontsize=14)
    plt.savefig(os.path.join(outdir, 'avg_token_len_violin_dist.svg'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentence_len_violin(data, outdir, quantile=0.99):
    d = {f'ProtBERTa_{aa_mapping}': np.array(data[f'ProtBERTa_{aa_mapping}']['seq_lens']) for aa_mapping in [2, 4, 8, 12, 20]}
    d = {k: v[v <= np.quantile(v, quantile)] for k, v in d.items()}

    sns.violinplot(d, palette=COLORS_FULL, fill=True)

    plt.xlabel('Model', fontsize=12)
    plt.xticks(fontsize=9)
    plt.ylabel('Sentence Length', fontsize=12)
    plt.title('Sentence Length Distribution', fontsize=14)
    plt.savefig(os.path.join(outdir, f'sentence_len_violin_dist_{quantile}_quantile.svg'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_token_len_dist(data, outdir, quantile=0.999):
    for aa_mapping in [2, 4, 8, 12, 20]:
        lens = data[f'ProtBERTa_{aa_mapping}']['all_token_lens']
        size = np.quantile(lens, quantile)
        sns.kdeplot(data[f'ProtBERTa_{aa_mapping}']['all_token_lens'], color=COLORS[aa_mapping], linewidth=2,
                    fill=True, bw_adjust=22, clip=(1, size))

    plt.title('Distribution of Token Length per Amino Acid Mapping')
    plt.xlabel('Token Length')
    plt.ylabel('Density')
    plt.legend(labels=CODING_LABELS, prop={'size': 12})
    plt.savefig(os.path.join(outdir, f'token_len_{quantile}_quantile_dist.svg'), dpi=300, bbox_inches='tight')
    plt.close()


def get_tokens(sentence, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(sentence['input_ids'])
    sentence['tokens'] = tokens
    return sentence


def get_lengths(sentence):
    sentence['token_lengths'] = [len(token) for token in sentence['tokens']]
    sentence['sum_token_lengths'] = sum(sentence['token_lengths'])
    sentence['avg_length'] = np.mean(sentence['token_lengths'])
    sentence['sentence_length'] = len(sentence['token_lengths'])
    return sentence


def get_tokenizer_properties(tokenizer_prefix, test_dataset, output_dir, col='prot', ncpu=10):
    all_data = {}
    common_tokens = {}
    len_data = []
    for aa_mapping in [2, 4, 8, 12, 20]:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_prefix + str(aa_mapping), add_prefix_space=False, truncation=False, pad_to_max_length=False, padding=False)
        test = get_tokenizer_dataset(test_dataset, mapping_code=aa_mapping, col=col)['train']
        test = test.map(lambda e: tokenizer(e[col], truncation=False, padding=False, add_special_tokens=False), batched=True, keep_in_memory=False, num_proc=ncpu)

        test = test.map(lambda x: get_tokens(x, tokenizer), batched=False, num_proc=ncpu)  # getting tokens
        test = test.map(lambda x: get_lengths(x), batched=False, num_proc=ncpu)  # calculating length

        counts = Counter()
        for i in range(len(test)):
            counts.update(test[i]['input_ids'])
        sorted_tokens = [a for a, b in counts.most_common()]
        common_tokens[f'ProtBERTa_{aa_mapping}'] = [tokenizer.decode(token_id) for token_id in sorted_tokens[:10]]

        token_lens = [element for sublist in test['token_lengths'] for element in sublist]
        all_data[f'ProtBERTa_{aa_mapping}'] = {'seq_lens': test['sentence_length'], 'avg_token_len_per_sent':test['avg_length'], 'all_token_lens': token_lens, 'token_counts': counts, 'sorted_tokens': sorted_tokens}
        len_data.append({'Model': f'ProtBERTa_{aa_mapping}', 'Token Length': f"{round(np.mean(token_lens), 2)}±{round(np.std(token_lens), 2)}",
                            'Sentence Length': f"{round(np.mean(test['sentence_length']), 2)}±{round(np.std(test['sentence_length']), 2)}"})

    df = pd.DataFrame(len_data)
    df.to_csv(os.path.join(output_dir, 'sentence_and_token_lens.tsv'), sep='\t', index=False)

    with open(os.path.join(output_dir, 'all_tokenizer_properties.pkl'), 'wb') as fout:
        pickle.dump(all_data, fout)

    # Creating plots from data
    plot_sentence_len_dist(all_data, output_dir)
    plot_avg_token_len_violin(all_data, output_dir)
    plot_sentence_len_violin(all_data, output_dir)
    plot_token_len_dist(all_data, output_dir)

    ctdf = pd.DataFrame(common_tokens).reset_index().rename({'index': 'rank'}, axis=1)
    ctdf['rank'] += 1
    ctdf.to_csv(os.path.join(output_dir, 'most_common_tokens.tsv'), sep='\t', index=False)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Analysis of the tokenizer properties')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--dataset', help='path to a directory with .csv files')
    parser.add_argument('--out_dir', help='path to a directory to save results')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    args = parser.parse_args()

    get_tokenizer_properties(args.tokenizer_prefix, args.dataset, args.out_dir, args.col_name, args.ncpu)
