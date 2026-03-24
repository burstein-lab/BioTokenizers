import os
import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from torch.nn.functional import cosine_similarity
from datasets import load_dataset
from data_processing.get_encoded_dataset import map_amino_acids
from utilities import load_tokenizer, load_model
from evaluation.eval_utilities import calc_metrics, return_all_eval_metrics_dict, plot_all_metric_results_with_SE, plot_binary_AUPR_AUROC_with_SE, finish_AUPR_AUROC_figure, COLORS
from model_training.roberta_with_advanced_pooling import mean_pooling
from pathlib import Path

AA_MAPPINGS = [2, 4, 8, 12, 20]
DIR_PATH = Path(__name__).parent.absolute()


def get_token_ids(sentence, tokenizer, max_len=1024):
    tok_res_1 = tokenizer(sentence['prot_1'], truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    sentence['input_ids_1'] = tok_res_1['input_ids']
    sentence['attention_mask_1'] = tok_res_1['attention_mask']

    tok_res_2 = tokenizer(sentence['prot_2'], truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    sentence['input_ids_2'] = tok_res_2['input_ids']
    sentence['attention_mask_2'] = tok_res_2['attention_mask']
    return sentence


def get_similarity(batch, model, device):
    att1, att2 = batch['attention_mask_1'].to(device), batch['attention_mask_2'].to(device)
    try:
        with torch.no_grad():
            output1 = model(input_ids=batch['input_ids_1'].to(device), attention_mask=att1)
    except Exception:
        print("input_ids_1:", batch['input_ids_1'].shape, "max id:", batch['input_ids_1'].max().item())
        print("att1:", batch['attention_mask_1'].shape)
        print("input_ids_2:", batch['input_ids_2'].shape, "max id:", batch['input_ids_2'].max().item())
        print("att2:", batch['attention_mask_2'].shape)
        print("vocab_size:", model.config.vocab_size)
        raise

    sequence_output1 = output1.last_hidden_state
    emb1 = mean_pooling(sequence_output1, att1)

    output2 = model(input_ids=batch['input_ids_2'].to(device), attention_mask=att2)
    sequence_output2 = output2.last_hidden_state
    emb2 = mean_pooling(sequence_output2, att2)

    batch['similarity'] = cosine_similarity(emb1, emb2).cpu()
    return batch


def get_all_metrics_se(df, n_bootstrap=100):
    bootstrapped_stats = []
    se_results = {}
    o_stats = get_all_metrics(df)
    se_results.update(o_stats)

    # Ensure inputs are numpy arrays
    for i in range(n_bootstrap):
        # Resample with replacement
        resampled_df = df.sample(frac=1, replace=True, random_state=i)
        # Calculate metrics for this bootstrap sample
        stats = get_all_metrics(resampled_df)
        bootstrapped_stats.append(stats)

    # Aggregate results into Confidence Intervals
    metrics = bootstrapped_stats[0].keys()

    for metric in metrics:
        values = [s[metric] for s in bootstrapped_stats]
        se = np.std(values) / np.sqrt(len(values))
        se_results[f"{metric}_se"] = se
        se_results[f"{metric}_bootstrap"] = np.array(values)  # Store all bootstrap values for potential further analysis
    return se_results


def get_all_metrics(df):
    probs = torch.Tensor(df['similarity'].astype('float64').values)
    probs = torch.cat([1 - probs.unsqueeze(1), probs.unsqueeze(1)], dim=1)
    res = return_all_eval_metrics_dict(probs, df['label'].astype('int32').values)
    rocauc_score, prauc_score, chosen_precision, chosen_recall, chosen_f1, mcc_prec, mcc_rec, mcc = calc_metrics(df['label'].tolist(), probs[:, 1].tolist())
    res.update({'auroc': rocauc_score, 'aupr': prauc_score, 'Precision of Best F1': chosen_precision,
                'Recall of Best F1': chosen_recall, 'Best F1': chosen_f1, 'Precision of Best MCC': mcc_prec,
                'Recall of Best MCC': mcc_rec, 'Best MCC': mcc})
    return res


def get_pairwise_similarity(dataset, model_path, tokenizer_file, aa_mapping, proc=10, device_num=-1, batch_size=16, max_len=1026):
    data_files = glob.glob(os.path.join(dataset, "*.csv"))
    dataset = load_dataset('csv', data_files=data_files, cache_dir=os.path.join(DIR_PATH, "cache"))['train']
    dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_1'), num_proc=proc)
    dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_2'), num_proc=proc)

    tokenizer = load_tokenizer(tokenizer_file, max_len)
    dataset = dataset.map(lambda x: get_token_ids(x, tokenizer, max_len-2), batched=True, num_proc=proc) # <s> and <\s> are added to sequence
    dataset.set_format(type="torch", columns=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'])

    model, device = load_model(model_path, device_num)
    dataset = dataset.map(lambda x: get_similarity(x, model, device), batched=True, batch_size=batch_size)

    df = dataset.to_pandas()
    df['similarity'] = df['similarity'].apply(lambda x: x if x <= 1.0 else 1.0)  # floating point errors
    return df


def eval_ProtBERTa_pairwise_similarity(dataset, model_path, tokenizer_prefix, output_dir, output_prefix, proc=10, device_num=-1, batch_size=16, max_len=1026, n_bootstrap=100):
    all_res = {}
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    for aa_mapping in AA_MAPPINGS:
        print(aa_mapping, flush=True)
        tokenizer_path = tokenizer_prefix + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        df = get_pairwise_similarity(dataset, model_path, tokenizer_path, aa_mapping, proc=proc, device_num=device_num, batch_size=batch_size, max_len=max_len)

        probs = torch.Tensor(df['similarity'].astype('float64').values)
        probs = torch.cat([1 - probs.unsqueeze(1), probs.unsqueeze(1)], dim=1)
        plot_binary_AUPR_AUROC_with_SE(df['label'].values, probs, f'ProtBERTa_{aa_mapping}', COLORS[f'ProtBERTa_{aa_mapping}'], axes, n_bootstrap=n_bootstrap)

        res_dict = get_all_metrics_se(df, n_bootstrap=n_bootstrap)  # Overall Metrics
        all_res[f'ProtBERTa_{aa_mapping}'] = res_dict

    res_df = pd.DataFrame(all_res).T
    rel_cols = [col for col in res_df.columns if col +'_se' in res_df.columns]
    for col in rel_cols:
        res_df[col] = res_df.apply(lambda r: f'{round(r[col], 3)}±{round(r[col+"_se"], 5)}', axis=1)
    res_df[['Model'] + rel_cols].to_csv(os.path.join(output_dir, f'{output_prefix}_overall.tsv'), sep='\t')

    finish_AUPR_AUROC_figure(f, axes, os.path.join(output_dir, f'{output_prefix}_zeroshot_AUROC_AUPR.svg'))
    plot_all_metric_results_with_SE(res_df[['auroc', 'aupr', 'Best F1', 'auroc_se', 'aupr_se', 'Best F1_se']], os.path.join(output_dir, f'{output_prefix}_auroc_aupr.svg'), metric='')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Running Zero-shot pair-wise classification with a pre-trained ProtBERTa model')
    parser.add_argument('--dataset', help='path to a directory with .csv files containing protein pairs and labels. Columns should be prot_1, prot_2, label')
    parser.add_argument('--model_path', help='path to a the finetuned model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--output_dir', help='path to output directory for results')
    parser.add_argument('--output_prefix', help='Prefix to output files.')
    parser.add_argument('--ncpu', type=int, default=10, help='Number of cpus to use. Default: 10')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size. Default: 16')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--n_bootstrap', type=int, default=100, help='Number of times to resample to calculate confidence interval. default: 100')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    args = parser.parse_args()

    eval_ProtBERTa_pairwise_similarity(args.dataset, args.model_path, args.tokenizer_prefix, args.output_dir,
                                       args.output_prefix, proc=args.ncpu, device_num=args.device, batch_size=args.batch_size, max_len=args.max_length, n_bootstrap=args.n_bootstrap)
