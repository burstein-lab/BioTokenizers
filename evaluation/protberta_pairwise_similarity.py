import os
import torch
import pandas as pd
import glob
from torch.nn.functional import cosine_similarity
from datasets import load_dataset
from data_processing.get_encoded_dataset import map_amino_acids
from utilities import load_tokenizer, load_model
from eval_utilities import calc_metrics, return_computed_metrics, plot_all_metric_results
from model_training.roberta_with_advanced_pooling import mean_pooling
from pathlib import Path

AA_MAPPINGS = [2, 4, 8, 12, 20]
MAX_LEN = 1026
DIR_PATH = Path(__name__).parent.absolute()


def get_token_ids(sentence, tokenizer):
    tok_res_1 = tokenizer(sentence['prot_1'], truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
    sentence['input_ids_1'] = tok_res_1['input_ids']
    sentence['attention_mask_1'] = tok_res_1['attention_mask']

    tok_res_2 = tokenizer(sentence['prot_2'], truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
    sentence['input_ids_2'] = tok_res_2['input_ids']
    sentence['attention_mask_2'] = tok_res_2['attention_mask']
    return sentence


def get_similarity(batch, model, device):
    att1, att2 = batch['attention_mask_1'].to(device), batch['attention_mask_2'].to(device)
    try:
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


def get_all_metrics(df):
    probs = torch.Tensor(df['similarity'].astype('float64').values)
    probs = torch.cat([1 - probs.unsqueeze(1), probs.unsqueeze(1)], dim=1)
    res = return_computed_metrics(probs, df['label'].astype('int32').values)
    rocauc_score, prauc_score, chosen_precision, chosen_recall, chosen_f1, mcc_prec, mcc_rec, mcc = calc_metrics(df['label'].tolist(), probs[:, 1].tolist())
    res.update({'auroc': rocauc_score, 'aupr': prauc_score, 'Precision of Best F1': chosen_precision,
                'Recall of Best F1': chosen_recall, 'Best F1': chosen_f1, 'Precision of Best MCC': mcc_prec,
                'Recall of Best MCC': mcc_rec, 'Best MCC': mcc})
    return res


def get_pairwise_similarity(dataset, aa_mapping, proc=10):
    data_files = glob.glob(os.path.join(dataset, "*.csv"))
    dataset = load_dataset('csv', data_files=data_files, cache_dir=os.path.join(DIR_PATH, "cache"))['train']
    dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_1'), num_proc=proc)
    dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_2'), num_proc=proc)

    tokenizer_file = os.path.join(DIR_PATH, 'tokenizers', f'BPE_tokenizer_prot_5000_min_freq_2_mapping_{aa_mapping}')
    tokenizer = load_tokenizer(tokenizer_file, MAX_LEN)
    dataset = dataset.map(lambda x: get_token_ids(x, tokenizer), batched=True, num_proc=proc)
    dataset.set_format(type="torch", columns=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'])

    model_name = os.path.join(DIR_PATH, 'models', f'pretrained-ProtBERTa_{aa_mapping}_1024/5_epochs/')
    model, device = load_model(model_name, 4)
    dataset = dataset.map(lambda x: get_similarity(x, model, device), batched=True, num_proc=1, batch_size=16)

    df = dataset.to_pandas()
    df['similarity'] = df['similarity'].apply(lambda x: x if x <= 1.0 else 1.0)  # floating point errors
    return df


def eval_ProtBERTa_pairwise_similarity(dataset, output_dir, output_prefix, proc=10):
    all_res = {}
    for aa_mapping in AA_MAPPINGS:
        print(aa_mapping, flush=True)
        df = get_pairwise_similarity(dataset, aa_mapping, proc=proc)
        res_dict = get_all_metrics(df)  # Overall Metrics
        all_res[f'ProtBERTa_{aa_mapping}'] = res_dict

    res_df = pd.DataFrame(all_res).T
    res_df.to_csv(os.path.join(output_dir, f'{output_prefix}_overall.tsv'), sep='\t')

    plot_all_metric_results(res_df[["precision_macro", "recall_macro", "f1_macro", 'accuracy']], os.path.join(output_dir, f'{output_prefix}_res_macro.svg'), metric='macro')
    plot_all_metric_results(res_df[["precision_weighted", "recall_weighted", "f1_weighted", 'accuracy']], os.path.join(output_dir, f'{output_prefix}_res_weighted.svg'), metric='weighted')
    plot_all_metric_results(res_df[['Precision of Best F1', 'Recall of Best F1', 'Best F1']], os.path.join(output_dir, f'{output_prefix}_best_f1.svg'), metric='')
    plot_all_metric_results(res_df[['Precision of Best MCC', 'Recall of Best MCC', 'Best MCC']], os.path.join(output_dir, f'{output_prefix}_best_mcc.svg'), metric='')
    plot_all_metric_results(res_df[['auroc', 'aupr']], os.path.join(output_dir, f'{output_prefix}_auroc_aupr.svg'), metric='')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Running Zero-shot pair-wise classification with a pre-trained ProtBERTa model')
    parser.add_argument('--dataset', help='path to a directory with .csv files containing protein pairs and labels. Columns should be prot_1, prot_2, label')
    parser.add_argument('--output_dir', help='path to output directory for results')
    parser.add_argument('--output_prefix', help='Prefix to output files.')
    parser.add_argument('--ncpu', type=int, default=10, help='Number of cpus to use. Default: 10')
    args = parser.parse_args()

    eval_ProtBERTa_pairwise_similarity(args.dataset, args.output_dir, args.output_prefix, proc=args.ncpu)
