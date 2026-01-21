import json
import pandas as pd
import numpy as np
from collections import defaultdict
import requests as r
from Bio import SeqIO
from io import StringIO
import os
import time

BASE_URL = "http://www.uniprot.org/uniprot/"
FAILED_IDS_URL = "https://rest.uniprot.org/unisave/"


def get_uniprot_protein(prot_id, second_try=False):
    if second_try:
        rel_url = f'{FAILED_IDS_URL}{prot_id}?format=fasta&uniqueSequences=true'
        print(rel_url)
    else:
        rel_url = f'{BASE_URL}{prot_id}.fasta'
    response = r.get(rel_url)
    data = ''.join(response.text)
    if second_try:
        print(data)
    Seq = StringIO(data)
    recs = list(SeqIO.parse(Seq, 'fasta'))
    return recs


def get_protein_sequences(prot_ids, output_fasta):
    all_recs = []
    for prot_id in prot_ids:
        try:
            time.sleep(0.05)
            recs = get_uniprot_protein(prot_id, False)
            if not recs:
                raise Exception
            all_recs += recs
        except Exception as e:
            try:
                time.sleep(0.3)
                # If the protein ID is not found, we can try to fetch it from the failed IDs URL
                recs = get_uniprot_protein(prot_id, True)
                if not recs:
                    raise Exception
                all_recs += recs
                print('try 2 successful for:', prot_id)
            except Exception as e:
                print(f"Failed to retrieve {prot_id}: {e}")

        time.sleep(0.1)  # To avoid hitting the server too hard

    SeqIO.write(all_recs, output_fasta, 'fasta')
    prot_d = pd.DataFrame.from_dict({rec.id: str(rec.seq) for rec in all_recs}, orient='index', columns=['prot']).reset_index()
    prot_d['ID'] = prot_d['index'].apply(lambda x:x.split('|')[1])
    return prot_d[['ID', 'prot']].set_index('ID')


def cluster_proteins(mmseqs_path, fasta_file, sensitivity=7.5, coverage=0.8, ncpus=4):
    tmp_file = fasta_file.replace('.fasta', '_mmseqs_tmp')
    command = f"{mmseqs_path} easy-cluster {fasta_file} {fasta_file.replace('.fasta', '')} {tmp_file} -s {sensitivity} -c {coverage} --threads {ncpus} -v 2"
    os.system(command)
    os.system(f"rm -r {tmp_file}")


def split_train_test(pdf, task_list, fasta_file, out_dir, mmseqs_path):
    # Splitting BRENDA dataset into train and test sets such that range values are in test
    cluster_proteins(mmseqs_path, fasta_file, sensitivity=7.5, coverage=0.8, ncpus=4)
    clus = pd.read_csv(fasta_file.replace('.fasta', '_cluster.tsv'), sep='\t', names=['rep', 'ID'])
    clus_dict = clus.set_index('ID')['rep'].to_dict()

    for task in task_list:
        rel_df = pdf[[task, 'prot']].dropna().reset_index()
        rel_df = rel_df.rename(columns={task: 'label'})
        test_df_range = rel_df[rel_df['label'].str.contains('-')]
        test_reps = [clus_dict.get(id, id) for id in test_df_range['ID']]
        all_test_inds = clus[clus['rep'].isin(test_reps)]['ID'].tolist()
        train_df = rel_df[~rel_df['ID'].isin(all_test_inds)]
        test_df = rel_df[rel_df['ID'].isin(all_test_inds)]
        print(f"Saving train and test for {task} with {len(train_df)} train and {len(test_df)} test samples, out of which {len(test_df_range)} is range.")
        test_df = test_df[~test_df['ID'].isin(test_df_range['ID'])]  # taking all non-range values

        # creating output directory
        rel_out_dir = os.path.join(out_dir, task)
        os.makedirs(rel_out_dir, exist_ok=True)
        os.makedirs(os.path.join(rel_out_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(rel_out_dir, 'test'), exist_ok=True)
        train_file = os.path.join(rel_out_dir, 'train', task + '_train.csv')
        test_file = os.path.join(rel_out_dir, 'test', task + '_test.csv')
        test_range_file = os.path.join(rel_out_dir, task + '_test_range_values.csv')

        train_df['label'] = train_df['label'].apply(float)
        test_df['label'] = test_df['label'].apply(float)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        test_df_range.to_csv(test_range_file, index=False)


def process_BRENDA_data(json_file, out_dir, mmseqs_path, rel_values=['temperature_optimum']):
    with open(json_file, 'r', encoding='UTF-8') as fin:
        d = json.load(fin)['data']

    acc_to_metrics = {}
    for ec_num, data in d.items():
        if 'protein' not in data:
            continue
        id_to_acc = {k: v['accessions'] for k, v in data['protein'].items() if 'accessions' in v}
        id_to_metrics = defaultdict(dict)
        for metric in rel_values:
            if metric not in data:
                continue
            rel_d = data[metric]
            for info in rel_d:
                if 'value' not in info:
                    continue
                value = info['value'].split(' {')[0]
                for prot in info['proteins']:
                    id_to_metrics[prot][metric] = np.nan if value == '-999' else value
        for k, v in id_to_acc.items():
            for acc in v:
                acc_to_metrics[acc] = id_to_metrics[k]

    df = pd.DataFrame(acc_to_metrics).T.reset_index().rename(columns={'index': 'ID'}).set_index('ID').dropna(how='all')
    rel_ids = df.index.tolist()

    fasta_file = os.path.join(out_dir, 'brenda_proteins.fasta')
    seq_df = get_protein_sequences(rel_ids, fasta_file)
    df = df.join(seq_df, how='inner')
    split_train_test(df, rel_values, fasta_file, out_dir, mmseqs_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Processing of TAPE datasets')
    parser.add_argument('--input_file', help='path to a the json with relevant data')
    parser.add_argument('--output_dir', help='Path to output directory where data will be saved')
    parser.add_argument('--mmseqs_path', '-mm', help='Path to MMseqs2 program for clustering the data')
    parser.add_argument('--values', nargs='+', help='List of values to take from BRENDA dataset, each one will be a task. Default: [temperature_optimum]', default=['temperature_optimum'])
    args = parser.parse_args()
    process_BRENDA_data(args.input_file, args.output_dir, args.mmseqs_path, args.values)

