import os
import glob
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def prepare_united_fasta(train, test, output_file, id_col='ID', seq_col='prot'):
    train['record'] = train.apply(lambda row: SeqRecord(Seq(row[seq_col]), id=row[id_col], description=''), axis=1)
    test['record'] = test.apply(lambda row: SeqRecord(Seq(row[seq_col]), id=row[id_col], description=''), axis=1)
    recs = train['record'].tolist() + test['record'].tolist()
    SeqIO.write(recs, output_file, "fasta")


def parse_clusters_mmseqs(clu_file):
    df = pd.read_csv(clu_file, sep='\t', names=['rep', 'member'])
    clus = df.groupby('rep')['member'].agg(list)
    return clus


def remove_similar_test_proteins(data_dir, output_file, tmp_dir, threshold=0.5, id_col='ID', seq_col='prot', mmseqs_path='mmseqs'):
    infasta = os.path.join(data_dir, 'tmp_fasta_for_cdhit.fasta')
    linc_outfile = os.path.join(data_dir, f'linclust_train_test_{threshold}')
    test_to_keep = []
    org_id_col = id_col

    train_files = [f for f in glob.glob(os.path.join(data_dir, 'train', '*.csv'))]
    test_files = [f for f in glob.glob(os.path.join(data_dir, 'test', '*.csv'))]
    train, test = pd.concat([pd.read_csv(f) for f in train_files]), pd.concat([pd.read_csv(f) for f in test_files])
    if id_col is None:
        id_col = 'index'
        train, test = train.reset_index(), test.reset_index()
        train[id_col] = train[id_col].apply(lambda x: f'train_{x}')
        test[id_col] = test[id_col].apply(lambda x: f'test_{x}')

    orig_test_len = len(test)

    prepare_united_fasta(train, test, infasta, id_col, seq_col)
    train, test = train.drop('record', axis=1), test.drop('record', axis=1)
    command = f"{mmseqs_path} easy-linclust {infasta} {linc_outfile} {tmp_dir} --min-seq-id {threshold} -c 0.8"
    os.system(command)

    clusters = parse_clusters_mmseqs(linc_outfile + '_cluster.tsv')
    train_ids = set(train[id_col])
    for clu in clusters:  # removing clusters with train proteins
        if not any([ind in train_ids for ind in clu]):
            test_to_keep += clu

    test = test.set_index(id_col).loc[test_to_keep].reset_index()
    print(f'There are {len(test)} proteins left in test out of {orig_test_len}')
    if org_id_col is None:
        test = test.drop(id_col, axis=1)

    os.system(f'rm -r {infasta} {linc_outfile}* {os.path.join(tmp_dir, "*")}')
    test.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Removing from test proteins similar to train proteins')
    parser.add_argument('--data_dir', help='path to data directory with train and test subdirs')
    parser.add_argument('--output_file', help='path to output file with filtered test set')
    parser.add_argument('--tmp_dir', help='path to temporary directory for mmseqs intermediate files')
    parser.add_argument('--threshold', type=float, default=0.5, help='similarity threshold for cd-hit. Default: 0.5')
    parser.add_argument('--id_col', default='ID', help='name of the column with protein IDs in csv files. Default: ID')
    parser.add_argument('--seq_col', default='prot', help='name of the column with protein sequences in csv files. Default: prot')
    parser.add_argument('--mmseqs_path', default='mmseqs', help='path to mmseqs executable. Default: mmseqs, which means it should be in PATH variable. Otherwise, provide path to mmseqs executable')
    args = parser.parse_args()

    remove_similar_test_proteins(args.data_dir, args.output_file, args.tmp_dir, args.threshold, args.id_col,
                                 args.seq_col, args.mmseqs_path)
