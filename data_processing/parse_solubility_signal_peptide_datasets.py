from Bio import SeqIO
import pandas as pd
import os

LABEL_BINARY_MAPPING = {"Extracellular": 1, "CYtoplasmicMembrane": 0, "Cytoplasmic": 1, "OuterMembrane": 0, "Cellwall": 0, "Periplasmic": 1}
LABEL_MULTI_MAPPING = {"Extracellular": 0, "Cellwall": 1, "OuterMembrane": 2, "Periplasmic": 3, "CYtoplasmicMembrane": 4, "Cytoplasmic": 5}


def parse_fasta(fasta, is_cv=False):
    recs = [{'ID': rec.id, 'prot': str(rec.seq)} for rec in SeqIO.parse(fasta, 'fasta')]
    df = pd.DataFrame(recs)
    if is_cv:
        df[['ID', 'label', 'fold']] = df['ID'].apply(lambda x: pd.Series(x.split('|'))[[0, 1, 3]])
    else:
        df[['ID', 'label']] = df['ID'].apply(lambda x: pd.Series(x.split('|'))[[0, 1]])

    df['label_multiclass'] = df['label'].map(LABEL_MULTI_MAPPING)
    df['label'] = df['label'].map(LABEL_BINARY_MAPPING)
    return df


def parse_deeploc_pro_dataset(input_fasta, output_path, is_cv=True):
    df = parse_fasta(input_fasta, is_cv=is_cv)
    if is_cv:
        train, test = df[df['fold'] != '4'], df[df['fold'] == '4']
        output_prefix = os.path.split(input_fasta)[-1].replace('_cv_set.fasta', '.csv')
        train.drop('fold', axis=1).to_csv(os.path.join(output_path, 'train', output_prefix), index=False)
        test.drop('fold', axis=1).to_csv(os.path.join(output_path, 'test', output_prefix), index=False)
    else:
        output_prefix = os.path.split(input_fasta)[-1].replace('.fasta', '.csv')
        df.to_csv(os.path.join(output_path, 'train', output_prefix), index=False)


def parse_signalp_dataset(input_fasta, output_path, is_cv=False):
    with open(input_fasta, 'r') as f:
        recs = f.read().split('>')[1:]
    rel_recs = []
    for rec in recs:
        if 'EUKARYA' in rec:
            continue
        lines = rec.strip().split('\n')
        header = lines[0].split('|')
        rel_recs.append({'ID': '|'.join(header[:2]), 'label': 0 if header[2] == 'NO_SP' else 1, 'fold': header[3], 'prot': lines[1].strip()})
    df = pd.DataFrame(rel_recs)
    if is_cv:
        train, test = df[df['fold'] != '2'], df[df['fold'] == '2']
        train.drop('fold', axis=1).to_csv(os.path.join(output_path, 'train', 'signalP6.csv'), index=False)
        test.drop('fold', axis=1).to_csv(os.path.join(output_path, 'test', 'signalP6.csv'), index=False)
    else:
        output_prefix = os.path.split(input_fasta)[-1].replace('.fasta', '.csv')
        df.to_csv(os.path.join(output_path, output_prefix), index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Processing of solubility and signal peptide datasets')
    parser.add_argument('--is_peptide', action='store_true', help='Choose this to process the signal peptide dataset. Default: False')
    parser.add_argument('--is_cv', action='store_true', help='Choose this if the input fasta contain split by fold, which is used to split to train and test. Default: False')
    parser.add_argument('--input_fasta', help='path to a the fasta with relevant sequences')
    parser.add_argument('--output_dir', help='Path to output directory')
    parser.set_defaults(is_peptide=False)
    parser.set_defaults(is_cv=False)
    args = parser.parse_args()

    if args.is_peptide:
        parse_signalp_dataset(args.input_fasta, args.output_dir, args.is_cv)
    else:
        parse_deeploc_pro_dataset(args.input_fasta, args.output_dir, args.is_cv)
