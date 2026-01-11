import os
import pandas as pd
from Bio import SeqIO
import numpy as np
from math import comb
from collections import Counter
from random import sample


def create_intra_family_pairs(df, n_pairs):
    rel_pairs = set()
    for family, group in df.groupby('family'):
        print(family, flush=True)
        l = group['ID'].tolist()
        # sampling positive interactions.
        first_items = sample(list(range(len(l)-1)), n_pairs)
        # sampling item that comes after the first one, so we won't have duplicates
        second_items = [sample(list(range(i+1, len(l))), 1)[0] for i in first_items]
        pairs = [(l[i1], l[i2]) for i1, i2 in zip(first_items, second_items)]
        rel_pairs.update(list(pairs))
    return rel_pairs


def create_inter_family_pairs(groups, n_pairs):
    n_groups = len(groups)
    pairs = set()
    group_pair_counts = Counter()
    for group_ind, group in enumerate(groups):
        while group_pair_counts[group_ind] < n_pairs:
            possible_groups = [i for i in range(group_ind+1, n_groups) if group_pair_counts[i] < n_pairs]
            if not possible_groups:  # All groups already has n_pairs pairs, choose one that has the least pairs
                min_pair_count = min([count for count in group_pair_counts.values() if count >= n_pairs]) # not taking ourselves
                possible_groups = [i for i in range(n_groups) if i != group_ind and group_pair_counts[i] == min_pair_count]
            other_group_ind = sample(possible_groups, 1)[0]
            pair = None
            while pair is None:  # sampling until we find new pair
                first_item = sample(group['ID'].tolist(), 1)[0]
                second_item = sample(groups[other_group_ind]['ID'].tolist(), 1)[0]
                if (first_item, second_item) not in pairs:
                    pair = (first_item, second_item)
            pairs.add(pair)
            group_pair_counts[group_ind] += 1
            group_pair_counts[other_group_ind] += 1
    return pairs


def create_pair_df(df, pairs, label):
    pairs = list(pairs)
    df1 = df.loc[[p[0] for p in pairs]].reset_index()
    df2 = df.loc[[p[1] for p in pairs]].reset_index()
    united = df1.join(df2, lsuffix='_1', rsuffix='_2')
    united['label'] = label
    return united


def add_sequences(df, fasta_dir):
    sequences = {}
    all_families = set(df['family_1']).union(set(df['family_2']))
    print(all_families, flush=True)
    for family in all_families:
        rel_ids = set(df[df['family_1'] == family]['ID_1'].tolist() + df[df['family_2'] == family]['ID_2'].tolist())
        sequences.update({rec.id: str(rec.seq) for rec in SeqIO.parse(os.path.join(fasta_dir, f'{family}.fasta'), 'fasta') if rec.id in rel_ids})

    df['prot_1'] = df['ID_1'].map(sequences)
    df['prot_2'] = df['ID_2'].map(sequences)


def create_homology_pairs_data(prot_families_dir, output_file):
    all_res = []
    for file_name in os.listdir(prot_families_dir):
        if file_name.endswith('.fasta'):
            file_path = os.path.join(prot_families_dir, file_name)
            family_id = file_name.split('.')[0]  # Extracting ID from the file name

            # Read the FASTA file
            rec_dict = [{'ID': rec.id, 'family': family_id} for rec in SeqIO.parse(file_path, 'fasta')]
            all_res.append(rec_dict)
            print(f"Processed family: {family_id}, Number of proteins: {len(rec_dict)}", flush=True)
    all_res = [item for sublist in all_res for item in sublist]  # Flatten the list of lists
    df = pd.DataFrame(all_res).drop_duplicates('ID')  # removing Ids from more than one family

    # removing small families
    sizes = df['family'].value_counts()
    min_family_size = np.quantile(sizes, 0.1)  # Minimum family size threshold
    to_rm = sizes < min_family_size
    to_rm = to_rm[to_rm].index.get_level_values(0).unique()
    df = df[~df['family'].isin(to_rm)]
    min_family_size = df['family'].value_counts().min()

    n_pairs = min(comb(min_family_size, 2), 15)
    print(f"Minimum family size: {min_family_size}", flush=True)
    print(f"Number of pairs sampled for each family: {n_pairs}", flush=True)

    pos_pairs = create_intra_family_pairs(df, n_pairs)
    # choosing 2*n_pairs since we want label ratio to be 1:1,
    # and in this function we count n_pairs for each group, and each pair has one count for each group
    neg_pairs = create_inter_family_pairs([group for _, group in df.groupby('family')], 2 * n_pairs)

    df = df.set_index('ID')
    pos_df = create_pair_df(df, pos_pairs, 1)
    neg_df = create_pair_df(df, neg_pairs, 0)
    pair_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, replace=False)
    add_sequences(pair_df, prot_families_dir)
    pair_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Creating a homology pairs dataset from protein families')
    parser.add_argument('--family_dir', help='path to directory with a fasta file per protein family')
    parser.add_argument('--output_file', help='Path to output csv file')
    args = parser.parse_args()

    create_homology_pairs_data(args.family_dir, args.output_file)


