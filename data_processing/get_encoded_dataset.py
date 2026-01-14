import os
import glob
import numpy as np
from pathlib import Path
from datasets import load_dataset


# Mapping based on Google Search, L - Hydrophilic, B - Hydrophobic
HYDROPHILIC_PHOBIC = {'S': 'L', 'T': 'L', 'N': 'L', 'K': 'L', 'E': 'L', 'Q': 'L', 'H': 'L', 'D': 'L', 'R': 'L', 'Z': 'L', 'B': 'L', 'A': 'B', 'G': 'B', 'I': 'B', 'L': 'B', 'J': 'B', 'M': 'B', 'V': 'B', 'P': 'B', 'F': 'B', 'W': 'B', 'C': 'B', 'Y': 'B'}
# Taken from here: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Basics_of_General_Organic_and_Biological_Chemistry_(Ball_et_al.)/18%3A_Amino_Acids_Proteins_and_Enzymes/18.01%3A_Properties_of_Amino_Acids
# P - Polar, N- non-polar, S- Negative, H- positive
POLARITY = {'G': 'N', 'A': 'N', 'V': 'N', 'L': 'N', 'I': 'N', 'J': 'N', 'F': 'N', 'W': 'N', 'M': 'N', 'P': 'N', 'S': 'P', 'T': 'P', 'C': 'P', 'Y': 'P', 'N': 'P', 'Q': 'P', 'D': 'S', 'E': 'S', 'H': 'H', 'K': 'H', 'R': 'H', 'Z': 'Z', 'B': 'B'}
# Taken from here: https://soe.unipune.ac.in/studymaterial/swapnaGaikwadOnline/aminoacids-171113130407.pdf
# S-Simple, H-Hydroxy, F-Sulfur-containing, M-Aromatic, C-Heterocyclic, N-Amine-containing, A-Acidic, B-Basic.
# 'B' (Aspartic acid or Asparagine) is translated to P to remove confusion with Basic
FUNCTIONALITY = {'G': 'S', 'V': 'S', 'A': 'S', 'L': 'S', 'I': 'S', 'J': 'S', 'S': 'H', 'T': 'H', 'C': 'F', 'M': 'F', 'F': 'M', 'Y': 'M', 'W': 'C', 'H': 'C', 'P': 'C', 'N': 'N', 'Q': 'N', 'D': 'A', 'E': 'A', 'K': 'B', 'R': 'B', 'Z': 'Z', 'B': 'P'}
# Alphabet used by LinClust for fast clustering of proteins (https://www.nature.com/articles/s41467-018-04964-5#Sec9)
MMSEQS = {'A': 'A', 'S': 'A', 'T': 'A', 'L': 'L', 'M': 'L', 'I': 'I', 'V': 'I', 'K': 'K', 'R': 'K', 'E':'E', 'Q': 'E', 'Z':'E', 'N': 'N', 'D': 'N', 'B':'N', 'F': 'F', 'Y': 'F', 'C':'C', 'G':'G', 'H':'H', 'P':'P', 'W': 'W'}
AA_MAPPING_DICT = {2: str.maketrans(HYDROPHILIC_PHOBIC), 4: str.maketrans(POLARITY), 8: str.maketrans(FUNCTIONALITY), 12: str.maketrans(MMSEQS), 20: {}}
CACHE_DIR = os.path.join(Path(__name__).parent.absolute(), "cache")


def map_amino_acids(data, mapping_code=20, col='prot'):
    mapping = AA_MAPPING_DICT[mapping_code]
    # encode amino acids according to dictionary
    if mapping:
        data[col] = np.char.translate(data[col], mapping).item()
    return data


def get_tokenizer_dataset(data_dir, mapping_code=20, file_num=0):
    pattern = os.path.join(data_dir, "*.csv")
    data_files = glob.glob(pattern)
    if file_num > 0:
       data_files = data_files[:file_num]
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset = load_dataset('csv', data_files=data_files, cache_dir=CACHE_DIR)
    dataset = dataset.filter(lambda x: '*' not in x['prot'][:-1])  # remove proteins those with * in the middle
    dataset = dataset.map(lambda x: map_amino_acids(x, mapping_code))
    return dataset


def get_train_test(data_dir, mapping_code=20, train_file_num=0, proc=10, is_eval=False, prot_sample=0, multi_dir=False):
    test_str = "eval" if is_eval else 'test'
    train_pattern = os.path.join(data_dir, 'train', '*' if multi_dir else '', '*.csv')
    test_pattern = os.path.join(data_dir, test_str, '*' if multi_dir and not is_eval else '', '*.csv')
    file_mapping = {'train': glob.glob(train_pattern), "test": glob.glob(test_pattern)}
    if train_file_num > 0:
        file_mapping['train'], file_mapping['test'] = file_mapping['train'][:train_file_num], file_mapping['test'][:int(train_file_num/4)]
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset = load_dataset('csv', data_files=file_mapping, cache_dir=CACHE_DIR)
    train, test = dataset['train'], dataset['test']

    train = train.shuffle(seed=42).select(range(prot_sample)) if prot_sample > 0 and len(train) > prot_sample else train
    if mapping_code != 20:
        train = train.map(lambda x: map_amino_acids(x, mapping_code), num_proc=proc)
        test = test.map(lambda x: map_amino_acids(x, mapping_code), num_proc=proc)
    return train, test


def get_downstream_train_test(data_dir, mapping_code=20, train_file_num=0, proc=10, get_val=False):
    train_pattern = os.path.join(data_dir, 'train', '*.csv')
    test_pattern = os.path.join(data_dir, 'test', '*.csv')
    file_mapping = {'train': glob.glob(train_pattern), "test": glob.glob(test_pattern)}
    if get_val:
        val_pattern = os.path.join(data_dir, 'validation', '*.csv')
        file_mapping['val'] = glob.glob(val_pattern)
    if train_file_num > 0:
        file_mapping['train'], file_mapping['test'] = file_mapping['train'][:train_file_num], file_mapping['test'][:int(train_file_num/4)]
    dataset = load_dataset('csv', data_files=file_mapping, cache_dir=CACHE_DIR)
    train, test = dataset['train'], dataset['test']
    val = dataset['val'] if get_val else None
    if mapping_code != 20:
        train = train.map(lambda x: map_amino_acids(x, mapping_code), num_proc=proc)
        test = test.map(lambda x: map_amino_acids(x, mapping_code), num_proc=proc)
        if get_val:
            val = val.map(lambda x: map_amino_acids(x, mapping_code), num_proc=proc)
    return train, test, val
