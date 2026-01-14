import os
from tokenizers import ByteLevelBPETokenizer, models, Tokenizer, trainers, pre_tokenizers
from data_processing.get_encoded_dataset import get_tokenizer_dataset
from utilities import clear_cache


ROBERTA_SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]


def get_training_corpus(dataset_dir, col='prot', aa_mapping_code=20):
    dataset = get_tokenizer_dataset(dataset_dir, mapping_code=aa_mapping_code)['train']
    for i in range(0, len(dataset), 1000):
        if i % 1000 == 0:
            clear_cache()
        yield dataset[i: i + 1000][col]


def train_tokenizer(dataset_dir, col_name, vocab_size, min_freq, output_file, aa_mapping=20):
    data_iterator = get_training_corpus(dataset_dir, col_name, aa_mapping)
    train_amino_acid_tokenizer(data_iterator, vocab_size, min_freq, output_file, aa_mapping)


def train_amino_acid_tokenizer(data_iterator, vocab_size, min_freq, output_file, aa_mapping):
    output_file = f'{output_file}_prot_{vocab_size}_min_freq_{min_freq}_mapping_{aa_mapping}'
    os.makedirs(output_file, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(data_iterator, vocab_size=vocab_size, min_frequency=min_freq, special_tokens=ROBERTA_SPECIAL_TOKENS)
    tokenizer.save_model(output_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Training a tokenizer from scratch.')
    parser.add_argument('--dataset_dir', help='path to dir with corpus files')
    parser.add_argument('--output_prefix', help='Prefix to output file, should include full path')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--vocab_size', '-vc', type=int, default=5000, help='size of vocabulary')
    parser.add_argument('--aa_mapping', '-am', type=int, default=20, help='How many options to encode amino acids. default: 20 (regular coding)')
    parser.add_argument('--min_freq', '-mf', type=int, default=2, help='How many times a token should be observed to be kept.default: 2')
    args = parser.parse_args()

    train_tokenizer(args.dataset_dir, args.col_name, args.vocab_size, args.min_freq, args.output_prefix, args.aa_mapping)