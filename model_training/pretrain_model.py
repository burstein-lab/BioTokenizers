import torch
import os
import argparse
from pathlib import Path
from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from .get_encoded_dataset import get_train_test
from utilities import clear_cache, load_tokenizer
from train_tokenizer import train_tokenizer
from datasets import load_from_disk

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAIN_DIR = Path(__name__).parent.absolute()
CACHE_DIR = os.path.join(MAIN_DIR, "cache/")
PROT_SAMPLE = 15_000_000

# this code was inspired from : https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/


def main(args):
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
    print(args.save_prefix)
    print(torch.cuda.is_available())
    device = torch.device(f"cuda" if torch.cuda.is_available() and args.device != -1 else "cpu")
    print(f"Device is: {device}")

    # clear the cache
    clear_cache()

    train_file_num = 10 if args.debug else 0
    if args.pre_tokenized:  # loading pre-processed dataset from disk
        train_dataset = load_from_disk(os.path.join(args.dataset,  f'ProtBerta_{args.aa_mapping}', 'train'))
        test_dataset = load_from_disk(os.path.join(args.dataset,  f'ProtBerta_{args.aa_mapping}', 'eval'))
    else:
        train_dataset, test_dataset = get_train_test(args.dataset, mapping_code=args.aa_mapping, train_file_num=train_file_num, proc=args.ncpu, is_eval=True, prot_sample=PROT_SAMPLE)
    print('got dataset')

    # configure model output path
    model_path = os.path.join(args.model_path, args.save_prefix)
    os.makedirs(model_path, exist_ok=True)

    # load a tokenizer or train from scratch if needed
    if not os.path.exists(args.tokenizer_file):
        output_prefix = args.tokenizer_file.split(f"_prot_")[0]
        train_tokenizer(args.tokenizer_dataset, args.col_name, args.vocab_size, args.min_freq, output_prefix, args.aa_mapping)

    tokenizer = load_tokenizer(args.tokenizer_file, args.max_length)
    print('loaded tokenizer')
    clear_cache()

    # No need to tokenize as dataset is already tokenized
    if not args.pre_tokenized:
        train_dataset = train_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True, padding="max_length"), batched=True, keep_in_memory=False, num_proc=args.ncpu)
        test_dataset = test_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True, padding="max_length"), batched=True, keep_in_memory=False, num_proc=args.ncpu)
        print('tokenized data')

    vocab_size = len(tokenizer)
    print(f'vocab_size: {vocab_size}')
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    # clear the cache
    clear_cache()

    # initialize the model with the config
    n_hidden = 8 if args.lang == 'prot' else 8
    model_config = RobertaConfig(vocab_size=vocab_size, max_position_embeddings=args.max_length, num_hidden_layers=n_hidden)
    if args.model:
        model = RobertaForMaskedLM.from_pretrained(args.model).to(device)
    else:
        model = RobertaForMaskedLM(config=model_config).to(device)

    print("Model size: " + str(sum([p.numel() for p in model.parameters()])))

    # initialize data collector for LM batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.p
    )

    training_args = TrainingArguments(
        output_dir=model_path,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        auto_find_batch_size=False,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=64,
        logging_steps=args.logging_interval,
        eval_steps=1000,
        save_steps=args.save_interval,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    clear_cache()

    # train the model
    trainer.train()
    trainer.save_model()

    clear_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-Training ProtBERTa Models')
    # training dataset
    parser.add_argument('--dataset', default='/davidb/ellarannon/microbial_encoder/corpus/', help='path to a directory with .csv train and test dataset')
    parser.add_argument('--aa_mapping', '-am', type=int, default=20, help='How many options to encode amino acids. default: 20 (regular coding)')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--pre_tokenized', action='store_true', help='Choose this if the dataset is already tokenized. Default: False')

    # training tokenizer
    parser.add_argument('--tokenizer_dataset', help='path to a directory with dataset to train the the tokenizer')
    parser.add_argument('--vocab-size', type=int, default=38_000, help='vocabulary size')
    parser.add_argument('--min_freq', '-mf', type=int, default=75, help='How many times a token should be observed to be kept.default: 75')
    parser.add_argument('--tokenizer_file', help='path to tokenizer file, saves into it if doesnt exist')

    parser.add_argument('--model-path', default=os.path.join(MAIN_DIR, 'models/'), help='path to a directory to save model outputs')
    parser.add_argument('--model', default='', help='path to existing model we want to load. If empty, we initialize it. default: ''')
    parser.add_argument('--save-prefix', default='pretrained-ProtBERTa', help='path prefix for saving models')
    parser.add_argument('--max-length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('-p', type=float, default=0.15, help='masking rate')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    # training parameters
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--save-interval', type=int, default=1000, help='number of step between data saving')
    parser.add_argument('--logging-interval', type=int, default=1000, help='number of step between data logginh')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')
    parser.set_defaults(debug=False)
    parser.set_defaults(pre_tokenized=False)
    args = parser.parse_args()
    main(args)
