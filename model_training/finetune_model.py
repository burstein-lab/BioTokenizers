import torch
import time
import os
import argparse
import datasets
from pathlib import Path
import pickle
from data_processing.get_encoded_dataset import get_downstream_train_test, map_amino_acids
from utilities import clear_cache, load_tokenizer
from evaluation.eval_utilities import return_all_eval_metrics_dict
from transformers import TrainingArguments, Trainer, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaConfig, DataCollatorWithPadding, DataCollatorForTokenClassification, EsmForSequenceClassification, EsmTokenizer
from datasets import concatenate_datasets


EVAL_SIZE = 10000
N_SAMPLE = 10000000
MAIN_DIR = Path(__name__).parent.absolute()


def compute_metrics(pred):
    labels = pred.label_ids
    probs = torch.nn.functional.softmax(torch.from_numpy(pred.predictions), dim=-1)
    return return_all_eval_metrics_dict(probs, labels)


def tokenize_pair(sample, tokenizer, max_len):
    tok_res_1 = tokenizer(sample['prot_1'], truncation=True, max_length=int((max_len/2)-2), padding=False)
    sample['input_ids_1'] = tok_res_1['input_ids']
    sample['attention_mask_1'] = tok_res_1['attention_mask']

    tok_res_2 = tokenizer(sample['prot_2'], truncation=True, max_length=int((max_len/2)-2), padding=False)
    sample['input_ids_2'] = tok_res_2['input_ids']
    sample['attention_mask_2'] = tok_res_2['attention_mask']
    return sample


def prepare_pairwise_dataset(dataset, aa_mapping, tokenizer, max_len, ncpus):
    if aa_mapping != 20:
        dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_1'), num_proc=ncpus)
        dataset = dataset.map(lambda x: map_amino_acids(x, aa_mapping, 'prot_2'), num_proc=ncpus)

    dataset = dataset.map(lambda x: tokenize_pair(x, tokenizer, max_len), num_proc=ncpus, batched=True)
    dataset.set_format(type="torch", columns=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'])

    dataset_1 = dataset.map(lambda s: unite_pairwise(s, '1', '2', tokenizer.convert_tokens_to_ids("</s>")), batched=False, num_proc=ncpus, remove_columns=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'])
    dataset_2 = dataset.map(lambda s: unite_pairwise(s, '2', '1', tokenizer.convert_tokens_to_ids("</s>")), batched=False, num_proc=ncpus, remove_columns=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'])
    dataset = concatenate_datasets([dataset_1, dataset_2]).shuffle(seed=42)
    return dataset


def unite_pairwise(batch, col1, col2, sep_token_id):
    id_sep = torch.ones(1, dtype=torch.long) * sep_token_id
    attention_sep = torch.ones(1, dtype=torch.long)
    # Concatenate along dimension 2 (the last dimension)
    concatenated_ids = torch.cat((batch[f'input_ids_{col1}'], id_sep, batch[f'input_ids_{col2}'][1:]), dim=-1)
    concatenated_attention = torch.cat((batch[f'attention_mask_{col1}'], attention_sep, batch[f'attention_mask_{col2}'][1:]), dim=-1)
    batch['input_ids'] = concatenated_ids
    batch['attention_mask'] = concatenated_attention
    return batch


def main(args):
    print(args.save_prefix)
    device = torch.device(f"cuda" if torch.cuda.is_available() and args.device != -1 else "cpu")

    # load dataset train and test splits
    get_val = os.path.exists(os.path.join(args.dataset, 'validation'))
    train_dataset, test_dataset, eval_dataset = get_downstream_train_test(args.dataset, mapping_code=20 if args.task == 'pairwise' else args.aa_mapping, train_file_num=0, proc=args.ncpu, get_val=get_val)

    model_path = os.path.join(args.model_outdir, args.save_prefix)
    tokenizer = load_tokenizer(args.tokenizer_file, args.max_length)
    config = RobertaConfig.from_pretrained(args.input_model, num_labels=args.n_labels)
    clear_cache()

    # tokenizing train & test dataset
    if args.task == 'pairwise':
        train_dataset = prepare_pairwise_dataset(train_dataset, args.aa_mapping, tokenizer, args.max_length, 10)
        test_dataset = prepare_pairwise_dataset(test_dataset, args.aa_mapping, tokenizer, args.max_length, 10)
        eval_dataset = prepare_pairwise_dataset(eval_dataset, args.aa_mapping, tokenizer, args.max_length, 10)
    else:
        train_dataset = train_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu).shuffle(seed=42)
        test_dataset = test_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu)
        eval_dataset = eval_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu) if eval_dataset is not None else None
    data_collator = DataCollatorWithPadding(tokenizer)

    # initialize the model with the config
    model = RobertaForSequenceClassification.from_pretrained(args.input_model, config=config).to(device)
    eval_dataset = eval_dataset if eval_dataset is not None else test_dataset.shuffle(seed=42).select(range(min(EVAL_SIZE, len(test_dataset))))  # to make training quicker

    # remove other columns and set input_ids and attention_mask as tensors
    train_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])
    eval_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])

    # turn labels to long
    train_dataset = train_dataset.cast_column('label', datasets.Value("int32"))
    test_dataset = test_dataset.cast_column('label', datasets.Value("int32"))
    eval_dataset = eval_dataset.cast_column('label', datasets.Value("int32"))

    if args.train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.train_samples, len(train_dataset))))
        print(f"Sampling {args.train_samples} samples from the training dataset.")

    vocab_size = len(tokenizer)
    print(f'vocab_size: {vocab_size}')
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    # clear the cache
    clear_cache()

    # configure model output path
    os.makedirs(model_path, exist_ok=True)

    if args.freeze:  # Freeze encoder layers - do not want to overfit...
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=model_path,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        auto_find_batch_size=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_soothing,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=args.logging_interval,
        save_steps=args.save_interval,
        eval_steps=args.eval_steps,
        # for multiclass - f1_weighted since we use argmax. else, aupr is used to not focus on one threshold
        metric_for_best_model="eval_aupr" if args.n_labels == 2 else "eval_f1_weighted",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # train the model
    before = time.time()
    trainer.train()
    after = time.time()
    print(f"Training time took {(after-before)/60} minutes")

    trainer.save_model(model_path)

    os.makedirs(os.path.join(model_path, f'{args.epochs}_epochs'), exist_ok=True)
    os.system(f'rm -r {os.path.join(model_path, "checkpoint-*")} && '
              f'find {model_path}/ -maxdepth 1 -type f -exec mv {{}} {os.path.join(model_path, f"{args.epochs}_epochs/.")} \;')

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training LM for classification tasks')
    # training dataset
    parser.add_argument('--dataset', help='path to a directory with .csv train, test and test_sampled dataset')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--task', default='other', help='The finetuning task. Either pairwise or other. default:other')
    parser.add_argument('--aa_mapping', '-am', type=int, default=20, help='How many options to encode amino acids. default: 20 (regular coding)')
    parser.add_argument('--train_samples', '-ts', type=int, default=0, help='How much to sample from train dataset. default: 0 (All training data)')
    parser.add_argument('--input_model', help='path to a directory with the relevant pretrained model')
    parser.add_argument('--max-length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--model_outdir', default=os.path.join(MAIN_DIR, 'models/'), help='path to a directory to save model outputs')
    parser.add_argument('--tokenizer_file', help='path to tokenizer file')
    parser.add_argument('--save-prefix', default='finetuned_ProtBERTa', help='path prefix for saving models')

    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('--n_labels', type=int, default=2, help='Number of possible classes. Default: 2')
    parser.add_argument('--freeze', action='store_true', help="Freeze all params other than the classification head, so we won't overfit. default: False")

    # training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default: 64')
    parser.add_argument('-ga', '--gradient_accumulation', type=int, default=2, help='Gradient Accumulation. Default: 8')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of data epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('-wr', '--warmup_ratio', type=float, default=0.0, help='warmup_ratio')
    parser.add_argument('-ls', '--label_soothing', type=float, default=0.0, help='label_soothing')
    parser.add_argument('--save-interval', type=int, default=5000, help='number of step between data saving')
    parser.add_argument('--logging-interval', type=int, default=100, help='number of step between data logging')
    parser.add_argument('--eval_steps', type=int, default=1000, help='number of step between running model on eval_set')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use. Choose -1 for cpu. Default: -1')
    parser.set_defaults(freeze=False)
    args = parser.parse_args()

    main(args)
