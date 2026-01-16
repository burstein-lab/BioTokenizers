# Written with the help of Claude.ai and ChatGPT

import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, PreTrainedModel, AutoConfig
from model_training.roberta_regression_model import RobertaForRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np
from data_processing.get_encoded_dataset import get_downstream_train_test
import time
import argparse
from utilities import load_tokenizer, clear_cache
import random
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


EVAL_SIZE = 10000
MAIN_DIR = Path(__name__).parent.absolute()


def compute_metrics(eval_pred):
    """
        eval_pred: Tuple of (predictions, labels)
    """
    predictions, labels = eval_pred

    # If predictions have multiple dimensions, flatten them
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    if not isinstance(labels, list) and len(labels.shape) > 1:
        labels = labels.flatten()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def main(args):
    print(args.save_prefix)
    print(torch.cuda.is_available(), flush=True)
    device = torch.device(f"cuda" if torch.cuda.is_available() and args.device != -1 else "cpu")

    get_val = os.path.exists(os.path.join(args.dataset, 'validation'))
    train_dataset, test_dataset, eval_dataset = get_downstream_train_test(args.dataset, mapping_code=args.aa_mapping, train_file_num=0, proc=args.ncpu, get_val=get_val)

    model_path = os.path.join(args.model_outdir, args.save_prefix)
    tokenizer = load_tokenizer(args.tokenizer_file, args.max_length)
    clear_cache()

    # tokenizing train & test dataset
    train_dataset = train_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu).shuffle(seed=42)
    test_dataset = test_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu)
    eval_dataset = eval_dataset.map(lambda e: tokenizer(e[args.col_name], truncation=True), batched=True, keep_in_memory=False, num_proc=args.ncpu) if eval_dataset is not None else test_dataset.shuffle(seed=42).select(range(min(EVAL_SIZE, len(test_dataset))))

    data_collator = DataCollatorWithPadding(tokenizer)

    # initialize the model with the config
    config = AutoConfig.from_pretrained(args.input_model)
    config.update({"dropout": args.dropout, "hidden_dim": args.hidden_dim, "pooling_method": args.pooling, "num_attention_heads": args.n_attention_heads, "loss": args.loss, "model_type": "roberta_regression"})
    model = RobertaForRegression(config).to(device)

    # remove other columns and set input_ids and attention_mask as tensors
    train_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])
    eval_dataset.set_format(type="torch", columns=["input_ids", 'label', "attention_mask"])

    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    # clear the cache
    clear_cache()

    # configure model output path
    os.makedirs(model_path, exist_ok=True)

    if args.freeze:  # Freeze encoder layers - do not want to overfit...
        total_layers = len(model.roberta.roberta.encoder.layer)
        for i in range(total_layers):
            for param in model.roberta.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=model_path,
        fp16=torch.cuda.is_available(),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_interval,
        eval_steps=args.eval_steps,
        save_steps=args.save_interval,
        max_grad_norm=1.0,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model='mae',
        greater_is_better=False,
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
    os.system(f'find {model_path}/ -maxdepth 1 -type f -exec mv {{}} {os.path.join(model_path, f"{args.epochs}_epochs/")} \; && '
              f'rm -r {os.path.join(model_path, "checkpoint-*")}')
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning LM on regression tasks')
    # training dataset - '../'
    parser.add_argument('--dataset', help='path to a directory with .csv train, test and test_sampled dataset')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--aa_mapping', '-am', type=int, default=20, help='Size of the chosen amino acid alphabet. default: 20 (regular coding)')
    parser.add_argument('--input_model', help='path to a directory with the relevant pretrained model')
    parser.add_argument('--max-length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--model_outdir', default=os.path.join(MAIN_DIR, 'models/'), help='path to a directory to save model outputs')
    parser.add_argument('--tokenizer_file', help='path to tokenizer file')
    parser.add_argument('--save-prefix', help='path prefix for saving models', default='finetuned_regrssion_ProtBERTa')

    #finetuning parameters
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='size of hidden dim if we want regressor to have hidden layer, None for no hidden layer. Default:256')
    parser.add_argument('--pooling', default='mean', help='which pooling method we want. can be cls, mean, max, attention, multihead_attention. default:mean')
    parser.add_argument('--n_attention_heads', type=int, default=8, help='Number of attention heads if we use multihead_attention pooling. Dimension size should be divisble by it. Default: 8')
    parser.add_argument('--loss', default='mae', help='which loss function method we want. can be mae, mse, huber. default:mae')

    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('--freeze', action='store_true', help="Freeze all params other than the classification head, so we won't overfit. default: False")

    # training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default: 64')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='number of data epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate. default: 2e-5')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01, help='Weight decay. Default: 0.01')
    parser.add_argument('--warmup_steps', '-ws', type=int, default=500, help='Warmup steps. Default: 500')
    parser.add_argument('--save-interval', type=int, default=1000, help='number of step between data saving')
    parser.add_argument('--logging-interval', type=int, default=100, help='number of step between data logging')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of step between running model on eval_set')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use')

    parser.set_defaults(freeze=False)
    args = parser.parse_args()

    main(args)
