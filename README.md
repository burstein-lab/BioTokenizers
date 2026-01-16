# BioTokenizers: Optimizing Protein Sequence Tokenization: Evaluating Reduced Amino Acid Alphabets for Efficient and Accurate NLP Applications

## Summary

Protein language models (pLMs) typically tokenize sequences at the single-amino-acid level using a 20-residue alphabet, resulting in long input sequences and high computational cost. Sub-word tokenization methods such as Byte Pair Encoding (BPE) can reduce sequence length but are limited by the sparsity of long patterns in the standard amino acid alphabet. Reduced amino acid alphabets, which group residues by physicochemical properties, offer a potential solution.

This repository focuses on the combined use of reduced amino acid alphabets and BPE tokenization in protein language models. It consists of code to pre-train RoBERTa-based pLMs using multiple reduced alphabets and evaluate their performance and runtime across diverse downstream tasks. 


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/burstein-lab/BioTokenizers.git
   cd BioTokenizers
   ```

2. Create a virtual environment with python3.11:
   ```
   python3 -m venv venv
   source ./venv/bin/activate
   pip install -r requirements.txt
   ```

## Getting the tokenizers and models
Downloading the trained tokenizers and pre-trained ProtBERTa models from the Zenodo database.

To get the trained tokenizers, use the following commands:
```
wget https://zenodo.org/record/18256943/files/ProtBERTa_tokenizers.tar.gz?download=1
tar -zxvf ProtBERTa_tokenizers.tar.gz
rm ProtBERTa_tokenizers.tar.gz
```
In order to get the pre-trained ProtBERTa models, use the command line:
```
wget https://zenodo.org/record/18257091/files/ProtBERTa_models.tar.gz?download=1
tar -zxvf ProtBERTa_models.tar.gz
rm ProtBERTa_models.tar.gz
```
## Usage

The main scripts in this repository are:

### 1. Train a Tokenizer

**Module:** `model_training.train_tokenizer`

This script trains a Byte Pair Encoding (BPE) tokenizer from scratch on a protein sequence corpus.

```
python -m model_training.train_tokenizer --dataset_dir <path_to_data_dir> --output_prefix <prefix_to_tokenizer_output> [OPTIONS]

Options:
--col_name, -col              Name of the column containing protein sequences  (default: 'prot').
--vocab_size, -vc             Size of the tokenizer vocabulary (default: 5000).
--aa_mapping, -am             Size of the chosen amino acid alphabet (e.g., 20 for standard amino acid encoding, default: 20).
--min_freq, -mf               Minimum frequency a token must appear to be included in the vocabulary (default: 2).
```

### 2. Pretrain a Protein Language Model (ProtBERTa)

**Module:** `model_training.pretrain_model`

This script pretrains a ProtBERTa-style transformer language model on protein sequences using masked language modeling. It can optionally train a tokenizer or load an existing one.

```
python -m model_training.pretrain_model --dataset <path_to_data_dir> --tokenizer_file <path_to_tokenizer> [OPTIONS]

Options:
--col_name, -col              Name of the column containing protein sequences  (default: 'prot').
--aa_mapping, -am             Size of the chosen amino acid alphabet (e.g., 20 for standard amino acid encoding, default: 20).
--pre_tokenized               Set this flag if the dataset is already tokenized.
--same_dir                    Set this flag if all dataset files are located in a single directory rather than subdirectories.
--debug                       Enable debug mode, training the model on a subset of the data.

# Tokenizer Options
--tokenizer_dataset           Path to a dataset directory used to train the tokenizer if tokenizer_file doesn't exist.
--vocab_size, -vc             Size of the tokenizer vocabulary, only used for training a new tokenizer (default: 5000).
--min_freq, -mf               Minimum frequency a token must appear to be included in the vocabulary (default: 2).

# Model Configuration
--model_outdir                Directory where model checkpoints and outputs will be saved (default: ./models/).
--model                       Path to an existing pretrained model to load. If not supplied, a new model is initialized.
--save-prefix                 Prefix used when saving model checkpoints (default: 'pretrained-ProtBERTa').
--max-length                  Maximum input sequence length (default: 1026).
-p                            Masking probability for masked language modeling (default: 0.15).
--n_hidden                    Number of transformer layers (default: 8).
--ncpu                        Number of CPU workers used for data loading (default: 10).

# Training Parameters
--batch_size, -b              Training batch size (default: 64).
--gradient_accumulation, -ga  Number of gradient accumulation steps (default: 8).
--epochs, -e                  Number of training epochs (default: 5).
--save-interval               Number of steps between saving model checkpoints (default: 1000).
--logging-interval            Number of steps between logging training metrics (default: 1000).
--eval_steps                  Number of steps between evaluations on the validation set (default: 1000).
--device                      Compute device to use (-1 for CPU, otherwise GPU index, default: -1).
```

### 3. Finetune a Pretrained Model for a Classification Task

**Module:** `model_training.finetune_model`

This script finetunes a pretrained language model on downstream classification tasks, such as binary or multi-class protein classification.
```
python -m model_training.finetune_model --dataset <path_to_data_dir> --tokenizer_file <path_to_tokenizer> --input_model <path_to_pretrained_model> [OPTIONS]
Options:
--col_name, -col              Name of the column containing protein sequences  (default: 'prot').
--aa_mapping, -am             Size of the chosen amino acid alphabet (e.g., 20 for standard amino acid encoding, default: 20).
--is_pairwise                 Set this flag to finetune the model on a pairwise classification task.
--train_samples, -ts          Number of training samples to use. Choose 0 to use the full training dataset (default: 0) .
--max-length                  Maximum input sequence length (default: 1026).

# Model and Output Options
--model_outdir                Directory where finetuned models and outputs will be saved (default: ./models/).
--save-prefix                 Prefix used when saving finetuned model checkpoints (default: finetuned_ProtBERTa).
--n_labels                    Number of target classes (default: 2).
--freeze                      Freeze all model parameters except the classification head.
--ncpu                        Number of CPU workers used for data loading (default: 10).

# Training Parameters
--batch_size, -b              Training batch size (default: 64).
--gradient_accumulation, -ga  Number of gradient accumulation steps (default: 2).
--epochs, -e                  Number of finetuning epochs (default: 10).
--learning_rate, -lr          Learning rate (default: 5e-5).
--weight_decay, -wd           Weight decay coefficient (default: 0.0).
--warmup_ratio, -wr           Warmup ratio for the learning rate scheduler (default: 0.0).
--label_soothing, -ls         Label smoothing factor (default: 0.0).
--save-interval               Number of steps between saving model checkpoints (default: 5000).
--logging-interval            Number of steps between logging training metrics (default: 100).
--eval_steps                  Number of steps between evaluations on the validation set (default: 1000).
--device                      Compute device to use (-1 for CPU, otherwise GPU index, default: -1).
```

### 4. Finetune a Pretrained Model for a Regression Task

**Module:** `model_training.train_roberta_regression`
Finetune a pretrained ProtBERTa on downstream regression tasks.

```
python -m model_training.train_roberta_regression --dataset <path_to_data_dir> --tokenizer_file <path_to_tokenizer> --input_model <path_to_pretrained_model> [options]

Options:
--col_name, -col              Name of the column containing protein sequences  (default: 'prot').
--aa_mapping, -am             Size of the chosen amino acid alphabet (e.g., 20 for standard amino acid encoding, default: 20).
--max-length                  Maximum input sequence length (default: 1026).

# Model and Output Options
--model_outdir                Directory where finetuned models and outputs will be saved (default: ./models/).
--save-prefix                 Prefix for saving finetuned regression models (default: finetuned_regrssion_ProtBERTa).
--pooling                     Pooling strategy for sequence representations. Options: 'cls', 'mean', 'max', 'attention', 'multihead_attention' (default: mean).
--n_attention_heads           Number of attention heads for the multihead_attention pooling (default: 8).
--hidden_dim                  Size of the hidden layer in the regression head. Set to None to disable the hidden layer (default: 256).
--loss                        Regression loss function. Options: 'mae', 'mse', 'huber' (default: 'mae').
--dropout                     Dropout rate for the task head (default: 0.15).

# Training Parameters
--freeze                      Freeze all pretrained model parameters and train only the regression head.
--ncpu                        Number of CPU workers used for data loading (default: 10).
--batch_size, -b              Training batch size (default: 64).
--epochs, -e                  Number of finetuning epochs (default: 15).
--learning_rate, -lr          Learning rate (default: 2e-5).
--weight_decay, -wd           Weight decay coefficient (default: 0.01).
--warmup_steps, -ws           Warmup steps for the learning rate scheduler (default: 500).
--save-interval               Number of steps between saving model checkpoints (default: 5000).
--logging-interval            Number of steps between logging training metrics (default: 100).
--eval_steps                  Number of steps between evaluations on the validation set (default: 500).
--device                      Compute device to use (-1 for CPU, otherwise GPU index, default: -1).
```

### 5. Evaluate Finetuned Models

**Module:** `evaluation.eval_model`

Evaluate finetuned classification or regression models and generate performance plots.

```
python -m evaluation.eval_model --model_path <path_to_finetuned_model> --tokenizer_prefix <prefix_path_to_tokenizer_file> --dataset <path_to_data_dir> --output_file <path_to_output_figure> [options]

**Evaluation Options**
--metric                      Metric aggregation method for classification ('micro', 'macro', or 'weighted', default: 'weighted').
--col_name, -col              Name of the column containing protein sequences  (default: 'prot').
--n_labels                    Number of target classes (default: 2).
--max-length                  Maximum input sequence length (default: 1026).
--ncpu                        Number of CPU workers used for data loading (default: 10).
--batch_size, -b              Training batch size (default: 64).
--device                      Compute device to use (-1 for CPU, otherwise GPU index, default: -1).
--is_pairwise                 Evaluate a pairwise classification model.
--is_regression               Evaluate a regression model.
```
#### Cleveland Plot for Regression Results

To generate Cleveland-style plots from existing regression results:

```
python -m evaluation.eval_model --file_lst results_task1.pkl results_task2.pkl --titles "Task 1" "Task 2" --metric <regression_metric> --output_file <path_to_output_figure>
--metric                      Regression metric to plot ('mse', 'rmse', or 'mae').
--file_lst                    List of pickle files containing saved regression evaluation results.
--titles                      Titles corresponding to each regression task (must match the order of 'file_lst').
```


## Contact
For questions about this repository, please contact us:
Ella Rannon: [ellarannon@mail.tau.ac.il](mailto:ellarannon@mail.tau.ac.il)
David Burstein: [davidbur@tauex.tau.ac.il](mailto:davidbur@tauex.tau.ac.il)
