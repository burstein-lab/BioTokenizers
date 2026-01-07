import gc
import os
import pickle
import torch
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaForTokenClassification, DataCollatorWithPadding
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model_training.roberta_regression_model import RobertaForRegression
from model_training.roberta_with_advanced_pooling import mean_pooling
from data_processing.get_encoded_dataset import map_amino_acids


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer(tokenizer_file, max_length):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_file, max_len=max_length-2, add_prefix_space=False, truncation=True, pad_to_max_length=True, padding="max_length")
    tokenizer.post_processor = BertProcessing(sep=("</s>", tokenizer.encode("</s>")[0]), cls=("<s>", tokenizer.encode("<s>")[0]))
    return tokenizer


def load_model(model_path, device_num, model_type=None, n_labels=2):
    config = RobertaConfig.from_pretrained(model_path, num_labels=n_labels)
    if model_type == 'SeqClass':
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    elif model_type == 'TokenClass':
        model = RobertaForTokenClassification.from_pretrained(model_path, config=config)
    elif model_type == 'regression':
        model = RobertaForRegression.from_pretrained(model_path)
    else:
        model = RobertaModel.from_pretrained(model_path, output_attentions=True, output_hidden_states=True)
    device = torch.device(f"cuda" if torch.cuda.is_available() and device_num != -1 else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def load_model_and_tokenizer(model_path, tokenizer_path, device_num, max_length=1026, model_type=None, n_labels=2):
    model, device = load_model(model_path, device_num, model_type=model_type, n_labels=n_labels)
    tokenizer = load_tokenizer(tokenizer_path, max_length)
    return model, tokenizer, device


@torch.inference_mode()
def run_model_in_batches(model, tokenizer, dataset, device, batch_size=128, col='prot', ncpus=10, ret_logits=True):
    all_model_res = []
    if "label" in dataset.column_names and "labels" not in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")
    if 'input_ids' not in dataset.column_names:
        dataset = dataset.map(lambda e: tokenizer(e[col], truncation=True), batched=True, keep_in_memory=False, num_proc=ncpus)
    # Remove the unwanted columns
    columns_to_remove = [col for col in dataset.column_names if col not in ['labels', 'input_ids', 'attention_mask']]
    dataset = dataset.remove_columns(columns_to_remove)
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    for batch in tqdm(eval_dataloader, desc='Evaluating Model'):
        clear_cache()
        input_ids, attention_mask = torch.tensor(batch['input_ids']).detach().to(device), torch.tensor(batch['attention_mask']).detach().to(device)
        with torch.cuda.amp.autocast(enabled=True):
            if ret_logits:
                res = model(input_ids, attention_mask=attention_mask)['logits'].to('cpu')
            else:  # getting pooled embeddings
                res = model(input_ids, attention_mask=attention_mask).last_hidden_state
                res = mean_pooling(res, attention_mask).to('cpu')
            all_model_res.append(res)

    return torch.cat(all_model_res, dim=0)


def create_model_embeddings(dataset, tokenizer_file, model_path, aa_mapping, task, out_dir, proc=10, col='prot', max_length=1026, device_num=-1, batch_size=32):
    model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_file, device_num, max_length=max_length)
    for op in ['train', 'test']:
        output_file = os.path.join(out_dir, f'{task}_ProtBERTa_{aa_mapping}_{op}_embs.pkl')
        if os.path.exists(output_file):
            continue
        else:
            print(f'Calculating embeddings for ProtBERTa{aa_mapping}', flush=True)
            ds = dataset[op].map(lambda x: map_amino_acids(x, int(aa_mapping)), num_proc=proc) if int(aa_mapping) != 20 else dataset[op]
            res = run_model_in_batches(model, tokenizer, ds, device, batch_size=batch_size, col=col, ncpus=proc, ret_logits=False)

            with open(output_file, 'wb') as fout:
                pickle.dump(res, fout)