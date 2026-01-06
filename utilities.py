import gc
import torch
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaForTokenClassification
from model_training.roberta_regression_model import RobertaForRegression # TODO add


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