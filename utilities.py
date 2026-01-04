import gc
import torch
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def load_tokenizer(tokenizer_file, max_length):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_file, max_len=max_length-2, add_prefix_space=False, truncation=True, pad_to_max_length=True, padding="max_length")
    tokenizer.post_processor = BertProcessing(sep=("</s>", tokenizer.encode("</s>")[0]), cls=("<s>", tokenizer.encode("<s>")[0]))
    return tokenizer