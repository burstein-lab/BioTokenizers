import dgeb
import re
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality
from transformers import RobertaModel, RobertaTokenizerFast, DefaultDataCollator, BatchEncoding
from typing import Dict, List, Literal, Optional
import torch
import logging
from functools import partial
import numpy as np
import tqdm as tqdm
from dgeb.eval_utils import pool
from tokenizers.processors import BertProcessing
from torch import Tensor
from datasets import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from data_processing.get_encoded_dataset import AA_MAPPING_DICT

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
REL_METRICS = ['bacarch_bigene_layer_4_f1', 'modac_paralogy_bigene_layer_4_recall_at_50', 'convergent_enzymes_classification_layer_4_f1', 'ec_classification_layer_4_f1', 'MIBIG_protein_classification_layer_4_f1', 'mopb_clustering_layer_4_v_measure', 'fefe_phylogeny_layer_4_top_corr', 'rpob_arch_phylogeny_layer_4_top_corr', 'rpob_bac_phylogeny_layer_4_top_corr', 'cyano_operonic_pair_layer_4_cos_sim_ap', 'ecoli_operonic_pair_layer_4_cos_sim_ap', 'vibrio_operonic_pair_layer_4_cos_sim_ap', 'arch_retrieval_layer_4_map_at_5', 'euk_retrieval_layer_4_map_at_5']
MODELS = ['ProtBERTa_2', 'ProtBERTa_4', 'ProtBERTa_8', 'ProtBERTa_12', 'ProtBERTa_20']
COLORS = {'ProtBERTa_2': '#E6AA61', 'ProtBERTa_4': '#e67961', 'ProtBERTa_8': '#ce4763', 'ProtBERTa_12': '#a3386f', 'ProtBERTa_20': '#672a6b'}


# Adapted from https://github.com/TattaBio/DGEB
class ProtBERTa(BioSeqTransformer):
    MODEL_NAMES = ['ProtBERTa_2', 'ProtBERTa_4', 'ProtBERTa_8', 'ProtBERTa_12', 'ProtBERTa_20']
    def __init__(
            self,
            model_name: str,
            tokenizer_path: str,
            layers: Optional[List[int] | Literal["mid"] | Literal["last"]] = None,
            devices: List[int] = [0],
            num_processes: int = 16,
            max_seq_length: int = 1024,
            l2_norm: bool = False,
            batch_size: int = 128,
            pool_type: str = "mean",
    ):
        super(BioSeqTransformer, self).__init__()

        self.id = self.__class__.__name__
        self.hf_name = model_name
        self.encoder = self._load_model(model_name)
        if not hasattr(self.encoder, "config"):
            raise ValueError(
                'The model from `self._load_model()` must have a "config" attribute.'
            )

        self.num_processes = num_processes
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pool_type = pool_type

        self.config = self.encoder.config
        self.tokenizer = self._get_tokenizer(tokenizer_path)
        self.aa_mapping = AA_MAPPING_DICT[int(tokenizer_path.split('_')[-1].strip('/'))]
        self.num_param = sum(p.numel() for p in self.encoder.parameters())
        self.data_collator = DefaultDataCollator()
        self.gpu_count = len(devices)
        self.l2_norm = l2_norm

        self.device = torch.device(
            f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
        )

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=devices)
        self.encoder.to(self.device)
        self.encoder.eval()

        mid_layer = self.num_layers // 2
        last_layer = self.num_layers - 1
        mid_layer_label = f"mid ({mid_layer})"
        last_layer_label = f"last ({self.num_layers - 1})"

        if layers is None:
            logger.debug(f"Using default layers: {mid_layer_label}, {last_layer_label}")
            self.layers = [mid_layer, last_layer]
            self.layer_labels = [mid_layer_label, last_layer_label]
        elif layers == "mid":
            self.layers = [mid_layer]
            self.layer_labels = [mid_layer_label]
        elif layers == "last":
            self.layers = [last_layer]
            self.layer_labels = [last_layer_label]
        else:
            self.layers = layers
            self.layer_labels = [str(layer) for layer in layers]
    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

    def _load_model(self, model_name):
        return RobertaModel.from_pretrained(model_name, trust_remote_code=True)

    def _get_tokenizer(self, tokenizer_path):
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, trust_remote_code=True, max_len=self.max_seq_length-2)
        tokenizer.post_processor = BertProcessing(sep=("</s>", tokenizer.encode("</s>")[0]), cls=("<s>", tokenizer.encode("<s>")[0]))
        return tokenizer

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        """Returns the output embedding for the given batch with shape [batch, num_layers, D]."""
        outputs = self.encoder(**batch_dict, output_hidden_states=True)
        embeds = [outputs.hidden_states[layer] for layer in self.layers]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        return embeds

    def _tokenize_func(
            self, tokenizer, examples: Dict[str, List], max_seq_length: int
    ) -> BatchEncoding:
        examples["input_seqs"] = [np.char.translate(seq, self.aa_mapping).item() for seq in examples["input_seqs"]]
        batch_dict = tokenizer(
            examples["input_seqs"],
            max_length=max_seq_length,
            padding=True,
            truncation=True,
        )
        return batch_dict

    @torch.no_grad()
    def encode(self, sequences, **kwargs) -> np.ndarray:
        """Returns a list of embeddings for the given sequences.
        Args:
            sequences (`List[str]`): List of sequences to encode
        Returns:
            `np.ndarray`: Embeddings for the given sequences of shape [num_sequences, num_layers, embedding_dim].
        """
        dataset = Dataset.from_dict({"input_seqs": sequences})
        dataset.set_transform(
            partial(
                self._tokenize_func, self.tokenizer, max_seq_length=self.max_seq_length
            )
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_processes,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        if max(self.layers) >= self.num_layers:
            raise ValueError(
                f"Layer {max(self.layers)} is not available in the model. Choose a layer between 0 and {self.num_layers - 1}"
            )

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(
                data_loader, desc="encoding", mininterval=10, disable=len(sequences) < 128
        ):
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            embeds = self._encode_single_batch(batch_dict)

            if self.l2_norm:
                embeds = F.normalize(embeds, p=2, dim=-1)
            encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


def process_DGEB_results(res_dir):
    res_dict = defaultdict(dict)
    for entry in os.listdir(res_dir):
        if not os.path.isdir(os.path.join(res_dir, entry)):
            continue
        model_name = entry
        dir_name = os.path.join(res_dir, entry)
        for filename in os.listdir(dir_name):
            task_name = filename.replace('.json', '')
            filepath = os.path.join(dir_name, filename)
            with open(filepath, 'r') as fin:
                results = json.load(fin)["results"]
                all_layers = [r['layer_number'] for r in results]
                min_layer, max_layer = min(all_layers), max(all_layers)
                layer_mapping = {min_layer: 4, max_layer: 7}
                for res in results:
                    layer = res['layer_number']
                    metrics = res['metrics']
                    curr_dict = {f'{task_name}_layer_{layer_mapping[layer]}_{d["display_name"]}': d["value"] for d in metrics}
                    res_dict[model_name].update(curr_dict)
    df = pd.DataFrame(res_dict)
    return df[[model for model in MODELS if model in df.columns]]  # ordering results by models


def plot_selected_metrics(res, output_dir, metric_lst=REL_METRICS, prefix='DGEB_selected_metrics'):
    rel_metrics = [metric.replace('layer_4_', '') for metric in metric_lst]
    for metric in metric_lst:
        res.loc[metric.replace('layer_4_', '')] = res.loc[[metric, metric.replace('layer_4', 'layer_7')], :].max()
    res.loc[rel_metrics].plot.bar(color=COLORS, alpha=0.9)
    plt.legend(fontsize=9)
    print(pd.DataFrame(res.loc[rel_metrics].mean()).T.round(3).iloc[:, :])
    plt.savefig(os.path.join(output_dir, f'{prefix}.svg'), dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Running DGEB tasks on ProtBERTa')
    parser.add_argument('--model_path', help='path to the pre-trained model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--output_dir', help='path to the directory to save the results')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    args = parser.parse_args()

    tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    evaluation = dgeb.DGEB(tasks=tasks)
    for aa_mapping in (2, 4, 8, 12, 20):
        tokenizer_path = args.tokenizer_prefix + str(aa_mapping)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', args.model_path)
        model = ProtBERTa(model_name=model_path, tokenizer_path=tokenizer_path)
        output_dir = os.path.join(args.output_dir, f'ProtBERTa_{aa_mapping}')
        os.makedirs(output_dir, exist_ok=True)
        evaluation.run(model, output_folder=output_dir)

    res = process_DGEB_results(args.output_dir)
    plot_selected_metrics(res, args.output_dir)
