import os
import re
from collections import defaultdict
import pickle
import glob
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Literal, Optional
from collections import Counter
from evaluation.eval_utilities import COLORS, plot_all_metric_results, get_results_dict, plot_binary_AUPR_AUROC, finish_AUPR_AUROC_figure
from utilities import create_model_embeddings


AA_MAPPINGS = [2, 4, 8, 12, 20]
CACHE_DIR = os.path.join(Path(__name__).parent.absolute(), "cache")


class EmbeddingClassifier:
    """
    Classifier for embeddings using KNN.
    Written with the help of Claude.ai

    Args:
        k: Number of neighbors for KNN (default: 5)
        distance_metric: 'cosine' or 'euclidean' (default: 'cosine')
        chunk_size: Number of queries to process at once (default: 100)
    """

    def __init__(
            self,
            k: int = 5,
            distance_metric: Literal['cosine', 'euclidean'] = 'cosine',
            chunk_size: int = 100
    ):
        self.k = k
        self.distance_metric = distance_metric
        self.chunk_size = chunk_size
        self.train_embeddings = None
        self.train_labels = None
        self.centroids = None
        self.classes = None

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Fit the classifier with training embeddings and labels.

        Args:
            embeddings: Training embeddings of shape (n_samples, embedding_dim)
            labels: Training labels of shape (n_samples,)
        """
        self.train_embeddings = embeddings
        self.train_labels = labels
        self.classes = torch.unique(labels)


    def _compute_distances(
            self,
            query: torch.Tensor,
            reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query and reference embeddings using chunked processing.

        Args:
            query: Query embeddings of shape (n_queries, embedding_dim)
            reference: Reference embeddings of shape (n_ref, embedding_dim)

        Returns:
            Distance matrix of shape (n_queries, n_ref)
        """
        n_queries = query.shape[0]

        # Pre-normalize once if using cosine distance
        if self.distance_metric == 'cosine':
            query = F.normalize(query, p=2, dim=1)
            reference = F.normalize(reference, p=2, dim=1)
        elif self.distance_metric == 'euclidean':
            reference = reference.unsqueeze(0)  # (1, n_ref, dim)

        # Process in chunks to reduce memory usage
        distances = []
        for i in range(0, n_queries, self.chunk_size):
            end_i = min(i + self.chunk_size, n_queries)
            query_chunk = query[i:end_i]

            if self.distance_metric == 'cosine':
                # Cosine distance = 1 - cosine similarity
                # Use @ operator instead of torch.mm for better memory efficiency
                similarities = query_chunk @ reference.t()
                chunk_distances = 1 - similarities

            elif self.distance_metric == 'euclidean':
                # Euclidean distance
                # Expand dimensions for broadcasting
                query_chunk = query_chunk.unsqueeze(1)  # (n_queries, 1, dim)
                chunk_distances = torch.sqrt(((query_chunk - reference) ** 2).sum(dim=2))

            distances.append(chunk_distances)
        return torch.cat(distances, dim=0)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for query embeddings.

        Args:
            embeddings: Query embeddings of shape (n_samples, embedding_dim)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        distances = self._compute_distances(embeddings, self.train_embeddings)

        # Get k nearest neighbors
        _, indices = torch.topk(distances, k=self.k, largest=False, dim=1)

        # Get labels of k nearest neighbors
        neighbor_labels = self.train_labels[indices]  # (n_samples, k)

        # Majority vote for each query
        predictions = []
        for i in range(embeddings.shape[0]):
            labels = neighbor_labels[i].cpu().numpy()
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)

        return torch.tensor(predictions, device=embeddings.device)

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities based on distances.

        Args:
            embeddings: Query embeddings of shape (n_samples, embedding_dim)

        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        distances = self._compute_distances(embeddings, self.train_embeddings)
        _, indices = torch.topk(distances, k=self.k, largest=False, dim=1)
        neighbor_labels = self.train_labels[indices]

        # Count votes for each class
        n_samples = embeddings.shape[0]
        n_classes = len(self.classes)
        probs = torch.zeros(n_samples, n_classes, device=embeddings.device)

        for i in range(n_samples):
            for j, cls in enumerate(self.classes):
                probs[i, j] = (neighbor_labels[i] == cls).float().mean()

        return probs


def get_train_test_embeddings(data_dir, emb_dir, tokenizer_path, model_path, task, aa_mapping, proc=10, col='prot', max_length=1026, device=-1, batch_size=32):
    file_pattern = {'train': glob.glob(os.path.join(data_dir, 'train', '*.csv')), "test": glob.glob(os.path.join(data_dir, 'test', '*.csv'))}
    dataset = load_dataset('csv', data_files=file_pattern, cache_dir=CACHE_DIR)
    dataset = dataset.class_encode_column('label')
    train_labels = torch.tensor(dataset['train']['label'])
    test_labels = list(dataset['test']['label'])

    train_emb_file = os.path.join(emb_dir, f'{task}_ProtBERTa_{aa_mapping}_train_embs.pkl')
    test_emb_file = os.path.join(emb_dir, f'{task}_ProtBERTa_{aa_mapping}_test_embs.pkl')
    if not os.path.exists(train_emb_file) or not os.path.exists(test_emb_file):
        create_model_embeddings(dataset, tokenizer_path, model_path, aa_mapping, task, emb_dir, proc=proc, col=col, max_length=max_length, device_num=device, batch_size=batch_size)

    with open(train_emb_file, 'rb') as fin:
        train_embeddings = pickle.load(fin)
        train_embeddings = torch.tensor(train_embeddings)

    with open(test_emb_file, 'rb') as fin:
        test_embeddings = pickle.load(fin)
        test_embeddings = torch.tensor(test_embeddings)

    return train_embeddings, train_labels, test_embeddings, test_labels


def get_zero_shot_performance_per_model(data_dir, emb_dir, model_path, tokenizer_path, aa_mapping, task, k=5, distance_metric='cosine', proc=10, col='prot', max_length=1026, device=-1, batch_size=32):
    model_name = f'ProtBERTa_{aa_mapping}'
    train_embeddings, train_labels, test_embeddings, test_labels = get_train_test_embeddings(data_dir, emb_dir, tokenizer_path, model_path, task, aa_mapping, proc=proc, col=col, max_length=max_length, device=device, batch_size=batch_size)

    classifier = EmbeddingClassifier(k=k, distance_metric=distance_metric)
    classifier.fit(train_embeddings,  train_labels)

    # Predict on test set
    probs = classifier.predict_proba(test_embeddings)
    res = get_results_dict(probs, test_labels, n_labels=len(classifier.classes), is_probs=True)
    res['Model'] = model_name
    res['probs'] = probs

    return res, test_labels


def get_zero_shot_performance(data_dir, emb_dir, out_dir, model_path, tokenizer_prefix, task, k=5, distance_metric='cosine', proc=10, metric='weighted', col='prot', max_length=1026, device=-1, batch_size=32):
    all_res = []
    f_knn, axes_knn = plt.subplots(1, 2, figsize=(10, 5))
    for aa_mapping in AA_MAPPINGS:
        model_name = f'ProtBERTa_{aa_mapping}'
        print(f'Calculating zero-shot performance for {model_name}', flush=True)
        model_path = re.sub('ProtBERTa_(20|12|4|8|2)', f'ProtBERTa_{aa_mapping}', model_path)
        tokenizer_path = tokenizer_prefix + str(aa_mapping)
        res_knn, labels = get_zero_shot_performance_per_model(data_dir, emb_dir, model_path, tokenizer_path, aa_mapping, task, k=k, distance_metric=distance_metric, proc=proc, col=col, max_length=max_length, device=device, batch_size=batch_size)
        plot_binary_AUPR_AUROC(labels, res_knn.pop('probs'), f'ProtBERTa_{aa_mapping}', COLORS[f'ProtBERTa_{aa_mapping}'], axes_knn)
        all_res.append(res_knn)

    finish_AUPR_AUROC_figure(f_knn, axes_knn, os.path.join(out_dir, f'{task}_ProtBERTa_knn_{k}_zeroshot_AUROC_AUPR.svg'))
    output_file = os.path.join(out_dir, f'{task}_ProtBERTa_knn_{k}_zeroshot_res_{metric}.svg')
    df = pd.DataFrame(all_res).set_index('Model').dropna(how='all', axis=1)  # remove columns with all NaN values
    df.to_pickle(output_file.replace('.svg', '.pkl'))
    plot_all_metric_results(df, output_file, metric, colors=COLORS)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Zero-Shot classification using ProtBERTa pre-trained embeddings')
    parser.add_argument('--data_dir', help='path to a directory with .csv train and test dataset')
    parser.add_argument('--emb_dir', help='path to a directory to save pre-trained embeddings files')
    parser.add_argument('--out_dir', help='path to a directory to save results')
    parser.add_argument('--model_path', help='path to the Pre-trained model. Should contain ProtBERTa_X in the title where X is the aa_mapping (alphabet size)')
    parser.add_argument('--tokenizer_prefix', help='Prefix path to tokenizer files, such that the full path is tokenizer_prefix + aa_mapping (alphabet size)')
    parser.add_argument('--task', help='Task name, used for saving the embedding and results files')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN (default: 5)')
    parser.add_argument('--distance_metric', help='Distance metric to use (cosine or euclidean). default: cosine', default='cosine')
    parser.add_argument('--ncpu', type=int, default=10, help='number of cpus')
    parser.add_argument('--metric', type=str, help='Metric to calculate (micro, macro, weighted). default: weighted', default='weighted')
    parser.add_argument('--col_name', '-col', type=str, default='prot', help='Column name for protein sequences. default: prot')
    parser.add_argument('--max_length', type=int, default=1026, help='maximal sequence length')
    parser.add_argument('--device', type=int, default=-1, help='compute device to use, -1 for cpu. default: -1')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size. Default: 32')
    args = parser.parse_args()

    get_zero_shot_performance(args.data_dir, args.emb_dir, args.out_dir, args.model_path, args.tokenizer_prefix, args.task, k=args.k,
                                  distance_metric=args.distance_metric, proc=args.ncpu, metric=args.metric, col=args.col_name, max_length=args.max_length,
                                  device=args.device, batch_size=args.batch_size)



