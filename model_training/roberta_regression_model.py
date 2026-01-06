from roberta_with_advanced_pooling import RobertaWithAdvancedPooling
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig
from safetensors.torch import save_file, load_file
import os


class RobertaForRegression(PreTrainedModel):
    """
    RoBERTa model for regression tasks using mean pooling.

    Args:
        model_name (str): Name of the pre-trained RoBERTa model
        dropout (float): Dropout rate for the classifier head
        hidden_dim (Optional[int]): Hidden dimension for additional layers
    """
    config_class = AutoConfig  # tells HF how to handle configs

    def __init__(self, config):
        super().__init__(config)

        model_name = getattr(config, "base_model_name_or_path", None)
        if model_name is None:
            model_name = getattr(config, "_name_or_path", "roberta-base")
        initialize_roberta = getattr(config, "initialize_roberta", True)
        dropout = getattr(config, "dropout", 0.1)
        hidden_dim = getattr(config, "hidden_dim", None)
        pooling_method = getattr(config, "pooling_method", "mean")
        num_attention_heads = getattr(config, "num_attention_heads", 8)
        loss = getattr(config, "loss", "mae")

        self.loss = loss
        n_labels = 1
        self.roberta = RobertaWithAdvancedPooling(model_name, num_labels=n_labels, pooling_method=pooling_method, dropout=dropout,
                                                  hidden_dim=hidden_dim, num_attention_heads=num_attention_heads, initialize_roberta=initialize_roberta)
        self.post_init()  # ensures weight init compatibility with HF

    def forward(self, input_ids, attention_mask, labels=None):
        # Get RoBERTa outputs
        logits = self.roberta(input_ids=input_ids,attention_mask=attention_mask)['logits'].squeeze(-1)

        loss = None
        if labels is not None:
            if self.loss == 'mae':
                loss_fn = nn.L1Loss()
            elif self.loss == 'mse':
                loss_fn = nn.MSELoss()
            elif self.loss == 'huber':
                loss_fn = nn.HuberLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and its configuration in a HuggingFace-compatible way.
        Works with Trainer.save_model().
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1. Save model weights (Trainer may pass state_dict)
        state_dict = kwargs.pop("state_dict", None)
        state_dict = self.state_dict() if state_dict is None else state_dict
        state_dict = {k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v for k, v in state_dict.items()}
        save_file(state_dict, os.path.join(save_directory, "model.safetensors"))

        roberta_config = self.roberta.roberta.config
        base_model_name = getattr(roberta_config, '_name_or_path', 'roberta-base')

        # 2. Save configuration needed to rebuild the model
        self.config.model_name = base_model_name
        self.config.base_model_name_or_path = base_model_name
        self.config.model_type = "roberta"
        self.config.architectures = ["RobertaForRegression"]
        self.config.vocab_size = roberta_config.vocab_size
        self.config.max_position_embeddings = roberta_config.max_position_embeddings
        self.config.hidden_size = roberta_config.hidden_size
        self.config.dropout = float(self.roberta.dropout.p)
        self.config.num_attention_heads = int(self.roberta.num_attention_heads)
        self.config.loss = self.loss
        self.config.pooling_method = self.roberta.pooling_method
        self.config.hidden_dim = self.roberta.hidden_dim
        self.config.initialize_roberta = False  # don't reinitialize when loading
        self.config.dtype = "float32"  # Save dtype info
        self.config.save_pretrained(save_directory)

        if hasattr(self, "generation_config") and self.generation_config is not None:
            self.generation_config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        """
        Load model weights + configuration from a directory created by save_pretrained().
        Avoids reinitializing the internal RoBERTa model.
        """
        # 1. Load saved configuration
        config = kwargs.pop("config", None)
        config = AutoConfig.from_pretrained(load_directory, **kwargs) if config is None else config
        config.initialize_roberta = False

        if hasattr(config, 'dtype'):
            delattr(config, 'dtype')

        # 3. Initialize model WITHOUT downloading weights again
        model = cls(config)

        # 4. Load weights
        pytorch_path = os.path.join(load_directory, "pytorch_model.bin")
        safetensors_path = os.path.join(load_directory, "model.safetensors")

        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")  # , weights_only=True
        else:
            raise FileNotFoundError(f"No model weights found in {load_directory}")

        torch_dtype = kwargs.pop("dtype", None)
        torch_dtype = torch.float32 if torch_dtype is None else torch_dtype
        model = model.to(torch_dtype)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"Missing keys when loading: {missing}", flush=True)
        if unexpected:
            print(f"Unexpected keys when loading: {unexpected}", flush=True)
        return model