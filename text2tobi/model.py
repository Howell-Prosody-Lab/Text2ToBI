# text2tobi/model.py
# ProsodyBoundaryModel definition, copied from distilBERT_pos.ipynb Cell 7.
# This class is NOT registered in HuggingFace's model hub, so it must be
# defined here for from_pretrained() to work at inference time.

import torch
import torch.nn as nn
from transformers import (
    DistilBertModel,
    DistilBertPreTrainedModel,
    AutoTokenizer,
)

# ── POS tag vocabulary (Universal Dependencies / spaCy UPOS) ─────────────────
# Must match the vocabulary defined in the training notebook exactly.
UNIVERSAL_TO_TOKEN = {
    "ADJ":   "adj",
    "ADP":   "adp",
    "ADV":   "adv",
    "AUX":   "aux",
    "CCONJ": "cc",
    "DET":   "det",
    "INTJ":  "ij",
    "NOUN":  "nn",
    "NUM":   "num",
    "PART":  "pt",
    "PRON":  "pro",
    "PROPN": "np",
    "PUNCT": "pun",
    "SCONJ": "sc",
    "SYM":   "sym",
    "VERB":  "vb",
    "X":     "xx",
    "SPACE": "sp",
}
UNK_POS_TOKEN = "unk"

_POS_TAG_NAMES = ["PAD"] + list(UNIVERSAL_TO_TOKEN.keys())
POS_TAG_TO_ID  = {tag: i for i, tag in enumerate(_POS_TAG_NAMES)}
NUM_POS_TAGS   = len(_POS_TAG_NAMES)   # 19


class ProsodyBoundaryModel(DistilBertPreTrainedModel):
    """
    Architecture
    ────────────
    DistilBERT encoder
        [+ optional POS embedding addition, post-transformer]
        └─► dropout (seq_classif_dropout)
             ├─► boundary_head    Linear(H → 2)   all positions
             ├─► intonation_head  Linear(H → 3)   rising / falling / level
             └─► break_idx_head   Linear(H → 2)   index-3 / index-4

    POS embedding design (combined mode)
    ─────────────────────────────────────
    When use_pos_embedding=True, a small nn.Embedding(NUM_POS_TAGS, pos_emb_dim)
    maps integer POS IDs to vectors of size pos_emb_dim (default 64). A Linear
    projection then maps these to H=768, and the result is ADDED to DistilBERT's
    last hidden state AFTER the transformer.
    """

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout    = nn.Dropout(config.seq_classif_dropout)

        self.use_pos_embedding = getattr(config, "use_pos_embedding", False)
        if self.use_pos_embedding:
            _pos_emb_dim  = getattr(config, "pos_emb_dim",  64)
            _num_pos_tags = getattr(config, "num_pos_tags", NUM_POS_TAGS)
            self.pos_embedding = nn.Embedding(
                _num_pos_tags, _pos_emb_dim, padding_idx=0
            )
            self.pos_proj = nn.Linear(_pos_emb_dim, config.hidden_size, bias=False)

        self.boundary_head   = nn.Linear(config.hidden_size, 2)
        self.intonation_head = nn.Linear(config.hidden_size, 3)
        self.break_idx_head  = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(self, input_ids, attention_mask, pos_ids=None, **kwargs):
        """
        Parameters
        ----------
        input_ids      : (B, T)
        attention_mask : (B, T)
        pos_ids        : (B, T) LongTensor | None  — only used in combined mode

        Returns
        -------
        dict with keys:
            boundary_logits    : (B, T, 2)
            intonation_logits  : (B, T, 3)
            break_idx_logits   : (B, T, 2)
        """
        outputs = self.distilbert(input_ids=input_ids,
                                  attention_mask=attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)   # (B, T, H)

        if self.use_pos_embedding and pos_ids is not None:
            pos_emb = self.pos_proj(self.pos_embedding(pos_ids))   # (B, T, H)
            seq_out = seq_out + pos_emb

        return {
            "boundary_logits":   self.boundary_head(seq_out),    # (B, T, 2)
            "intonation_logits": self.intonation_head(seq_out),  # (B, T, 3)
            "break_idx_logits":  self.break_idx_head(seq_out),   # (B, T, 2)
        }

    @classmethod
    def _can_set_experts_implementation(cls):
        return False


def load_model_and_tokenizer(checkpoint_path: str):
    """
    Load ProsodyBoundaryModel and tokenizer from a local checkpoint directory
    or a HuggingFace Hub ID.

    Parameters
    ----------
    checkpoint_path : str
        Absolute local path or HuggingFace Hub model ID.

    Returns
    -------
    model     : ProsodyBoundaryModel (eval mode, CPU)
    tokenizer : PreTrainedTokenizerFast
    """
    try:
        model = ProsodyBoundaryModel.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except OSError as e:
        raise RuntimeError(
            f"Could not load model from '{checkpoint_path}'.\n"
            f"  If using a local path, check that the directory exists and contains "
            f"pytorch_model.bin (or model.safetensors), config.json, tokenizer.json, "
            f"tokenizer_config.json, vocab.txt, and special_tokens_map.json.\n"
            f"  If using a HuggingFace Hub ID, run: text2tobi download <model_name>\n"
            f"  Original error: {e}"
        ) from e

    model.eval()
    return model, tokenizer
