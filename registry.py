# text2tobi/registry.py
# Model registry: name → checkpoint config.
# Swap hub_path to a HuggingFace Hub ID once the checkpoint is uploaded.
# For local use, set hub_path to an absolute path on disk.

MODEL_REGISTRY = {
    "libri+sbc_pos_stl": {
        "hub_path": None,           # set to HF Hub ID or local path before use
        "requires_pos": True,
    },
    # Future models slotted in here.
}

DEFAULT_MODEL = "libri+sbc_pos_stl"
