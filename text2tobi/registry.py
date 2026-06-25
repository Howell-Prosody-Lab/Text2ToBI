# text2tobi/registry.py
# Model registry: name → checkpoint config.
# Set hub_path to a HuggingFace Hub ID (e.g. "your-handle/text2tobi") once uploaded.
# For local use, set hub_path to an absolute path on disk instead.

MODEL_REGISTRY = {
    "libri+peoples+sbc": {
        "hub_path": "Primaski/text2tobi",
        "requires_pos": False,
    },
    # Future models slotted in here.
}

DEFAULT_MODEL = "libri+peoples+sbc"
