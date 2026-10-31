"""Backbone presets keyed by (model name, dataset name).

Base presets are used for sampled JVP. Full-batch differences are listed
separately in FULL_BATCH_OVERRIDES.
"""
from typing import Any, Dict, Tuple


def get_backbone_configs() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return fresh preset dictionaries so callers can safely apply overrides."""
    configs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for dataset, emb_dim, depth, patch_size, batch_size in (
        ("ETTh2", 64, 1, 32, 128),
        ("ETTm2", 128, 1, 64, 128),
        ("ECL", 128, 2, 32, 64),
    ):
        configs[("TSLANet", dataset)] = {
            "batch_size": batch_size,
            "learning_rate": 5e-4,
            "dropout": 0.5,
            "d_model": 64,
            "emb_dim": emb_dim,
            "depth": depth,
            "patch_size": patch_size,
        }

    for dataset, hidden_size, batch_size, learning_rate in (
        ("ETTh2", 256, 128, 5e-3),
        ("ETTm2", 128, 128, 5e-3),
        ("ECL", 256, 64, 5e-4),
    ):
        configs[("FilterNet", dataset)] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": 0.0,
            "d_model": 64,
            "hidden_size": hidden_size,
        }

    for dataset, d_model, e_layers, batch_size, learning_rate in (
        ("ETTh2", 128, 2, 128, 1e-4),
        ("ETTm2", 128, 2, 128, 1e-4),
        ("ECL", 512, 3, 64, 5e-4),
    ):
        configs[("iTransformer", dataset)] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": 0.1,
            "d_model": d_model,
            "d_ff": d_model,
            "e_layers": e_layers,
            "n_heads": 8,
        }

    configs[("pdf", "ETTh2")] = {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "fc_dropout": 0.1,
        "d_model": 16,
        "d_ff": 128,
        "e_layers": 3,
        "n_heads": 4,
        "kernel_list": [3, 7, 11],
        "period": [24, 180, 720],
        "patch_len": [4, 16, 48],
        "stride": [4, 16, 48],
    }
    configs[("pdf", "ETTm2")] = {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "fc_dropout": 0.1,
        "d_model": 48,
        "d_ff": 48,
        "e_layers": 3,
        "n_heads": 16,
        "kernel_list": [3, 7, 11],
        "period": [102, 720],
        "patch_len": [16, 24],
        "stride": [16, 24],
    }
    configs[("pdf", "ECL")] = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "fc_dropout": 0.1,
        "d_model": 128,
        "d_ff": 256,
        "e_layers": 3,
        "n_heads": 32,
        "kernel_list": [3, 5, 7, 7],
        "period": [8, 12, 24, 180, 720],
        "patch_len": [1, 2, 3, 16, 48],
        "stride": [1, 2, 3, 16, 48],
    }
    return configs


# Full-JVP TSLANet/ECL used B=32 after B=64 ran out of memory.
FULL_BATCH_OVERRIDES = {("TSLANet", "ECL"): {"batch_size": 32}}
