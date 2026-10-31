"""Portable, collision-resistant run identifiers and machine-readable results."""
import hashlib
import json
from pathlib import Path
import re


def run_setting(args):
    """Identify the training configuration independently of output location.

    Evaluation-only arguments are excluded so a checkpoint's one-shot and
    rollout results live alongside its training results, not in new run folders.
    """
    config = vars(args).copy()
    for key in ("output_dir", "checkpoints", "load_ckpt", "is_training", "test_pred_lens"):
        config.pop(key, None)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_id)[:40]
    return (f"{args.model}_{label}_L{args.seq_len}_H{args.pred_len}"
            f"_s{args.seed}_cl{args.regu_coef:g}_mlp{args.regu_coef_mlp}"
            f"_k{args.checker_batch_size}_r{args.checker_batch_ratio:g}_{digest}")


def write_json(path, value):
    """Write readable experiment metadata, rejecting non-finite metric values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
