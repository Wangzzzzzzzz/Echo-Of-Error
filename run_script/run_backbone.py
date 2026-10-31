#!/usr/bin/env python3
"""Backbone CL experiments. Dry-run by default; --execute runs locally.

No scheduler, cluster account, or hard-coded data path is required.
"""
import argparse
from itertools import product
from pathlib import Path
import shlex
import subprocess
import sys

if __package__:
    from .experiment_settings import get_backbone_configs, FULL_BATCH_OVERRIDES
else:
    from experiment_settings import get_backbone_configs, FULL_BATCH_OVERRIDES

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "consistency_loss/run_forecasting_res_nl.py"
DATASETS = {
    "ETTh2": ("ETT-small", "ETTh2.csv", "ETTh2", 7),
    "ETTm2": ("ETT-small", "ETTm2.csv", "ETTm2", 7),
    "ECL": ("electricity", "electricity.csv", "custom", 321),
}
MODELS = ("FilterNet", "TSLANet", "iTransformer", "pdf")
BACKBONE_LAMBDAS = (0, 0.125, 0.25, 0.5, 2, 8)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "model_result/experiments")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--lambdas", nargs="+", type=float)
    parser.add_argument("--checker-sizes", nargs="+", type=int, default=[0])
    parser.add_argument("--jvp-bn-policy", choices=["subset_train_stats", "running_eval_stats"],
                        default="subset_train_stats")
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, help="Override preset physical batch sizes explicitly.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Explicit default; cannot combine with --execute.")
    return parser


def arguments(options):
    """Convert runner options to argv, expanding list-valued flags in place."""
    result = []
    for key, value in options.items():
        result.append("--" + key)
        values = value if isinstance(value, (tuple, list)) else [value]
        result.extend(str(v) for v in values)
    return result


def _validate_experiment_args(args, datasets, lambdas):
    """Check the complete plan before yielding any training commands."""
    if any(d not in DATASETS for d in datasets):
        raise ValueError("backbone presets cover ETTh2, ETTm2 and ECL only")
    if args.train_steps < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("train_steps and batch_size must be positive")
    if any(x <= 0 for x in args.pred_lens):
        raise ValueError("horizons must be positive")
    if any(k < 0 for k in args.checker_sizes):
        raise ValueError("checker sizes must be nonnegative")
    if any(not 0 <= value < float("inf") for value in lambdas):
        raise ValueError("lambdas must be finite and nonnegative")


def _backbone_options(args, dataset, common, seeds, lambdas):
    """Expand backbone presets across checker size, horizon, seed and lambda."""
    configs = get_backbone_configs()
    for model, checker_size in product(args.models, args.checker_sizes):
        config = dict(configs[(model, dataset)])
        if checker_size == 0:
            config.update(FULL_BATCH_OVERRIDES.get((model, dataset), {}))
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size

        for horizon, seed, coefficient in product(args.pred_lens, seeds, lambdas):
            yield dict(
                common,
                **config,
                model=model,
                val_type="res",
                pred_len=horizon,
                seed=seed,
                regu_coef=coefficient,
                checker_batch_size=checker_size,
                jvp_bn_policy=args.jvp_bn_policy,
            )


def build_commands(args):
    """Yield deterministic commands without reading data or starting training.

    Dataset defaults and shared flags live here; backbone presets are expanded
    in _backbone_options. All grid expansion has a deterministic order.
    """
    datasets = args.datasets or list(DATASETS)
    lambdas = args.lambdas if args.lambdas is not None else BACKBONE_LAMBDAS
    _validate_experiment_args(args, datasets, lambdas)

    for dataset in datasets:
        folder, filename, data, channels = DATASETS[dataset]
        seeds = args.seeds or [2021, 2022, 2023]
        common = dict(
            task_name="long_term_forecast",
            is_training=1,
            root_path=str(args.data_root / folder),
            data_path=filename,
            model_id=dataset,
            data=data,
            features="M",
            seq_len=720,
            label_len=0,
            enc_in=channels,
            dec_in=channels,
            c_out=channels,
            train_steps=args.train_steps,
            val_every=min(50, args.train_steps),
            output_dir=str(args.output_root / "backbone"),
            instance_norm=True,
            detaching=True,
        )
        for options in _backbone_options(args, dataset, common, seeds, lambdas):
            yield [args.python, "-u", str(RUNNER), *arguments(options)]


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    # Commands run from ROOT; resolve user paths before changing the subprocess cwd.
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    try:
        commands = list(build_commands(args))
    except ValueError as error:
        parser.error(str(error))
    if args.execute:
        # Validate all inputs before starting any training.
        for command in commands:
            root = Path(command[command.index("--root_path") + 1])
            path = root / command[command.index("--data_path") + 1]
            if not path.is_file():
                parser.error(f"missing dataset: {path}")
    print(f"{len(commands)} training commands; mode={'EXECUTE' if args.execute else 'DRY RUN'}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
