#!/usr/bin/env python3
"""Parameterized launcher for the CausalMoE experiments.

Examples:
    python scripts/run.py --list
    python scripts/run.py --model smoe-m --task pretrain --dry-run
    python scripts/run.py --model smoe-m --task pretrain
    python scripts/run.py --model smoe-m --task finetune_code --dataset MBPP
    python scripts/run.py --model smoe-m-causal-loss --task corr2cause
    python scripts/run.py --model smoe-m-causal-loss --task causalbench --eval
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
ENTRYPOINT = ROOT / "train_causal.py"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def get_task_config(tasks, task):
    return tasks[task]


def build_command(args, model, task_cfg):
    task = args.task
    data_root = resolve(args.data_root)
    ckpt_root = resolve(args.ckpt_root)

    if task == "pretrain":
        data_path = data_root / task_cfg["data_subdir"]
        ckpt_dir = ckpt_root / model["id"] / "pretrain"
        lr = model["lr_pretrain"]
        lr_warmup = model["lr_warmup_pretrain"]
        pretrained = args.pretrained_weight
    elif task == "finetune_code":
        data_path = data_root / args.dataset
        ckpt_dir = ckpt_root / model["id"] / "finetune" / args.dataset
        lr = model["lr_finetune"]
        lr_warmup = model["lr_warmup_finetune"]
        pretrained = args.pretrained_weight or str(
            ckpt_root / model["id"] / "pretrain" / "smoe.pt"
        )
    else:
        data_path = data_root / task_cfg["data_subdir"]
        ckpt_dir = ckpt_root / model["id"] / "reasoning" / task
        lr = model["lr_reasoning"]
        lr_warmup = model["lr_warmup_reasoning"]
        pretrained = args.pretrained_weight or str(
            ckpt_root / model["id"] / "pretrain" / "smoe.pt"
        )

    if pretrained:
        pretrained = str(resolve(pretrained))
    checkpoint = str(ckpt_dir / "smoe.pt")

    cmd = [
        args.python,
        "-m",
        "torch.distributed.launch",
        "--master_port",
        str(args.master_port),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--use_env",
        str(ENTRYPOINT),
        "--data",
        str(data_path),
        "--base_arch",
        model["base_arch"],
        "--architecture",
        model["architecture"],
        "--gate_name",
        "smoe",
        "--nlayers",
        str(model["layers"]),
        "--hid-sz",
        str(model["hidden_size"]),
        "--inner-hid-sz",
        str(model["inner_hidden_size"]),
        "--nheads",
        str(model["num_heads"]),
        "--block-sz",
        str(model["block_size"]),
        "--attn-span",
        str(model["attn_span"]),
        "--dropout",
        str(model["dropout"]),
        "--load_balance",
        str(model["load_balance"]),
        "--optim",
        model["optim"],
        "--lr",
        str(lr),
        "--lr-warmup",
        str(lr_warmup),
        "--niter",
        str(task_cfg["niter"]),
        "--batch-sz",
        str(task_cfg["batch_size"]),
        "--batch-split",
        str(task_cfg["batch_split"]),
        "--nbatches",
        str(task_cfg["nbatches"]),
        "--distributed",
        "--checkpoint",
        checkpoint,
        "--cmp-sz",
        str(model["cmp_size"]),
        "--graph-sz",
        str(model["graph_size"]),
    ]

    for key in ("gamma1", "gamma2", "mu", "beta1", "beta2"):
        value = model.get(key)
        if value is not None:
            cmd.extend(["--{}".format(key), str(value)])

    if pretrained:
        cmd.extend(["--pretrained_weight", pretrained])
    if model.get("cal"):
        cmd.append("--causal-loss")
    if args.eval:
        cmd.extend(["--resume", "--full-eval-mode"])
    if args.extra:
        cmd.extend(shlex.split(args.extra))

    return cmd, data_path, ckpt_dir, pretrained, checkpoint


def print_model_list(models):
    print("{:<28} {:<42} {:<8} {:<5} {:<4}".format("id", "display_name", "family", "causal", "cal"))
    for model in models:
        print(
            "{:<28} {:<42} {:<8} {:<5} {:<4}".format(
                model["id"],
                model["display_name"],
                model["family"],
                str(model["causal"]),
                str(model["cal"]),
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Launch CausalMoE experiments.")
    parser.add_argument("--list", action="store_true", help="List the 21 model configurations.")
    parser.add_argument("--model", help="Model id from configs/models.json.")
    parser.add_argument(
        "--task",
        choices=["pretrain", "finetune_code", "corr2cause", "causalbench"],
        help="Experiment task.",
    )
    parser.add_argument("--dataset", help="Code dataset for --task finetune_code.")
    parser.add_argument("--data-root", default="data_directory", help="Root of the external data directory.")
    parser.add_argument("--ckpt-root", default="checkpoints", help="Root of checkpoints and logs.")
    parser.add_argument("--pretrained-weight", default=None, help="Override the pretrained checkpoint.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--cuda", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=10013)
    parser.add_argument("--python", default="python", help="Python executable.")
    parser.add_argument("--extra", default="", help="Extra raw arguments appended to the command.")
    parser.add_argument("--skip-checks", action="store_true", help="Do not check data/checkpoint paths.")
    parser.add_argument(
        "--allow-missing-pretrained",
        action="store_true",
        help="Allow fine-tuning/reasoning without an existing pretrained checkpoint.",
    )
    args = parser.parse_args()

    models = load_json(CONFIG_DIR / "models.json")["models"]
    tasks = load_json(CONFIG_DIR / "tasks.json")

    if args.list:
        print_model_list(models)
        return 0

    if not args.model or not args.task:
        parser.error("--model and --task are required unless --list is used.")
    if args.task == "finetune_code" and not args.dataset:
        parser.error("--dataset is required for --task finetune_code.")

    model = next((m for m in models if m["id"] == args.model), None)
    if model is None:
        parser.error("Unknown model id: {}. Use --list to see available ids.".format(args.model))

    task_cfg = get_task_config(tasks, args.task)
    cmd, data_path, ckpt_dir, pretrained, checkpoint = build_command(args, model, task_cfg)

    print("Model:      {} ({})".format(model["id"], model["display_name"]))
    print("Task:       {}".format(args.task))
    print("Data:       {}".format(data_path))
    print("Checkpoint: {}".format(checkpoint))
    if pretrained:
        print("Pretrained: {}".format(pretrained))
    print("Command:    {}".format(" ".join(shlex.quote(x) for x in cmd)))

    if args.dry_run:
        return 0

    if not args.skip_checks:
        if not data_path.exists():
            print("ERROR: data path does not exist: {}".format(data_path), file=sys.stderr)
            print("Place the dataset there, or pass --data-root / --skip-checks.", file=sys.stderr)
            return 2
        if args.task in ("finetune_code", "corr2cause", "causalbench"):
            if pretrained and not Path(pretrained).exists() and not args.allow_missing_pretrained:
                print("ERROR: pretrained checkpoint does not exist: {}".format(pretrained), file=sys.stderr)
                print("Run the pretrain task first, or pass --pretrained-weight.", file=sys.stderr)
                return 2
        if args.eval and not Path(checkpoint).exists():
            print("WARNING: checkpoint does not exist yet: {}".format(checkpoint), file=sys.stderr)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    sys.exit(main())
