"""Reproduce the learning-rate timeline from the active TDF-Font config.

This script intentionally uses tiny dummy parameters: optimizer and scheduler
state transitions are identical to train.py, while no model, dataset, or CUDA
memory is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def _lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _print_lrs(
    label: str,
    optimizer_kp: torch.optim.Optimizer,
    optimizer_generator: torch.optim.Optimizer,
    optimizer_discriminator: torch.optim.Optimizer,
) -> None:
    print(f"\n{label}")
    print(f"  optimizer_kp (Style Encoder): {_lrs(optimizer_kp)[0]:.12g}")
    print(f"  optimizer_kp (Transformer):   {_lrs(optimizer_kp)[1]:.12g}")
    print(f"  optimizer_generator:          {_lrs(optimizer_generator)[0]:.12g}")
    print(f"  optimizer_discriminator:      {_lrs(optimizer_discriminator)[0]:.12g}")


def _one_optimizer_step(
    optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
) -> None:
    optimizer.zero_grad()
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the actual TDF-Font optimizer/scheduler LR timeline."
    )
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train_params = config["train_params"]

    # Two groups reproduce train.py::optimizer_kp: Style Encoder and KP head.
    style_parameter = torch.nn.Parameter(torch.zeros(()))
    transformer_parameter = torch.nn.Parameter(torch.zeros(()))
    generator_parameter = torch.nn.Parameter(torch.zeros(()))
    discriminator_parameter = torch.nn.Parameter(torch.zeros(()))

    lr_generator = float(train_params.get("lr_generator", 2e-4))
    if bool(config.get("fine_tune_generator", False)):
        lr_generator = float(
            train_params.get("lr_generator_finetune", lr_generator)
        )

    optimizer_kp = torch.optim.Adam(
        [
            {
                "params": [style_parameter],
                "lr": float(train_params.get("lr_style_encoder", lr_generator)),
            },
            {
                "params": [transformer_parameter],
                "lr": float(train_params.get("lr_style_kp_head", lr_generator)),
            },
        ],
        betas=(0.5, 0.999),
    )
    optimizer_generator = torch.optim.Adam(
        [{"params": [generator_parameter], "lr": lr_generator}],
        betas=(0.5, 0.999),
    )
    optimizer_discriminator = torch.optim.Adam(
        [
            {
                "params": [discriminator_parameter],
                "lr": float(train_params.get("lr_discriminator", lr_generator)),
            }
        ],
        betas=(0.5, 0.999),
    )

    print(f"Config: {config_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch location: {torch.__file__}")
    print(f"epoch_milestones: {train_params.get('epoch_milestones', [0])}")
    print("scheduler gamma: 0.1")

    _print_lrs(
        "1. After optimizer creation, before scheduler creation",
        optimizer_kp,
        optimizer_generator,
        optimizer_discriminator,
    )

    milestones = train_params.get("epoch_milestones", [0])
    scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_generator, milestones, gamma=0.1, last_epoch=-1
    )
    scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_discriminator, milestones, gamma=0.1, last_epoch=-1
    )

    print(
        "\nScheduler last_epoch immediately after construction: "
        f"generator={scheduler_generator.last_epoch}, "
        f"discriminator={scheduler_discriminator.last_epoch}"
    )
    _print_lrs(
        "2. After scheduler creation, before the first training iteration",
        optimizer_kp,
        optimizer_generator,
        optimizer_discriminator,
    )

    # One step is sufficient: the LR stays unchanged during all iterations of
    # epoch 0 because train.py calls scheduler.step() only at the epoch boundary.
    _one_optimizer_step(optimizer_kp, [style_parameter, transformer_parameter])
    _one_optimizer_step(optimizer_generator, [generator_parameter])
    _one_optimizer_step(optimizer_discriminator, [discriminator_parameter])
    scheduler_generator.step()
    scheduler_discriminator.step()

    print(
        "\nScheduler last_epoch after epoch-0 scheduler.step(): "
        f"generator={scheduler_generator.last_epoch}, "
        f"discriminator={scheduler_discriminator.last_epoch}"
    )
    _print_lrs(
        "3. After scheduler.step() at the end of epoch 0",
        optimizer_kp,
        optimizer_generator,
        optimizer_discriminator,
    )
    _print_lrs(
        "4. At the beginning of epoch 1 (before any new step)",
        optimizer_kp,
        optimizer_generator,
        optimizer_discriminator,
    )


if __name__ == "__main__":
    main()
