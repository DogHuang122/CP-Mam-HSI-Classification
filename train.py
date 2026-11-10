"""Train CP-Mamba and save the final checkpoint with its preprocessing."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cpmamba import CP_Mamba
from data import SCENES, PatchDataset, fit_preprocessing, load_scene, project_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SCENES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new run directory.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43, help="Training seed. The split seed is fixed.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch installation and GPU are required.")
    if args.output_dir.exists():
        parser.error("--output-dir already exists. Choose a new run directory.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    image, coords, labels = load_scene(args.data_dir, args.dataset)
    prep = fit_preprocessing(image, coords, labels, args.dataset)
    padded = project_scene(image, prep)
    del image
    train_data = PatchDataset(padded, prep, "train")
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = CP_Mamba(num_classes=SCENES[args.dataset].classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "split_and_pca.npz", **prep)
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_sum = torch.zeros((), device=device)
        for patches, targets in loader:
            patches = patches.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(patches), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * targets.size(0)
        loss_mean = loss_sum.item() / len(train_data)
        if not math.isfinite(loss_mean):
            raise RuntimeError(f"Nonfinite training loss in epoch {epoch}.")
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  loss {loss_mean:.6f}", flush=True)

    checkpoint = args.output_dir / "final_weights.pth"
    temp_checkpoint = checkpoint.with_suffix(".tmp")
    torch.save({
        "dataset": args.dataset,
        "epoch": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
    }, temp_checkpoint)
    temp_checkpoint.replace(checkpoint)
    print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
