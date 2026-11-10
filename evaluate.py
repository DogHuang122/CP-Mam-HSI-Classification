"""Evaluate a CP-Mamba checkpoint on its held-out test pixels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cpmamba import CP_Mamba
from data import SCENES, PatchDataset, load_preprocessing, load_scene, project_scene


def classification_metrics(confusion):
    count = confusion.sum()
    support = confusion.sum(axis=1)
    correct = confusion.diagonal()
    present = support > 0
    oa = correct.sum() / count
    class_accuracy = correct[present] / support[present]
    expected = (support / count) @ (confusion.sum(axis=0) / count)
    return {
        "test_samples": int(count),
        "oa": float(100 * oa),
        "aa": float(100 * class_accuracy.mean()),
        "kappa": float(100 * (oa - expected) / (1 - expected)) if expected < 1 else None,
        "per_class_accuracy": {
            int(index + 1): float(100 * accuracy)
            for index, accuracy in zip(np.flatnonzero(present), class_accuracy)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=SCENES, help="Required for a bare state dict.")
    parser.add_argument("--split-file", type=Path, help="Defaults to split_and_pca.npz beside the checkpoint.")
    parser.add_argument("--batch-size", type=int, help="Defaults to the saved batch size or 64.")
    parser.add_argument("--predictions", type=Path, help="Optional path to a test predictions .npz.")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch installation and GPU are required.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    stored_dataset = checkpoint.get("dataset")
    if args.dataset and stored_dataset and args.dataset != stored_dataset:
        parser.error("--dataset does not match the checkpoint")
    dataset = stored_dataset or args.dataset
    if dataset not in SCENES:
        parser.error("Provide --dataset for a bare state dict.")
    split_file = args.split_file or args.checkpoint.with_name("split_and_pca.npz")
    image, coords, labels = load_scene(args.data_dir, dataset)
    prep = load_preprocessing(split_file, dataset, coords, labels)
    padded = project_scene(image, prep)
    del image
    test_data = PatchDataset(padded, prep, "test")
    batch_size = args.batch_size or checkpoint.get("batch_size", 64)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    classes = SCENES[dataset].classes
    model = CP_Mamba(num_classes=classes).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for patches, _ in loader:
            logits = model(patches.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)
    confusion = np.bincount(
        classes * test_data.labels + predictions, minlength=classes**2,
    ).reshape(classes, classes)
    print(json.dumps({"dataset": dataset, **classification_metrics(confusion)}, indent=2))
    if args.predictions:
        np.savez_compressed(
            args.predictions, coords=test_data.coords, labels=test_data.labels + 1,
            predictions=predictions + 1,
        )


if __name__ == "__main__":
    main()
