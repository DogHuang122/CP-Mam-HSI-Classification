"""Dataset preparation shared by the training and evaluation scripts."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Scene:
    image_file: str
    image_key: str
    label_file: str
    label_key: str
    classes: int
    train_ratio: float
    split_seed: int


SCENES = {
    "IP": Scene("Indian_pines_corrected.mat", "indian_pines_corrected",
                "Indian_pines_gt.mat", "indian_pines_gt", 16, 0.05, 42),
    "PU": Scene("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt", 9, 0.005, 43),
    "SA": Scene("Salinas_corrected.mat", "salinas_corrected",
                "Salinas_gt.mat", "salinas_gt", 16, 0.005, 43),
    "HH": Scene("WHU_Hi_HongHu.mat", "WHU_Hi_HongHu",
                "WHU_Hi_HongHu_gt.mat", "WHU_Hi_HongHu_gt", 22, 0.005, 43),
}
BANDS = 30
PATCH_SIZE = 11


def load_scene(data_dir, dataset):
    spec = SCENES[dataset]
    image = loadmat(data_dir / spec.image_file, variable_names=[spec.image_key])[spec.image_key]
    gt = loadmat(data_dir / spec.label_file, variable_names=[spec.label_key])[spec.label_key]
    if image.ndim != 3 or image.shape[:2] != gt.shape:
        raise ValueError("The hyperspectral image and ground truth shapes do not match.")
    coords = np.column_stack(np.nonzero(gt))
    labels = gt[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
    if not np.array_equal(np.unique(labels), np.arange(spec.classes)):
        raise ValueError(f"Ground truth for {dataset} must contain labels 1 through {spec.classes}.")
    return image, coords, labels


def fit_preprocessing(image, coords, labels, dataset):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    spec = SCENES[dataset]
    pca = PCA(n_components=BANDS, whiten=True, random_state=42)
    pca.fit(image.reshape(-1, image.shape[-1]))
    train, test = train_test_split(
        np.arange(len(labels)), train_size=spec.train_ratio,
        stratify=labels, random_state=spec.split_seed,
    )
    return {
        "dataset": np.array(dataset),
        "coords": coords,
        "labels": labels,
        "train": train,
        "test": test,
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_variance": pca.explained_variance_,
    }


def load_preprocessing(path, dataset, coords, labels):
    with np.load(path, allow_pickle=False) as saved:
        prep = dict(saved)
    if "dataset" in prep and prep["dataset"].item() != dataset:
        raise ValueError(f"The saved preprocessing belongs to a different dataset than {dataset}.")
    if not np.array_equal(prep["coords"], coords) or not np.array_equal(prep["labels"], labels):
        raise ValueError("The saved split does not match the labeled pixels in the ground truth.")
    train, test = prep["train"], prep["test"]
    if (any(idx.ndim != 1 or idx.dtype.kind not in "iu" or idx.size == 0 for idx in (train, test))
            or not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(len(labels)))):
        raise ValueError("The saved train and test indices must partition the labeled pixels.")
    return prep


def project_scene(image, prep):
    bands = image.shape[-1]
    mean = prep["pca_mean"]
    components = prep["pca_components"]
    variance = prep["pca_variance"]
    if mean.shape != (bands,) or components.shape != (BANDS, bands) or variance.shape != (BANDS,):
        raise ValueError("The saved PCA dimensions do not match the hyperspectral image.")
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError("Whitening requires finite, positive PCA variances.")
    projected = (image.reshape(-1, bands) - mean) @ components.T
    projected /= np.sqrt(variance)
    projected = projected.reshape(*image.shape[:2], BANDS).astype(np.float32)
    radius = PATCH_SIZE // 2
    return np.pad(projected, ((radius, radius), (radius, radius), (0, 0)))


class PatchDataset(Dataset):
    """Extract patches on demand from the zero-padded PCA cube."""

    def __init__(self, padded, prep, split):
        indices = prep[split]
        self.padded = padded
        self.coords = prep["coords"][indices]
        self.labels = prep["labels"][indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row, col = self.coords[index]
        patch = self.padded[row:row + PATCH_SIZE, col:col + PATCH_SIZE].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(patch)), self.labels[index]
