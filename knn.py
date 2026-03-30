# knn_train_eval.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def replace_label_with_repeat(
    labels: np.ndarray,
    data: np.ndarray,
    old_label: int,
    new_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace all samples with label `old_label` by samples of label `new_label`.
    If `new_label` samples are fewer, repeat them until counts match.

    Notes
    -----
    Mutates `labels` and `data` in-place (kept identical to your original behavior).
    """
    if labels.shape[0] != data.shape[0]:
        raise ValueError("labels 和 data 第一维必须一致")

    old_idx = np.where(labels[:] == old_label)[0]
    new_idx = np.where(labels[:] == new_label)[0]
    if len(new_idx) == 0:
        raise ValueError(f"标签 {new_label} 不存在，无法替换")

    repeat_times = int(np.ceil(len(old_idx) / len(new_idx)))
    new_idx_extended = np.tile(new_idx, repeat_times)[: len(old_idx)]

    labels[old_idx] = new_label
    data[old_idx] = data[new_idx_extended]
    return labels, data


def flatten_xy(samples_xy: np.ndarray) -> np.ndarray:
    """
    (N, T, 2) -> (N, 2*T) by concatenating x then y.
    Kept identical to: np.concatenate((data[:,:,0], data[:,:,1]), axis=1)
    """
    return np.concatenate((samples_xy[:, :, 0], samples_xy[:, :, 1]), axis=1)


@dataclass(frozen=True)
class Config:
    dataset_num: str = "9_3"
    dataset_dir: Path = Path("./dataset")
    log_dir: Path = Path("./experiments_logs/knn")

    replace_old_label: int = 3
    replace_new_label: int = 4

    k_values: Tuple[int, ...] = (3,)

    @property
    def train_path(self) -> Path:
        return self.dataset_dir / f"motion{self.dataset_num}" / "train.pt"

    @property
    def test_path(self) -> Path:
        return self.dataset_dir / f"motion{self.dataset_num}" / "test.pt"


def load_pt_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load torch-saved dict: {'samples': ..., 'labels': ...} as numpy arrays.
    """
    blob = torch.load(str(path))
    samples = blob["samples"].cpu().numpy()
    labels = blob["labels"].cpu().numpy()
    return samples, labels


def prepare_xy_data(
    samples: np.ndarray,
    labels: np.ndarray,
    old_label: int,
    new_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep your preprocessing exactly:
    - keep only first 2 features: [:, :, :2]
    - replace label with repeat strategy
    - flatten x/y to 2*T features
    """
    samples_xy = samples[:, :, :2]
    labels, samples_xy = replace_label_with_repeat(labels, samples_xy, old_label=old_label, new_label=new_label)
    x = flatten_xy(samples_xy)
    y = labels
    return x, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    Same metrics as your script; confusion matrix transposed.
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    confu = confusion_matrix(y_true, y_pred).T
    return {"acc": acc, "pre": pre, "recall": rec, "F1": f1, "confu": confu}


def save_knn_outputs(
    out_dir: Path,
    dataset_num: str,
    prob_test: np.ndarray,
    pred_test: np.ndarray,
    prob_train: np.ndarray,
) -> None:
    """
    Save outputs with same filenames as original.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"prob_test_{dataset_num}.npy", prob_test)
    np.save(out_dir / f"pred_{dataset_num}.npy", pred_test)
    np.save(out_dir / f"prob_train_{dataset_num}.npy", prob_train)


def run_for_k(
    k: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Train + predict + metrics for a single k (same behavior).
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    pred_test = knn.predict(x_test)
    prob_train = knn.predict_proba(x_train)
    prob_test = knn.predict_proba(x_test)

    metrics = compute_metrics(y_test, pred_test)
    return knn, pred_test, prob_test, prob_train, metrics


def main() -> None:
    cfg = Config()

    train_samples, train_labels = load_pt_dataset(cfg.train_path)
    test_samples, test_labels = load_pt_dataset(cfg.test_path)

    x_train, y_train = prepare_xy_data(
        train_samples,
        train_labels,
        old_label=cfg.replace_old_label,
        new_label=cfg.replace_new_label,
    )
    x_test, y_test = prepare_xy_data(
        test_samples,
        test_labels,
        old_label=cfg.replace_old_label,
        new_label=cfg.replace_new_label,
    )

    for k in cfg.k_values:
        _, pred_test, prob_test, prob_train, metrics = run_for_k(k, x_train, y_train, x_test, y_test)

        print(metrics)
        print(f"k = {k}, Accuracy = {metrics['acc']:.2f}")

        save_knn_outputs(
            out_dir=cfg.log_dir,
            dataset_num=cfg.dataset_num,
            prob_test=prob_test,
            pred_test=pred_test,
            prob_train=prob_train,
        )


if __name__ == "__main__":
    main()