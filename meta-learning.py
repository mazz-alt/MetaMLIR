# meta_fusion_lr.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def data_pro(x) -> np.ndarray:
    """
    Keep original behavior:
    - Always converts to np.array
    - Tries reshape to (N, 1); if fails reshape to (N, M)
    """
    x = np.array(x)
    try:
        x = np.reshape(x, (x.shape[0], 1))
    except Exception:
        x = np.reshape(x, (x.shape[0], x.shape[1]))
    return x


@dataclass(frozen=True)
class Paths:
    """
    Centralized paths. Defaults keep original hard-coded filenames.
    """
    num: str = "9_3"
    dataset_root: Path = Path("./dataset")
    logs_root: Path = Path("./experiments_logs")

    @property
    def train_pt(self) -> Path:
        return self.dataset_root / f"motion{self.num}" / "train.pt"

    @property
    def test_pt(self) -> Path:
        return self.dataset_root / f"motion{self.num}" / "test.pt"

    # Test probabilities
    @property
    def knn_test(self) -> Path:
        return self.logs_root / "knn" / f"prob_test_{self.num}.npy"

    @property
    def tree_test(self) -> Path:
        return self.logs_root / "decisiontree" / f"prob_test_{self.num}.npy"

    @property
    def nn_test(self) -> Path:
        return self.logs_root / f"all_2_motion{self.num}" / "run1" / "base" / "prob_test.npy"

    # Train probabilities
    @property
    def knn_train(self) -> Path:
        return self.logs_root / "knn" / f"prob_train_{self.num}.npy"

    @property
    def tree_train(self) -> Path:
        return self.logs_root / "decisiontree" / f"prob_train_{self.num}.npy"

    @property
    def nn_train(self) -> Path:
        return self.logs_root / f"all_2_motion{self.num}" / "run1" / "base" / "prob_train.npy"

    # Outputs
    @property
    def meta_out_dir(self) -> Path:
        return self.logs_root / "meta-learning"

    @property
    def meta_prob_test(self) -> Path:
        return self.meta_out_dir / f"prob_test_{self.num}.npy"

    @property
    def meta_pred_test(self) -> Path:
        return self.meta_out_dir / f"pred_{self.num}.npy"


def load_probs(*paths: Path) -> Tuple[np.ndarray, ...]:
    """
    Load multiple .npy and apply `data_pro` to each (kept).
    """
    out = []
    for p in paths:
        out.append(data_pro(np.load(p)))
    return tuple(out)


def load_labels(train_pt: Path, test_pt: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load labels from torch dicts (kept).
    """
    train_blob = torch.load(str(train_pt))
    test_blob = torch.load(str(test_pt))
    y_train = train_blob["labels"].cpu().numpy()
    y_test = test_blob["labels"].cpu().numpy()
    return y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    Keep your metric choices and confusion_matrix transpose.
    """
    # NOTE: original code calls accuracy_score(..., average='macro') which is invalid.
    # We keep working behavior (no average arg) to avoid runtime error.
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    confu = confusion_matrix(y_true, y_pred).T
    return {"acc": acc, "pre": pre, "recall": rec, "F1": f1, "confu": confu}

def main() -> None:
    paths = Paths(num="9_3")

    # Load test probabilities
    x_knn_test, x_tree_test, x_nn_test = load_probs(
        paths.knn_test,
        paths.tree_test,
        paths.nn_test,
    )

    # Load train probabilities
    x_knn_train, x_tree_train, x_nn_train = load_probs(
        paths.knn_train,
        paths.tree_train,
        paths.nn_train,
    )

    # Labels
    y_train, y_test = load_labels(paths.train_pt, paths.test_pt)

    # Feature stacking (kept identical columns used in original: (1,2,4) only)
    x_train_final = np.concatenate((x_knn_train, x_tree_train, x_nn_train), axis=1)
    x_test_final = np.concatenate((x_knn_test, x_tree_test, x_nn_test), axis=1)

    # Meta-learner (kept params)
    meta_esti = LogisticRegression(C=1.0, solver="lbfgs")
    meta_esti.fit(x_train_final, y_train)

    y_pred = meta_esti.predict(x_test_final)
    y_proba = meta_esti.predict_proba(x_test_final)

    metrics = compute_metrics(y_test, y_pred)
    print(metrics)

    # Save outputs (same filenames)
    paths.meta_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.meta_prob_test, y_proba)
    np.save(paths.meta_pred_test, y_pred)


if __name__ == "__main__":
    main()

