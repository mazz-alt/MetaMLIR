from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


def replace_label_with_repeat(
    labels: np.ndarray,
    data: np.ndarray,
    old_label: int,
    new_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace samples labeled `old_label` with samples labeled `new_label`.
    If `new_label` samples are fewer than `old_label` samples, repeat indices until matched.

    Notes
    -----
    - This function mutates `labels` and `data` in-place (kept identical to original behavior).
    - Expects labels and data to share the same first dimension length.
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


def flatten_xy_features(samples: np.ndarray) -> np.ndarray:
    """
    Convert samples of shape (N, T, 2) -> (N, 2*T) by concatenating x and y.
    Kept identical to: np.concatenate((data[:,:,0], data[:,:,1]), axis=1)
    """
    return np.concatenate((samples[:, :, 0], samples[:, :, 1]), axis=1)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_num: str = "9_3"
    base_dir: Path = Path("./dataset")
    log_dir: Path = Path("./experiments_logs/decisiontree")

    @property
    def dataset_dir(self) -> Path:
        return self.base_dir / f"motion{self.dataset_num}"

    def train_path(self) -> Path:
        return self.dataset_dir / "train.pt"

    def test_path(self) -> Path:
        return self.dataset_dir / "test.pt"


@dataclass(frozen=True)
class LabelReplaceConfig:
    old_label: int = 4
    new_label: int = 3


@dataclass(frozen=True)
class ModelConfig:
    min_samples_leaf: int = 40
    criterion: str = "entropy"
    random_state: int = 42


def load_torch_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load {samples, labels} from a torch .pt dict and return as numpy arrays.
    """
    blob = torch.load(str(path))
    samples = blob["samples"].cpu().numpy()
    labels = blob["labels"].cpu().numpy()
    return samples, labels


def prepare_xy_classification_data(
    samples: np.ndarray,
    labels: np.ndarray,
    replace_cfg: LabelReplaceConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep identical preprocessing:
    - use only first 2 features: samples[:, :, :2]
    - replace label old->new with repeat strategy
    - flatten x,y to (N, 2*T)
    """
    samples_xy = samples[:, :, :2]
    labels, samples_xy = replace_label_with_repeat(
        labels=labels,
        data=samples_xy,
        old_label=replace_cfg.old_label,
        new_label=replace_cfg.new_label,
    )
    x = flatten_xy_features(samples_xy)
    y = labels
    return x, y


def train_decision_tree(x_train: np.ndarray, y_train: np.ndarray, cfg: ModelConfig) -> DecisionTreeClassifier:
    """
    Train DecisionTreeClassifier with the same hyperparameters as original.
    """
    model = DecisionTreeClassifier(
        min_samples_leaf=cfg.min_samples_leaf,
        criterion=cfg.criterion,
        random_state=cfg.random_state,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """
    Compute metrics exactly like original:
    - macro precision/recall/f1
    - confusion matrix transposed
    """
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro")
    recall = recall_score(y_test, pred, average="macro")
    f1 = f1_score(y_test, pred, average="macro")
    confu = confusion_matrix(y_test, pred).T
    return {"acc": acc, "pre": precision, "recall": recall, "F1": f1, "confu": confu}


def save_outputs(
    cfg: DatasetConfig,
    dataset_num: str,
    model: DecisionTreeClassifier,
    pred: np.ndarray,
    prob_test: np.ndarray,
    prob_train: np.ndarray,
) -> None:
    """
    Save .npy outputs with the same filenames as original script.
    """
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.log_dir / f"feature_{dataset_num}.npy", model.feature_importances_)
    np.save(cfg.log_dir / f"pred_{dataset_num}.npy", pred)
    np.save(cfg.log_dir / f"prob_test_{dataset_num}.npy", prob_test)
    np.save(cfg.log_dir / f"prob_train_{dataset_num}.npy", prob_train)


def main() -> None:
    cfg = DatasetConfig(dataset_num="34_3")
    replace_cfg = LabelReplaceConfig(old_label=4, new_label=3)
    model_cfg = ModelConfig(min_samples_leaf=40, criterion="entropy", random_state=42)

    # Load datasets
    train_samples, train_labels = load_torch_dataset(cfg.train_path())
    test_samples, test_labels = load_torch_dataset(cfg.test_path())

    # Prepare features/labels (kept identical logic)
    x_train, y_train = prepare_xy_classification_data(train_samples, train_labels, replace_cfg)
    x_test, y_test = prepare_xy_classification_data(test_samples, test_labels, replace_cfg)

    # Train
    model = train_decision_tree(x_train, y_train, model_cfg)

    # Predict + probabilities (kept)
    pred_test = model.predict(x_test)
    prob_train = model.predict_proba(x_train)
    prob_test = model.predict_proba(x_test)
    _pred_train = model.predict(x_train)  # kept (unused like original)

    # Metrics
    metrics = evaluate_model(model, x_test, y_test)
    print(metrics)

    # Save outputs
    save_outputs(
        cfg=cfg,
        dataset_num=cfg.dataset_num,
        model=model,
        pred=pred_test,
        prob_test=prob_test,
        prob_train=prob_train,
    )


if __name__ == "__main__":
    main()