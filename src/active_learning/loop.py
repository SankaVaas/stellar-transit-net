"""
loop.py
-------
Active learning training loop.

Orchestrates the full AL experiment:
    1. Start with a small labelled seed set (10% of data)
    2. Train base model on labelled set
    3. Run inference on unlabelled pool
    4. Query oracle for labels on most informative samples
    5. Add newly labelled samples to training set
    6. Retrain from scratch (or fine-tune) and evaluate
    7. Record learning curve: AUPRC vs labels used
    8. Repeat for N iterations

The learning curve is the key deliverable — it shows how much
better the model gets per label spent, compared to random sampling.
A good query strategy achieves 95% of full-data performance using
only 30-40% of labels. That's the NASA-relevant result.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import mlflow
import yaml
from pathlib import Path
from copy import deepcopy

from .oracle import SimulatedOracle
from .query_strategies import max_entropy_query, bald_query, least_confident_query
from src.preprocessing.pipeline import TransitDataset, build_dataloaders
from src.models.cnn_tcn import CNNTCN

log = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auprc  = average_precision_score(labels, probs[:, 1])
    acc    = (probs.argmax(axis=1) == labels).mean()
    return {"auprc": auprc, "accuracy": acc, "probs": probs, "labels": labels}


def build_model(config: dict, device: str) -> nn.Module:
    """Instantiate a fresh CNN-TCN model from config."""
    cfg = config["cnn_tcn"]
    model = CNNTCN(
        in_channels=1,
        cnn_channels=cfg["cnn_channels"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        tcn_channels=cfg["tcn_channels"],
        tcn_kernel_size=cfg["tcn_kernel_size"],
        tcn_dropout=cfg["tcn_dropout"],
        num_classes=cfg["num_classes"],
        fc_hidden=cfg["fc_hidden"],
    )
    return model.to(device)


def run_active_learning(
    full_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    model_config: dict,
    train_config: dict,
    strategy: str = "max_entropy",
    n_iterations: int = 20,
    query_batch_size: int = 50,
    initial_fraction: float = 0.10,
    device: str = "cpu",
    experiment_name: str = "active_learning",
) -> pd.DataFrame:
    """
    Run the full active learning experiment and return a learning curve DataFrame.

    Args:
        full_manifest: complete dataset manifest (labelled + unlabelled pool)
        val_manifest: held-out validation set (never queried)
        test_manifest: held-out test set (never queried)
        model_config: model architecture config
        train_config: training hyperparameter config
        strategy: 'max_entropy' | 'bald' | 'least_confident' | 'random'
        n_iterations: number of AL rounds
        query_batch_size: labels to request per round
        initial_fraction: fraction of data labelled at start
        device: torch device

    Returns:
        learning_curve: DataFrame with columns
            [iteration, n_labeled, auprc_al, auprc_random, strategy]
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Initialise oracle with seed labels ───────────────────────────────────
    n_initial = int(len(full_manifest) * initial_fraction)
    # Stratified seed: equal positives and negatives
    pos_idx = full_manifest[full_manifest["label"] == 1].index.tolist()
    neg_idx = full_manifest[full_manifest["label"] == 0].index.tolist()
    n_each  = n_initial // 2
    seed_idx = np.array(
        np.random.choice(pos_idx, min(n_each, len(pos_idx)), replace=False).tolist() +
        np.random.choice(neg_idx, min(n_each, len(neg_idx)), replace=False).tolist()
    )

    oracle    = SimulatedOracle(full_manifest, seed_idx)
    # Random baseline oracle (same seed, queries randomly)
    oracle_rnd = SimulatedOracle(full_manifest.copy(), seed_idx.copy())

    al_cfg  = train_config["active_learning"]
    t_cfg   = train_config["training"]
    aug_cfg = train_config.get("augmentation", {})

    learning_curve_rows = []

    mlflow.set_experiment(experiment_name)

    for iteration in range(n_iterations + 1):
        log.info(f"\n{'='*60}")
        log.info(f"AL Iteration {iteration} | Strategy: {strategy} | Labeled: {oracle.n_labeled}")

        with mlflow.start_run(run_name=f"al_{strategy}_iter{iteration}", nested=True):
            mlflow.log_param("iteration", iteration)
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("n_labeled", oracle.n_labeled)

            # ── Train AL model ────────────────────────────────────────────────
            labeled_manifest = oracle.get_labeled_manifest()
            train_ds = TransitDataset(labeled_manifest, augment_data=aug_cfg.get("enabled", True), augment_config=aug_cfg)
            val_ds   = TransitDataset(val_manifest, augment_data=False)
            test_ds  = TransitDataset(test_manifest, augment_data=False)

            train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,  num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)

            model = build_model(model_config, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
            criterion = nn.CrossEntropyLoss()

            # Shorter training per AL iteration to keep runtime manageable
            epochs = min(t_cfg["epochs"], 20)
            best_val_auprc, best_state = 0.0, None
            for epoch in range(epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics = evaluate(model, val_loader, device)
                if val_metrics["auprc"] > best_val_auprc:
                    best_val_auprc = val_metrics["auprc"]
                    best_state = deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            test_metrics = evaluate(model, test_loader, device)
            mlflow.log_metric("test_auprc", test_metrics["auprc"])
            mlflow.log_metric("test_accuracy", test_metrics["accuracy"])

            # ── Train random baseline model ────────────────────────────────────
            rnd_labeled = oracle_rnd.get_labeled_manifest()
            rnd_train_ds = TransitDataset(rnd_labeled, augment_data=False)
            rnd_loader   = DataLoader(rnd_train_ds, batch_size=t_cfg["batch_size"], shuffle=True, num_workers=0)
            rnd_model    = build_model(model_config, device)
            rnd_optimizer = torch.optim.AdamW(rnd_model.parameters(), lr=t_cfg["lr"])
            rnd_best, rnd_state = 0.0, None
            for epoch in range(epochs):
                train_one_epoch(rnd_model, rnd_loader, rnd_optimizer, criterion, device)
                vm = evaluate(rnd_model, val_loader, device)
                if vm["auprc"] > rnd_best:
                    rnd_best = vm["auprc"]
                    rnd_state = deepcopy(rnd_model.state_dict())
            rnd_model.load_state_dict(rnd_state)
            rnd_metrics = evaluate(rnd_model, test_loader, device)

            learning_curve_rows.append({
                "iteration":    iteration,
                "n_labeled":    oracle.n_labeled,
                "auprc_al":     test_metrics["auprc"],
                "auprc_random": rnd_metrics["auprc"],
                "accuracy_al":  test_metrics["accuracy"],
                "strategy":     strategy,
            })
            log.info(f"Test AUPRC — AL: {test_metrics['auprc']:.4f} | Random: {rnd_metrics['auprc']:.4f}")

            # ── Query next batch (skip on last iteration) ──────────────────────
            if iteration < n_iterations:
                unlabeled_manifest = oracle.get_unlabeled_manifest()
                unlabeled_ds = TransitDataset(unlabeled_manifest, augment_data=False)
                unlabeled_loader = DataLoader(unlabeled_ds, batch_size=t_cfg["batch_size"], shuffle=False, num_workers=0)

                # Get probabilities for unlabelled pool
                pool_metrics = evaluate(model, unlabeled_loader, device)
                pool_probs   = pool_metrics["probs"]

                # Select query indices using chosen strategy
                if strategy == "max_entropy":
                    query_idx = max_entropy_query(pool_probs, query_batch_size)
                elif strategy == "least_confident":
                    query_idx = least_confident_query(pool_probs, query_batch_size)
                elif strategy == "random":
                    query_idx = np.random.choice(len(pool_probs), query_batch_size, replace=False)
                else:
                    query_idx = max_entropy_query(pool_probs, query_batch_size)

                oracle.query(query_idx, iteration=iteration)

                # Random baseline: always queries randomly
                rnd_query_idx = np.random.choice(
                    len(oracle_rnd.unlabeled_indices), query_batch_size, replace=False
                )
                oracle_rnd.query(rnd_query_idx, iteration=iteration)

    learning_curve = pd.DataFrame(learning_curve_rows)
    log.info("\nActive learning complete.")
    log.info(learning_curve[["iteration", "n_labeled", "auprc_al", "auprc_random"]].to_string())
    return learning_curve