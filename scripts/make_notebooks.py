"""Generate the three PRD notebooks as .ipynb files."""
from __future__ import annotations

import json
from pathlib import Path


def nb(cells: list[str]) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"c{i}",
                "metadata": {},
                "outputs": [],
                "source": c.splitlines(keepends=True),
            }
            for i, c in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


EDA = [
    (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from src.utils.config import load_config, resolve_path\n\n"
        "cfg = load_config()\n"
        "tx = pd.read_parquet(resolve_path(cfg.get('paths.interim_data')))\n"
        "print(f\"{len(tx):,} cleaned transactions, "
        "{tx['customer_id'].nunique():,} customers\")\n"
        "tx.head()"
    ),
    (
        "# Univariate: spend, frequency, recency distributions\n"
        "feats = pd.read_parquet(resolve_path(cfg.get('paths.processed_features')))\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, col in zip(axes, ['monetary_total', 'frequency', 'recency_days']):\n"
        "    feats[col].clip(upper=feats[col].quantile(0.99)).hist(bins=40, ax=ax)\n"
        "    ax.set_title(col)\n"
        "plt.tight_layout()"
    ),
    (
        "# Bivariate: features vs churn target\n"
        "import seaborn as sns\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, col in zip(axes, ['recency_days', 'frequency', 'monetary_total']):\n"
        "    sns.boxplot(data=feats, x='churn', y=feats[col].clip("
        "upper=feats[col].quantile(0.99)), ax=ax)\n"
        "    ax.set_title(col)\n"
        "plt.tight_layout()"
    ),
    (
        "# Hypotheses the modeling stage will test (PRD Section 6.4):\n"
        "# H1: recency_days is the dominant churn signal (strongest SHAP feature).\n"
        "# H2: short-window momentum (spend_last_30d/90d) adds signal beyond RFM.\n"
        "# H3: returns_ratio correlates positively with churn (dissatisfaction).\n"
        "# H4: country effects exist but are weaker than behavioural features.\n"
        "# H5: CLV is predictable for repeat customers, noisy for one-off buyers."
    ),
    (
        "# Multicollinearity check (VIF) before finalizing the feature set\n"
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from src.models.train_churn import get_feature_columns\n"
        "num_cols, _ = get_feature_columns(feats)\n"
        "X = StandardScaler().fit_transform(feats[num_cols].astype(float))\n"
        "vif = pd.DataFrame({'feature': num_cols,\n"
        "    'vif': [variance_inflation_factor(X, i) for i in range(len(num_cols))]})\n"
        "vif.sort_values('vif', ascending=False).head(10)"
    ),
]

FE = [
    (
        "from src.utils.config import load_config, resolve_path\n"
        "from src.features.build_features import build_feature_table\n"
        "cfg = load_config()\n"
        "tx = pd.read_parquet(resolve_path(cfg.get('paths.interim_data')))\n"
        "table = build_feature_table(tx, str(cfg.get('features.snapshot_date')),\n"
        "    int(cfg.get('features.churn_window_days')),\n"
        "    int(cfg.get('features.clv_window_days')))\n"
        "table.describe().T"
    ),
    (
        "# Leakage spot-check: features use pre-cutoff data only.\n"
        "# (Pinned permanently by tests/test_features.py.)\n"
        "from src.features.build_features import split_by_cutoff\n"
        "history, future = split_by_cutoff(tx, str(cfg.get('features.snapshot_date')))\n"
        "assert (history['invoice_date'] < future['invoice_date'].min()) or future.empty\n"
        "print('cutoff windows disjoint: OK')"
    ),
    (
        "# Feature-target correlation snapshot\n"
        "corr = table.drop(columns=['customer_id', 'country'])"
        ".corr(numeric_only=True)['churn'].sort_values()\n"
        "corr.head(12), corr.tail(12)"
    ),
]

ME = [
    (
        "import pandas as pd\n"
        "from src.utils.config import load_config, resolve_path\n"
        "cfg = load_config()\n"
        "art = resolve_path(cfg.get('paths.models_dir'))\n"
        "comp = pd.read_csv(art / 'model_comparison.csv')\n"
        "comp[['model', 'val_roc_auc', 'val_recall', 'test_roc_auc', "
        "'test_recall', 'train_seconds']]"
    ),
    (
        "import json\n"
        "metrics = json.loads((art / 'dl_metrics.json').read_text())\n"
        "pd.DataFrame(metrics).T"
    ),
    (
        "# Loss curves (ANN/LSTM): convergence & overfitting evidence\n"
        "import matplotlib.pyplot as plt\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "for ax, name in zip(axes, ['ann', 'lstm']):\n"
        "    lc = pd.read_csv(art / f'{name}_loss_curve.csv')\n"
        "    lc.plot(ax=ax); ax.set_title(f'{name.upper()} loss')\n"
        "    ax.set_xlabel('epoch')\n"
        "plt.tight_layout()"
    ),
    (
        "# Segmentation diagnostics\n"
        "k_sel = pd.read_csv(art / 'k_selection.csv')\n"
        "k_sel.plot(x='k', y=['inertia', 'silhouette'], "
        "secondary_y='silhouette', figsize=(7, 4))"
    ),
]


def main() -> None:
    out = Path("notebooks")
    out.mkdir(exist_ok=True)
    for name, cells in [
        ("01_eda", EDA),
        ("02_feature_engineering", FE),
        ("03_model_experiments", ME),
    ]:
        path = out / f"{name}.ipynb"
        path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
