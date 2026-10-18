"""Generate the docs diagrams (PRD Sections 14 and 18).

Architecture diagram (4 layers), ER diagram for the PostgreSQL schema, and
the batch-scoring workflow diagram, saved as PNG files in docs/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#EAF2FB", fontsize=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            fc=fc, ec="#2C5F8A", lw=1.2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, color="#2C5F8A", lw=1.4)
    )


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(5, 9.6, "Vantara Customer Intelligence Platform — Architecture",
            ha="center", fontsize=13, weight="bold")

    # Data layer
    _box(ax, 0.4, 7.0, 2.6, 1.6, "Data layer\nOnline Retail II (UCI)\nraw CSV/XLSX\n+ PostgreSQL\n(customers, transactions,\npredictions, segments)")
    # Processing layer
    _box(ax, 3.8, 7.0, 2.6, 1.6, "Processing layer\nPandas/NumPy pipeline\nclean → validate →\nfeature engineering\n(RFM, CLV, sequences)")
    # Modeling layer
    _box(ax, 7.2, 7.0, 2.4, 1.6, "Modeling layer\nchurn: LR/DT/RF/KNN/\nXGB/LGBM + ANN/LSTM\nCLV regression\nK-Means/GMM, autoencoder")
    _arrow(ax, 3.0, 7.8, 3.8, 7.8)
    _arrow(ax, 6.4, 7.8, 7.2, 7.8)

    # Serving layer
    _box(ax, 3.8, 3.6, 2.6, 1.4, "Serving layer\nFastAPI (REST)\nmodel service")
    _box(ax, 7.2, 3.6, 2.4, 1.4, "Streamlit dashboard\nsegments, leaderboard,\nSHAP panel, batch CSV")
    _arrow(ax, 7.8, 7.0, 5.8, 5.0)
    _arrow(ax, 6.4, 4.3, 7.2, 4.3)

    # DB feedback
    _box(ax, 0.4, 3.6, 2.6, 1.4, "Predictions store\nPostgreSQL / SQLite\nscored batches\nsegment assignments")
    _arrow(ax, 3.8, 4.3, 3.0, 4.3)
    _arrow(ax, 1.7, 5.0, 1.7, 7.0)

    # Batch scoring job
    _box(ax, 3.4, 0.8, 3.4, 1.2, "Batch scoring job\n(scheduled) — scores customers,\nwrites predictions to PostgreSQL",
         fc="#FDF3E3")
    _arrow(ax, 5.1, 3.6, 5.1, 2.0)

    plt.tight_layout()
    plt.savefig(DOCS / "architecture_diagram.png", dpi=150)
    plt.close()


def er_diagram() -> None:
    tables = {
        "customers": ["customer_id PK", "country", "segment FK→segments",
                      "engagement_score"],
        "transactions": ["transaction_id PK", "invoice", "stock_code",
                           "quantity", "price", "invoice_date",
                           "customer_id FK→customers"],
        "predictions": ["prediction_id PK", "customer_id FK→customers",
                        "churn_probability", "churn_label", "clv_estimate",
                        "model_name", "scored_at"],
        "segments": ["segment_id PK", "segment_label", "customers",
                     "churn_rate", "future_clv"],
    }
    positions = {
        "customers": (0.4, 5.2),
        "transactions": (0.4, 1.4),
        "predictions": (5.6, 5.2),
        "segments": (5.6, 1.4),
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(5, 7.7, "Entity–Relationship Diagram (predictions store)",
            ha="center", fontsize=13, weight="bold")
    for name, cols in tables.items():
        x, y = positions[name]
        h = 0.4 * (len(cols) + 1)
        _box(ax, x, y, 4.0, h, "", fc="white")
        ax.add_patch(FancyBboxPatch((x, y + h - 0.5), 4.0, 0.5,
                                    boxstyle="round,pad=0.02",
                                    fc="#2C5F8A", ec="#2C5F8A"))
        ax.text(x + 2.0, y + h - 0.25, name, ha="center", va="center",
                fontsize=10, color="white", weight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 0.15, y + h - 0.8 - 0.4 * i, c, fontsize=8.5,
                    family="monospace")
    _arrow(ax, 4.4, 6.4, 5.6, 6.4)   # customers → predictions
    _arrow(ax, 2.4, 3.6, 2.4, 5.2)   # transactions → customers
    _arrow(ax, 5.6, 2.6, 4.4, 2.6)   # segments → customers
    plt.tight_layout()
    plt.savefig(DOCS / "er_diagram.png", dpi=150)
    plt.close()


def workflow_diagram() -> None:
    steps = [
        "Load raw\nOnline Retail II",
        "Clean + validate\n(dedupe, returns,\nIQR, schema checks)",
        "Feature engineering\n(point-in-time cutoff,\nRFM + behaviour)",
        "Train models\n6 classical + ANN\nLSTM + autoencoder",
        "Evaluate\nAUC / recall / F1\nsingle test pass",
        "Explain\nSHAP + LIME + PDP",
        "Serve\nFastAPI + Streamlit\nbatch scoring → DB",
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.text(7, 2.8, "End-to-end ML workflow", ha="center", fontsize=12,
            weight="bold")
    for i, s in enumerate(steps):
        x = 0.2 + i * 2.0
        _box(ax, x, 0.8, 1.7, 1.2, s, fontsize=8)
        if i < len(steps) - 1:
            _arrow(ax, x + 1.7, 1.4, x + 2.0, 1.4)
    plt.tight_layout()
    plt.savefig(DOCS / "workflow_diagram.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    architecture_diagram()
    er_diagram()
    workflow_diagram()
    print(f"Diagrams written to {DOCS}")
