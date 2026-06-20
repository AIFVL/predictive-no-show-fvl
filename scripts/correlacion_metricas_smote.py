"""
Script para generar matrices de correlación de métricas entre modelos
y visualizar comparativas de resultados con SMOTE
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "catboost": REPO_ROOT / "outputs" / "catboost" / "catboost_final_metrics.json",
    "lightgbm": REPO_ROOT / "outputs" / "lightgbm" / "lightgbm_final_metrics.json",
    "xgboost": REPO_ROOT / "outputs" / "xgboost" / "xgboost_final_metrics.json",
    "stacking": REPO_ROOT / "outputs" / "stacking" / "stacking_final_metrics.json",
}

METRICS_TO_COMPARE = [
    "accuracy",
    "balanced_accuracy",
    "f1_no_show",
    "precision_no_show",
    "recall_no_show",
    "cohen_kappa",
    "roc_auc",
    "pr_auc",
]


def load_all_metrics():
    """Load metrics from all models"""
    all_metrics = {}

    for model_name, metrics_path in MODELS.items():
        if not Path(metrics_path).exists():
            print(f"  No metrics found for {model_name}")
            continue

        with open(metrics_path, "r") as f:
            data = json.load(f)
            all_metrics[model_name] = data.get("test", {})

    return all_metrics


def create_correlation_matrix(all_metrics):
    """Create a correlation matrix of metrics across models"""

    # Create a dataframe with metrics
    df_data = {}
    for model_name, metrics in all_metrics.items():
        df_data[model_name] = {
            metric: metrics.get(metric, 0) for metric in METRICS_TO_COMPARE
        }

    df = pd.DataFrame(df_data).T

    # Create correlation matrix between models based on metrics
    # This shows how similar the models' performance are
    correlation = df.T.corr()

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Correlation"},
        ax=ax,
        vmin=-1,
        vmax=1,
    )

    ax.set_title(
        "Matriz de Correlación de Modelos (Basado en Métricas de Test con SMOTE)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Modelos", fontsize=12)
    ax.set_ylabel("Modelos", fontsize=12)

    plt.tight_layout()

    # Save figure
    output_path = REPO_ROOT / "outputs" / "models_correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Matriz de correlación guardada en: {output_path}")

    return correlation


def create_metrics_comparison(all_metrics):
    """Create side-by-side comparison of metrics"""

    # Create dataframe
    df_data = {}
    for model_name, metrics in all_metrics.items():
        df_data[model_name] = {
            metric: metrics.get(metric, 0) for metric in METRICS_TO_COMPARE
        }

    df = pd.DataFrame(df_data)

    # Create visualization with multiple subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    for idx, metric in enumerate(METRICS_TO_COMPARE):
        ax = axes[idx]
        values = df.loc[metric]

        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Comparación de Métricas - Todos los Modelos con SMOTE",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save figure
    output_path = REPO_ROOT / "outputs" / "metrics_comparison_all_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Gráfico de comparación guardado en: {output_path}")

    return df


def create_detailed_summary(all_metrics):
    """Create a detailed summary of all metrics"""

    df_data = {}
    for model_name, metrics in all_metrics.items():
        df_data[model_name] = {
            metric: metrics.get(metric, 0) for metric in METRICS_TO_COMPARE
        }

    df = pd.DataFrame(df_data).T

    print("\n" + "=" * 100)
    print(" RESUMEN DE MÉTRICAS - TODOS LOS MODELOS CON SMOTE")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100 + "\n")

    # Save to CSV
    csv_path = REPO_ROOT / "outputs" / "metrics_summary_smote.csv"
    df.to_csv(csv_path)
    print(f" Resumen guardado en CSV: {csv_path}\n")

    return df


def main():
    print("\n" + "=" * 70)
    print(" GENERANDO MATRICES DE CORRELACIÓN DE MÉTRICAS")
    print("=" * 70 + "\n")

    # Load all metrics
    all_metrics = load_all_metrics()

    if not all_metrics:
        print(" No se encontraron métricas para ningún modelo")
        return

    print(f" Métricas cargadas para {len(all_metrics)} modelos\n")

    # Create correlation matrix
    correlation = create_correlation_matrix(all_metrics)

    # Create metrics comparison
    df_comparison = create_metrics_comparison(all_metrics)

    # Create detailed summary
    df_summary = create_detailed_summary(all_metrics)

    # Print best model for each metric
    print("\n" + "=" * 70)
    print(" MEJORES MODELOS POR MÉTRICA (CON SMOTE)")
    print("=" * 70 + "\n")

    for metric in METRICS_TO_COMPARE:
        best_model = df_comparison.loc[metric].idxmax()
        best_value = df_comparison.loc[metric].max()
        print(f"  {metric:.<30} {best_model:.<15} {best_value:.4f}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
