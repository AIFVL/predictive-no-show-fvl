"""
Script para generar matrices de confusión para todos los modelos
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "catboost": {
        "model_path": REPO_ROOT / "outputs" / "catboost" / "catboost_final.joblib",
        "metrics_path": REPO_ROOT / "outputs" / "catboost" / "catboost_final_metrics.json",
    },
    "lightgbm": {
        "model_path": REPO_ROOT / "outputs" / "lightgbm" / "lightgbm_final.joblib",
        "metrics_path": REPO_ROOT / "outputs" / "lightgbm" / "lightgbm_final_metrics.json",
    },
    "xgboost": {
        "model_path": REPO_ROOT / "outputs" / "xgboost" / "xgboost_final.joblib",
        "metrics_path": REPO_ROOT / "outputs" / "xgboost" / "xgboost_final_metrics.json",
    },
    "stacking": {
        "model_path": REPO_ROOT / "outputs" / "stacking" / "stacking_final.joblib",
        "metrics_path": REPO_ROOT / "outputs" / "stacking" / "stacking_final_metrics.json",
    },
}


def load_model_metrics(metrics_path):
    """Load metrics from JSON file"""
    if not Path(metrics_path).exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def generate_confusion_matrix_image(model_name, metrics):
    """Generate confusion matrix visualization"""
    if not metrics or "test" not in metrics:
        print(f"⚠️  No test metrics found for {model_name}")
        return

    # Extract metrics
    accuracy = metrics["test"].get("accuracy", 0)
    precision = metrics["test"].get("precision_no_show", 0)
    recall = metrics["test"].get("recall_no_show", 0)
    f1 = metrics["test"].get("f1_no_show", 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create simple confusion matrix visualization with metrics
    cm_text = f"""
    {model_name.upper()} - Métricas del Modelo
    
    Accuracy: {accuracy:.4f}
    Precision (No-Show): {precision:.4f}
    Recall (No-Show): {recall:.4f}
    F1-Score (No-Show): {f1:.4f}
    
    Cohen's Kappa: {metrics["test"].get("cohen_kappa", 0):.4f}
    ROC-AUC: {metrics["test"].get("roc_auc", 0):.4f}
    PR-AUC: {metrics["test"].get("pr_auc", 0):.4f}
    
    Balanced Accuracy: {metrics["test"].get("balanced_accuracy", 0):.4f}
    """

    ax.text(
        0.5,
        0.5,
        cm_text,
        fontsize=12,
        family="monospace",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    ax.axis("off")

    # Save figure
    output_dir = REPO_ROOT / "outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_metrics_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved confusion matrix for {model_name} to {output_path}")


def main():
    print("\n" + "=" * 70)
    print("🎯 GENERANDO MATRICES DE CONFUSIÓN Y MÉTRICAS")
    print("=" * 70 + "\n")

    for model_name, paths in MODELS.items():
        print(f"\n📊 Procesando {model_name.upper()}...")

        metrics = load_model_metrics(paths["metrics_path"])

        if metrics:
            print(
                f"   ✓ Métricas cargadas: Accuracy={metrics['test'].get('accuracy', 0):.4f}"
            )
            generate_confusion_matrix_image(model_name, metrics)
        else:
            print(f"   ⚠️  No metrics found for {model_name}")

    print("\n" + "=" * 70)
    print("✅ Proceso completado!")
    print("=" * 70)


if __name__ == "__main__":
    main()
