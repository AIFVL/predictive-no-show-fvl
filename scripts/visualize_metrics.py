import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# =========================
# Cargar datos desde JSON
# =========================

def load_model_metrics(model_name):
    """Carga las métricas de un modelo desde su archivo JSON"""
    json_path = f"outputs/{model_name}/{model_name}_final_metrics.json"
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        return metrics['test']  # Usando métricas de test
    else:
        print(f"Advertencia: No se encontró {json_path}")
        return None

# Cargar datos de todos los modelos
models = ['xgboost', 'lightgbm', 'stacking', 'catboost']
model_labels = ['XGBoost', 'LightGBM', 'Stacking', 'CatBoost']

# Diccionario para almacenar los datos
data = {
    'Model': model_labels,
    'Accuracy': [],
    'Recall': [],
    'F1-Score': [],
    'Cohen': [],
    'ROC-AUC': []
}

# Cargar métricas de cada modelo
for model in models:
    metrics = load_model_metrics(model)
    if metrics:
        data['Accuracy'].append(metrics['accuracy'])
        data['Recall'].append(metrics['recall_no_show'])
        data['F1-Score'].append(metrics['f1_no_show'])
        data['Cohen'].append(metrics['cohen_kappa'])
        data['ROC-AUC'].append(metrics['roc_auc'])

df = pd.DataFrame(data)

print("\n=========================")
print("Métricas de Modelos (Test)")
print("=========================\n")
print(df.to_string(index=False))
print("\n")

# =========================
# Configuración gráfica
# =========================

metrics = ['Accuracy', 'Recall', 'F1-Score', 'Cohen', 'ROC-AUC']

x = np.arange(len(df['Model']))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 7))

# Colores elegantes/profesionales
colors = [
    '#4C72B0',  # azul
    '#55A868',  # verde
    '#C44E52',  # rojo
    '#8172B2',  # morado
    '#CCB974'   # dorado
]

# =========================
# Crear barras
# =========================

for i, metric in enumerate(metrics):
    bars = ax.bar(
        x + i * width,
        df[metric],
        width=width,
        label=metric,
        color=colors[i],
        edgecolor='black',
        linewidth=0.7
    )

    # Etiquetas encima de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

# =========================
# Estilo profesional
# =========================

ax.set_title(
    'Comparison of Model Performance Metrics',
    fontsize=18,
    fontweight='bold',
    pad=20
)

ax.set_xlabel(
    'Machine Learning Models',
    fontsize=13,
    labelpad=10
)

ax.set_ylabel(
    'Metric Score',
    fontsize=13,
    labelpad=10
)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(df['Model'], fontsize=11)

ax.set_ylim(0.55, 1.0)

# Grid elegante
ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.4
)

# Remover bordes innecesarios
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Leyenda
ax.legend(
    title='Metrics',
    fontsize=10,
    title_fontsize=11,
    frameon=True
)

# Línea objetivo
ax.axhline(
    y=0.8,
    color='red',
    linestyle='--',
    linewidth=2,
    alpha=0.8,
    label='Target Threshold (0.8)'
)

# Texto de referencia
ax.text(
    len(df['Model']) - 0.3,
    0.82,
    'Target = 0.8',
    color='red',
    fontsize=10,
    va='bottom'
)

plt.tight_layout()

# Guardar la gráfica
output_path = 'outputs/metrics_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Gráfica guardada en: {output_path}")

plt.show()
