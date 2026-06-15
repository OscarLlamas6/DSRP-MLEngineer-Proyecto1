"""
Módulo de evaluación de modelos de clasificación binaria.

Provee funciones para calcular métricas, graficar matrices de confusión,
curvas ROC y comparaciones entre múltiples modelos.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Calcula métricas de clasificación para un modelo entrenado.

    Args:
        model: Modelo scikit-learn entrenado.
        X_test: Features del conjunto de prueba.
        y_test: Labels reales del conjunto de prueba.
        model_name: Nombre del modelo para logging.

    Returns:
        Diccionario con accuracy, precision, recall, f1 y roc_auc.
    """
    y_pred = model.predict(X_test)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grafica la matriz de confusión del modelo.

    Args:
        model: Modelo scikit-learn entrenado.
        X_test: Features del conjunto de prueba.
        y_test: Labels reales.
        model_name: Título del gráfico.
        save_path: Ruta para guardar el gráfico (.png o .html). Opcional.

    Returns:
        Figura de matplotlib.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Mala calidad", "Buena calidad"],
        yticklabels=["Mala calidad", "Buena calidad"],
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.tight_layout()
    return fig


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Grafica la curva ROC del modelo.

    Args:
        model: Modelo scikit-learn entrenado (debe tener predict_proba).
        X_test: Features del conjunto de prueba.
        y_test: Labels reales.
        model_name: Nombre del modelo para la leyenda.
        save_path: Ruta para guardar el gráfico. Opcional.

    Returns:
        Figura de matplotlib, o None si el modelo no soporta predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        print(f"  {model_name} no soporta predict_proba. ROC no disponible.")
        return None

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    ax.set_title(f"Curva ROC — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.tight_layout()
    return fig


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Genera un gráfico de barras comparando métricas de múltiples modelos.

    Args:
        results: Diccionario {nombre_modelo: {métrica: valor}}.
        save_path: Ruta para guardar el gráfico. Opcional.

    Returns:
        Figura de matplotlib.
    """
    df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", ax=ax, rot=0, colormap="Set2", edgecolor="white", linewidth=0.5)
    ax.set_title("Comparación de Modelos — Métricas de Evaluación", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.35)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    plt.tight_layout()
    return fig
