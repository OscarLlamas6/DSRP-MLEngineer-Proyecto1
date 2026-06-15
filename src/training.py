"""
Módulo de entrenamiento y persistencia de modelos de Machine Learning.

Soporta Regresión Logística, Árbol de Decisión y Random Forest
mediante una interfaz unificada.
"""

import os
import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


AVAILABLE_MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def get_model(model_name: str, **kwargs) -> Any:
    """Instancia un modelo por nombre.

    Args:
        model_name: Clave del modelo. Opciones: 'logistic_regression',
            'decision_tree', 'random_forest'.
        **kwargs: Hiperparámetros pasados directamente al constructor del modelo.

    Returns:
        Instancia del modelo scikit-learn sin entrenar.

    Raises:
        ValueError: Si model_name no está en AVAILABLE_MODELS.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. "
            f"Opciones: {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model_name](**kwargs)


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Entrena un modelo scikit-learn.

    Args:
        model: Instancia del modelo sin entrenar.
        X_train: Array de features de entrenamiento.
        y_train: Array del target de entrenamiento.

    Returns:
        Modelo entrenado.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: str) -> None:
    """Persiste un modelo entrenado como archivo .pkl.

    Args:
        model: Modelo scikit-learn entrenado.
        path: Ruta completa del archivo de salida (.pkl).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {path}")


def load_model(path: str) -> Any:
    """Carga un modelo desde un archivo .pkl.

    Args:
        path: Ruta al archivo .pkl del modelo.

    Returns:
        Modelo scikit-learn deserializado.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
