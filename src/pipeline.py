"""
Pipeline orientado a objetos para clasificación de calidad de vino.

Aplica principios SOLID:
- SRP (Single Responsibility): cada método tiene una sola responsabilidad.
- OCP (Open/Closed): el modelo se inyecta en el constructor; agregar nuevos
  modelos no requiere modificar la clase.
- DIP (Dependency Inversion): depende de la abstracción scikit-learn (fit/predict),
  no de implementaciones concretas.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.preprocessing import (
    DATA_URL,
    FEATURE_COLS,
    create_binary_target,
    load_data,
    scale_features,
    split_features_target,
)
from src.evaluation import evaluate_model
from src.training import save_model, train_model


class WineQualityPipeline:
    """Pipeline completo de clasificación de calidad de vino tinto.

    Encapsula la carga, preprocesamiento, entrenamiento y predicción en
    una interfaz unificada. El modelo de clasificación se inyecta en el
    constructor (Dependency Inversion Principle).

    Attributes:
        model: Estimador scikit-learn a utilizar.
        threshold: Umbral de binarización del target (default 6).
        test_size: Proporción del conjunto de test (default 0.2).
        random_state: Semilla aleatoria (default 42).
        is_fitted: True si el pipeline ha sido entrenado.
        feature_names: Nombres de las features del modelo.
        metrics_: Diccionario con métricas del último entrenamiento.
    """

    def __init__(
        self,
        model: Any,
        threshold: int = 6,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Inicializa el pipeline con un modelo inyectado.

        Args:
            model: Instancia de estimador scikit-learn sin entrenar.
            threshold: Umbral para binarizar la calidad (default 6).
            test_size: Fracción de datos para test (default 0.2).
            random_state: Semilla para reproducibilidad (default 42).
        """
        self.model = model
        self.threshold = threshold
        self.test_size = test_size
        self.random_state = random_state
        self.is_fitted: bool = False
        self.feature_names: List[str] = FEATURE_COLS
        self.metrics_: Dict[str, float] = {}
        self._scaler = None

    def load_and_preprocess(
        self, url: str = DATA_URL
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carga y preprocesa el dataset desde una URL o ruta local.

        Args:
            url: Fuente del dataset (URL o ruta).

        Returns:
            Tupla (X_train, X_test, y_train, y_test) como arrays numpy.
        """
        from sklearn.model_selection import train_test_split

        df = load_data(url)
        df = create_binary_target(df, "quality", self.threshold)
        df = df.drop_duplicates().reset_index(drop=True)

        X, y = split_features_target(df, "quality_binary", self.feature_names)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        self._scaler = scaler
        return X_train_sc, X_test_sc, y_train.values, y_test.values

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "WineQualityPipeline":
        """Entrena el modelo con los datos proporcionados.

        Args:
            X_train: Features de entrenamiento escaladas.
            y_train: Target de entrenamiento.

        Returns:
            La instancia del pipeline (para encadenamiento de métodos).
        """
        self.model = train_model(self.model, X_train, y_train)
        self.is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evalúa el modelo en el conjunto de test.

        Args:
            X_test: Features de test escaladas.
            y_test: Target real de test.

        Returns:
            Diccionario con métricas de evaluación.

        Raises:
            RuntimeError: Si el pipeline no ha sido entrenado.
        """
        if not self.is_fitted:
            raise RuntimeError("El pipeline no ha sido entrenado. Llama a fit() primero.")
        self.metrics_ = evaluate_model(self.model, X_test, y_test)
        return self.metrics_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones binarias para nuevas muestras.

        Args:
            X: Array de features (ya escalado).

        Returns:
            Array de predicciones binarias (0 o 1).

        Raises:
            RuntimeError: Si el pipeline no ha sido entrenado.
        """
        if not self.is_fitted:
            raise RuntimeError("El pipeline no ha sido entrenado. Llama a fit() primero.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Genera probabilidades de predicción (clase positiva).

        Args:
            X: Array de features (ya escalado).

        Returns:
            Array de probabilidades para la clase positiva.

        Raises:
            RuntimeError: Si el pipeline no ha sido entrenado.
            AttributeError: Si el modelo no soporta predict_proba.
        """
        if not self.is_fitted:
            raise RuntimeError("El pipeline no ha sido entrenado. Llama a fit() primero.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{type(self.model).__name__} no soporta predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, artifacts_dir: str) -> Dict[str, str]:
        """Persiste el modelo y el escalador en el directorio de artefactos.

        Args:
            artifacts_dir: Directorio donde guardar los artefactos.

        Returns:
            Diccionario con las rutas guardadas.

        Raises:
            RuntimeError: Si el pipeline no ha sido entrenado.
        """
        if not self.is_fitted:
            raise RuntimeError("El pipeline no ha sido entrenado. Llama a fit() primero.")

        os.makedirs(artifacts_dir, exist_ok=True)
        model_name = type(self.model).__name__.lower()
        paths: Dict[str, str] = {}

        model_path = os.path.join(artifacts_dir, f"{model_name}_pipeline.pkl")
        save_model(self.model, model_path)
        paths["model"] = model_path

        if self._scaler is not None:
            scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self._scaler, f)
            paths["scaler"] = scaler_path

        if self.metrics_:
            metrics_path = os.path.join(artifacts_dir, f"{model_name}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_, f, indent=2)
            paths["metrics"] = metrics_path

        return paths

    def run(self, url: str = DATA_URL, artifacts_dir: Optional[str] = None) -> Dict[str, float]:
        """Ejecuta el pipeline completo: carga → entrena → evalúa → guarda.

        Args:
            url: Fuente del dataset.
            artifacts_dir: Directorio para guardar artefactos. Opcional.

        Returns:
            Diccionario con métricas de evaluación.
        """
        X_train, X_test, y_train, y_test = self.load_and_preprocess(url)
        self.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)

        if artifacts_dir:
            self.save(artifacts_dir)

        return metrics

    def __repr__(self) -> str:
        status = "entrenado" if self.is_fitted else "sin entrenar"
        return (
            f"WineQualityPipeline("
            f"model={type(self.model).__name__}, "
            f"threshold={self.threshold}, "
            f"status={status})"
        )
