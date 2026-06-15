"""
Módulo de preprocesamiento de datos para clasificación de calidad de vino.

Provee funciones para cargar, transformar y persistir datos del dataset
Red Wine Quality (UCI Machine Learning Repository).
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

FEATURE_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Carga el dataset Red Wine Quality desde una URL o ruta local.

    Args:
        url: URL o ruta al archivo CSV con separador ';'.

    Returns:
        DataFrame con los datos crudos.
    """
    df = pd.read_csv(url, sep=";")
    return df


def create_binary_target(
    df: pd.DataFrame,
    target_col: str = "quality",
    threshold: int = 6,
    new_col: str = "quality_binary",
) -> pd.DataFrame:
    """Crea una columna de target binario a partir de la calidad numérica.

    Args:
        df: DataFrame de entrada.
        target_col: Nombre de la columna de calidad original.
        threshold: Umbral de calidad. Valores >= threshold → 1 (buena calidad).
        new_col: Nombre de la nueva columna binaria.

    Returns:
        DataFrame con la columna binaria añadida.
    """
    df = df.copy()
    df[new_col] = (df[target_col] >= threshold).astype(int)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "quality_binary",
    feature_cols: Optional[list] = None,
):
    """Separa el DataFrame en matriz de features X y vector de target y.

    Args:
        df: DataFrame con features y target.
        target_col: Nombre de la columna target.
        feature_cols: Lista de columnas a usar como features. Si es None
            usa FEATURE_COLS por defecto.

    Returns:
        Tupla (X, y) como DataFrames/Series de pandas.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Escala features con StandardScaler ajustado solo en train.

    Args:
        X_train: Features de entrenamiento.
        X_test: Features de prueba.
        save_path: Ruta para persistir el escalador (.pkl). Opcional.

    Returns:
        Tupla (X_train_scaled, X_test_scaled, scaler) como arrays numpy.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    url: str = DATA_URL,
    target_col: str = "quality",
    threshold: int = 6,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_path: Optional[str] = None,
):
    """Pipeline completo de preprocesamiento desde URL hasta arrays listos para ML.

    Args:
        url: Fuente del dataset.
        target_col: Columna de calidad original.
        threshold: Umbral de binarización.
        test_size: Proporción del conjunto de prueba (0 < test_size < 1).
        random_state: Semilla para reproducibilidad.
        scaler_path: Ruta para guardar el escalador. Opcional.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    df = load_data(url)
    df = create_binary_target(df, target_col, threshold)
    X, y = split_features_target(df, "quality_binary")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, save_path=scaler_path
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, FEATURE_COLS
