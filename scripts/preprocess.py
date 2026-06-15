"""
Script de preprocesamiento de datos — Red Wine Quality.

Descarga el dataset desde UCI, binariza el target, escala las features
y persiste los conjuntos de train/test en formato Parquet.

Uso:
    python scripts/preprocess.py
    python scripts/preprocess.py --threshold 6 --test-size 0.2 --random-state 42
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    DATA_URL,
    FEATURE_COLS,
    create_binary_target,
    load_data,
    scale_features,
    split_features_target,
)

DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")


def main(threshold: int = 6, test_size: float = 0.2, random_state: int = 42) -> None:
    """Ejecuta el pipeline completo de preprocesamiento.

    Args:
        threshold: Umbral de calidad para binarización (default: 6).
        test_size: Proporción del conjunto de prueba (default: 0.2).
        random_state: Semilla para reproducibilidad (default: 42).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"[1/5] Cargando datos desde UCI...")
    df = load_data(DATA_URL)
    print(f"      Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    raw_path = os.path.join(DATA_DIR, "winequality-red.csv")
    df.to_csv(raw_path, index=False, sep=";")
    print(f"      Datos crudos guardados en: {raw_path}")

    print(f"\n[2/5] Binarizando target (threshold={threshold})...")
    df = create_binary_target(df, "quality", threshold)
    class_dist = df["quality_binary"].value_counts().to_dict()
    print(f"      Distribución de clases: {class_dist}")

    print("\n[3/5] Dividiendo features y target...")
    X, y = split_features_target(df, "quality_binary", FEATURE_COLS)

    print(f"\n[4/5] Split train/test (test_size={test_size}, stratify=True)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, scaler_path)
    print(f"\n[5/5] Escalando features (StandardScaler)...")
    print(f"      Escalador guardado en: {scaler_path}")

    train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    train_df["quality_binary"] = y_train.values
    train_path = os.path.join(DATA_DIR, "preprocessed_train.parquet")
    train_df.to_parquet(train_path, index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)
    test_df["quality_binary"] = y_test.values
    test_path = os.path.join(DATA_DIR, "preprocessed_test.parquet")
    test_df.to_parquet(test_path, index=False)

    print(f"\n✓ Train guardado en : {train_path}")
    print(f"✓ Test  guardado en : {test_path}")
    print("\nPreprocesamiento completado con éxito.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento del dataset Red Wine Quality (UCI)."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=6,
        help="Umbral de calidad para binarización (default: 6).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de datos de prueba (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42).",
    )
    args = parser.parse_args()
    main(args.threshold, args.test_size, args.random_state)
