"""
Script de entrenamiento de modelos de Machine Learning.

Carga los datos preprocesados, entrena el modelo seleccionado,
lo evalúa y persiste el modelo y sus métricas.

Uso:
    python scripts/train.py
    python scripts/train.py --model random_forest --random-state 42
    python scripts/train.py --model logistic_regression
    python scripts/train.py --model decision_tree
"""

import argparse
import json
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.evaluation import evaluate_model
from src.training import get_model, save_model, train_model

DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")


def main(model_name: str = "random_forest", random_state: int = 42) -> None:
    """Entrena y evalúa un modelo de clasificación de calidad de vino.

    Args:
        model_name: Identificador del modelo a entrenar.
            Opciones: 'logistic_regression', 'decision_tree', 'random_forest'.
        random_state: Semilla aleatoria para reproducibilidad.
    """
    train_path = os.path.join(DATA_DIR, "preprocessed_train.parquet")
    test_path = os.path.join(DATA_DIR, "preprocessed_test.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Datos preprocesados no encontrados. "
            "Ejecuta primero: python scripts/preprocess.py"
        )

    print(f"[1/4] Cargando datos preprocesados...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=["quality_binary"]).values
    y_train = train_df["quality_binary"].values
    X_test = test_df.drop(columns=["quality_binary"]).values
    y_test = test_df["quality_binary"].values
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    print(f"\n[2/4] Entrenando modelo: {model_name}...")
    model = get_model(model_name, random_state=random_state)
    model = train_model(model, X_train, y_train)
    print("      Entrenamiento completado.")

    print("\n[3/4] Evaluando modelo en conjunto de prueba...")
    metrics = evaluate_model(model, X_test, y_test, model_name)
    print(f"      {'Métrica':<15} {'Valor':>8}")
    print(f"      {'-'*25}")
    for metric, value in metrics.items():
        print(f"      {metric:<15} {value:>8.4f}")

    print("\n[4/4] Guardando modelo y métricas...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, f"{model_name}.pkl")
    save_model(model, model_path)

    metrics_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Métricas guardadas en: {metrics_path}")

    print("\nEntrenamiento completado con éxito.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos ML para clasificación de calidad de vino."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic_regression", "decision_tree", "random_forest"],
        help="Modelo a entrenar (default: random_forest).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42).",
    )
    args = parser.parse_args()
    main(args.model, args.random_state)
