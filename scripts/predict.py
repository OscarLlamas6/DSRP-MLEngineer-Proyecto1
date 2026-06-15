"""
Script de predicción usando modelo entrenado.

Carga un modelo y escalador persistidos, procesa el CSV de entrada
y genera predicciones de calidad de vino.

Uso:
    python scripts/predict.py --input data/winequality-red.csv
    python scripts/predict.py --input data/sample.csv --model artifacts/random_forest.pkl
"""

import argparse
import os
import pickle
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import FEATURE_COLS

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")


def main(
    input_path: str,
    model_path: str = None,
    scaler_path: str = None,
    output_path: str = None,
) -> None:
    """Realiza predicciones de calidad de vino sobre un CSV de entrada.

    Args:
        input_path: Ruta al CSV de entrada (separador ';').
        model_path: Ruta al modelo .pkl. Default: artifacts/random_forest.pkl.
        scaler_path: Ruta al escalador .pkl. Default: artifacts/scaler.pkl.
        output_path: Ruta de salida para las predicciones. Default: auto-generada.
    """
    if model_path is None:
        model_path = os.path.join(ARTIFACTS_DIR, "random_forest.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

    for path, name in [(model_path, "Modelo"), (scaler_path, "Escalador")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} no encontrado en '{path}'. "
                "Ejecuta primero: python scripts/train.py"
            )

    print(f"[1/4] Cargando modelo desde: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"[2/4] Cargando escalador desde: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[3/4] Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path, sep=";")
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    labels = pd.Series(predictions).map({1: "Buena calidad", 0: "Mala calidad"})

    df["prediction"] = predictions
    df["prediction_label"] = labels.values

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
        df["prob_buena_calidad"] = probabilities

    print("\nPrimeras 10 predicciones:")
    cols_show = ["prediction_label"] + (
        ["prob_buena_calidad"] if "prob_buena_calidad" in df.columns else []
    )
    print(df[cols_show].head(10).to_string(index=True))

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predictions.csv"

    df.to_csv(output_path, index=False)
    print(f"\n[4/4] Predicciones guardadas en: {output_path}")
    print(f"\nResumen: {(predictions == 1).sum()} vinos de buena calidad / "
          f"{(predictions == 0).sum()} vinos de mala calidad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicción de calidad de vino tinto con modelo entrenado."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo CSV de entrada (separador ';').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al modelo .pkl (default: artifacts/random_forest.pkl).",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="Ruta al escalador .pkl (default: artifacts/scaler.pkl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida para predicciones CSV.",
    )
    args = parser.parse_args()
    main(args.input, args.model, args.scaler, args.output)
