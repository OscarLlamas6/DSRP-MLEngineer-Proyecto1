"""
Módulo de código reutilizable para el proyecto de clasificación de calidad de vino.

Submódulos:
    preprocessing: Carga, limpieza y transformación de datos.
    training: Entrenamiento y persistencia de modelos.
    evaluation: Métricas y visualizaciones de rendimiento.
    pipeline: Pipeline OOP que encapsula el flujo completo (POO + SOLID).
"""

from src.preprocessing import load_data, create_binary_target, split_features_target, scale_features, preprocess_pipeline
from src.training import get_model, train_model, save_model, load_model
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, compare_models
from src.pipeline import WineQualityPipeline

__all__ = [
    "load_data",
    "create_binary_target",
    "split_features_target",
    "scale_features",
    "preprocess_pipeline",
    "get_model",
    "train_model",
    "save_model",
    "load_model",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "compare_models",
    "WineQualityPipeline",
]
