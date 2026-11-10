"""Accès rapide à la configuration (config.yaml)."""
from src.data import load_config


def leaky_columns() -> list[str]:
    return load_config()["leaky_columns"]
