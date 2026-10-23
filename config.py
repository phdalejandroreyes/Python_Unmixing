
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    """Carga la configuración del proyecto."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuración inválida en: {CONFIG_PATH}"
        )

    return config
