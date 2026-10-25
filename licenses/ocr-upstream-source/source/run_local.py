"""PyCharm-friendly entry point for local WeVisDoc inference."""

from pathlib import Path

from wevisdoc.local import main_from_config


CONFIG_PATH = Path(__file__).with_name("inference_config.json")


if __name__ == "__main__":
    raise SystemExit(main_from_config(CONFIG_PATH))
