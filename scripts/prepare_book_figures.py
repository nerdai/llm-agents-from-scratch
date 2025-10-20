"""Prepare book figures.

This Python script loads the `book_figures.yml` and a directory where the book
figures can be found, and creates copies of each figure, using the naming
convention required for Manning books.
"""

from pathlib import Path

import fire
import yaml

FIGURES_YAML_FILENAME = "book_figures.yml"
DEFAULT_CONFIG_PATH = Path(__file__).parents[1].absolute()


def main(figures_path: Path, config_path: Path | None = None) -> None:
    """Main runner.

    Create .png and .svg copies of book figures for every chapter using the
    naming convention as specified by Manning.

    Args:
        figures_path (Path): _description_
        config_path (Path | None, optional): _description_. Defaults to None.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    with open(config_path / FIGURES_YAML_FILENAME, "r") as f:
        cfg = yaml.safe_load(f)

    print(cfg)


if __name__ == "__main__":
    fire.Fire(main)
