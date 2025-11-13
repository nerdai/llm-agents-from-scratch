"""Prepare book figures.

This Python script loads the `book_figures.yml` and a directory where the book
figures can be found, and creates copies of each figure, using the naming
convention required for Manning books.
"""

import shutil
import subprocess
from pathlib import Path

import fire
import yaml

FIGURES_YAML_FILENAME = "book_figures.yml"
DEFAULT_CONFIG_PATH = Path(__file__).parents[1].absolute()


def _get_image_id(chapter: str, fig_num: str) -> str:
    if int(fig_num) < 10:  # noqa: PLR2004
        return f"{chapter.title()}_F0{fig_num}_Fajardo"
    return f"{chapter.title()}_F{fig_num}_Fajardo"


def main(
    figures_path: Path | str,
    config_path: Path | None = None,
    include_svg: bool = False,
) -> None:
    """Main runner.

    Create .png and .svg copies of book figures for every chapter using the
    naming convention as specified by Manning.

    Args:
        figures_path (Path | str): Path to where book figures are located.
        config_path (Path | None, optional): Path to where `book_figures.yml`
        include_svg (bool): Whether or not to include .svg version of images

    Examples:
        >>> uv run python scripts/prepare_book_figures.py --figures_path "./"
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    with open(config_path / FIGURES_YAML_FILENAME, "r") as f:
        cfg = yaml.safe_load(f)

    if isinstance(figures_path, str):
        figures_path = Path(figures_path)

    # loop through all figures
    output_path = figures_path / "all_chapters"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)
    cp_commands = []
    for ch, data in cfg.items():
        # png
        cp_commands.extend(
            [
                "cp",
                f"{(figures_path / ch / fig_name).as_posix()}.png",
                f"{output_path / _get_image_id(ch, fig_num)}.png",
            ]
            for fig_num, fig_name in data["figures"].items()
        )

        # svg
        if include_svg:
            cp_commands.extend(
                [
                    "cp",
                    f"{(figures_path / ch / fig_name).as_posix()}.svg",
                    f"{output_path / _get_image_id(ch, fig_num)}.svg",
                ]
                for fig_num, fig_name in data["figures"].items()
                if fig_name not in data["png_only"]
            )

    for cmd in cp_commands:
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    fire.Fire(main)
