# ruff: noqa: PLC0415
"""Pandas notebook utils."""

from ._check_installation import check_notebooks_utils_installed


def display_dataframes() -> None:
    """Formatter to display pd.DataFrames in notebooks."""
    check_notebooks_utils_installed()

    import pandas as pd
    from IPython.display import HTML, display

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    display(
        HTML("""
    <style>
    .output_scroll {
        overflow-x: scroll;
    }
    table.dataframe {
        white-space: nowrap;
    }
    </style>
    """),
    )
