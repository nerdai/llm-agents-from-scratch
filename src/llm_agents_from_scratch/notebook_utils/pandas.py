# ruff: noqa: PLC0415
"""Pandas notebook utils."""

from ..utils import check_extra_was_installed


def set_dataframe_display_options() -> None:
    """Formatter to display pd.DataFrames in notebooks."""
    check_extra_was_installed(
        extra="notebook-utils",
        packages=["pandas", "IPython"],
    )

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
