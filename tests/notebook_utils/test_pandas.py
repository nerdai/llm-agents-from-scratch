"""Unit tests for pandas notebook utils."""

from unittest.mock import MagicMock, patch


@patch(
    "llm_agents_from_scratch.notebook_utils.pandas.check_extra_was_installed",
)
@patch("pandas.set_option")
@patch("IPython.display.display")
@patch("IPython.display.HTML")
def test_set_dataframe_display_options(
    mock_html: MagicMock,
    mock_display: MagicMock,
    mock_set_option: MagicMock,
    mock_check_extra: MagicMock,
) -> None:
    """Tests set_dataframe_display_options()."""
    from llm_agents_from_scratch.notebook_utils.pandas import (  # noqa: PLC0415
        set_dataframe_display_options,
    )

    expected_pd_set_option_call_count = 3

    # act
    set_dataframe_display_options()

    # assert
    mock_check_extra.assert_called_once_with(
        extra="notebook-utils",
        packages=["pandas", "IPython"],
    )
    assert mock_set_option.call_count == expected_pd_set_option_call_count
    mock_html.assert_called_once()
    mock_display.assert_called_once()
