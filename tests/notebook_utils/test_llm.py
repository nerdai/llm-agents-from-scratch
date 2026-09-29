"""Unit tests for LLM notebook utils."""

from unittest.mock import MagicMock, patch

import pytest

from llm_agents_from_scratch.errors import UnsupportedProviderError
from llm_agents_from_scratch.notebook_utils import llm as llm_mod
from llm_agents_from_scratch.notebook_utils.llm import (
    DEFAULT_OLLAMA_HOST,
    OLLAMA_CLOUD_HOST,
    ensure_ollama,
    make_llm,
    ollama_settings,
    resolve_provider,
)

ALL_KEYS = (
    "LLM_PROVIDER",
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from whatever the developer has exported."""
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_ollama_llm():
    """Patch OllamaLLM at its source, since make_llm imports it lazily."""
    with patch("llm_agents_from_scratch.llms.ollama.OllamaLLM") as mock:
        yield mock


@pytest.fixture
def mock_openai_llm():
    """Patch OpenAILLM at its source, since make_llm imports it lazily."""
    with patch("llm_agents_from_scratch.llms.openai.OpenAILLM") as mock:
        yield mock


@pytest.fixture
def mock_anthropic_llm():
    """Patch AnthropicLLM at its source, since make_llm imports it lazily."""
    with patch("llm_agents_from_scratch.llms.anthropic.AnthropicLLM") as mock:
        yield mock


@pytest.fixture
def mock_ensure():
    """Stub out the Ollama bootstrap."""
    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.ensure_ollama",
    ) as mock:
        yield mock


# -- resolve_provider ------------------------------------------------------


def test_resolve_provider_defaults_to_ollama() -> None:
    """With nothing set, the book's default provider is used."""
    assert resolve_provider() == "ollama"


def test_resolve_provider_explicit_arg_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit argument beats the env var."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    assert resolve_provider("openai") == "openai"


def test_resolve_provider_reads_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_PROVIDER flips every notebook at once."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert resolve_provider() == "openai"


def test_resolve_provider_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case and surrounding whitespace are forgiven."""
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")

    assert resolve_provider() == "openai"


def test_resolve_provider_rejects_unknown() -> None:
    """An unrecognized provider fails loudly rather than silently."""
    with pytest.raises(UnsupportedProviderError, match="Unknown LLM provider"):
        resolve_provider("gemini")


@pytest.mark.parametrize(
    "key",
    ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
)
def test_api_key_presence_does_not_select_provider(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key exported for unrelated reasons must not hijack the default.

    This is the core guarantee of the design: readers follow the book on
    Ollama even if they happen to have closed-provider keys in their shell.
    """
    monkeypatch.setenv(key, "sk-not-for-this-book")

    assert resolve_provider() == "ollama"


# -- make_llm, ollama path -------------------------------------------------


def test_make_llm_default_is_local_ollama(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """The zero-config path reproduces what the notebooks did before."""
    make_llm()

    mock_ensure.assert_called_once_with(DEFAULT_OLLAMA_HOST)
    mock_ollama_llm.assert_called_once_with(
        model="qwen3:14b",
        host=None,
        think=False,
    )


def test_make_llm_small_role(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """The small role maps to ch09's smaller local model."""
    make_llm(role="small")

    assert mock_ollama_llm.call_args.kwargs["model"] == "qwen3:8b"


def test_make_llm_ollama_cloud(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLLAMA_API_KEY routes to Ollama Cloud without starting a local server."""
    monkeypatch.setenv("OLLAMA_API_KEY", "key")

    make_llm()

    mock_ensure.assert_not_called()
    mock_ollama_llm.assert_called_once_with(
        model="kimi-k2.7-code:cloud",
        host=OLLAMA_CLOUD_HOST,
        json_prompt_mode=True,
        think=False,
    )


def test_make_llm_caller_kwargs_win_over_defaults(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """A notebook that wants thinking mode still gets it."""
    make_llm(think=True)

    assert mock_ollama_llm.call_args.kwargs["think"] is True


def test_make_llm_host_override(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """A custom host is both bootstrapped and passed through."""
    make_llm(host="http://elsewhere:11434")

    mock_ensure.assert_called_once_with("http://elsewhere:11434")
    assert mock_ollama_llm.call_args.kwargs["host"] == "http://elsewhere:11434"


# -- ollama_settings -------------------------------------------------------


def test_ollama_settings_local(mock_ensure: MagicMock) -> None:
    """Without a key: local model, default host bootstrapped, no cloud mode."""
    settings = ollama_settings()

    mock_ensure.assert_called_once_with(DEFAULT_OLLAMA_HOST)
    assert settings == {"model": "qwen3:14b", "host": None, "think": False}


def test_ollama_settings_cloud(
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With OLLAMA_API_KEY: the same cloud model and host make_llm() uses."""
    monkeypatch.setenv("OLLAMA_API_KEY", "key")

    settings = ollama_settings()

    mock_ensure.assert_not_called()
    assert settings == {
        "model": "kimi-k2.7-code:cloud",
        "host": OLLAMA_CLOUD_HOST,
        "json_prompt_mode": True,
        "think": False,
    }


def test_ollama_settings_matches_make_llm(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A notebook doing OllamaLLM(**ollama_settings()) gets make_llm()'s LLM."""
    monkeypatch.setenv("OLLAMA_API_KEY", "key")

    make_llm(role="small")

    mock_ollama_llm.assert_called_once_with(**ollama_settings(role="small"))


# -- make_llm, model resolution -------------------------------------------


def test_explicit_model_wins(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit model beats both the env var and the role table."""
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")

    make_llm(model="explicit")

    assert mock_ollama_llm.call_args.kwargs["model"] == "explicit"


def test_env_model_beats_role_table(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """<PROVIDER>_MODEL applies to every role."""
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")

    make_llm(role="small")

    assert mock_ollama_llm.call_args.kwargs["model"] == "from-env"


def test_unknown_role_raises(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """An unknown role names the roles that do exist."""
    with pytest.raises(UnsupportedProviderError, match="Known roles"):
        make_llm(role="nonsense")


# -- make_llm, closed providers -------------------------------------------


def test_make_llm_openai(
    mock_openai_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choosing OpenAI skips the Ollama bootstrap entirely."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    make_llm(provider="openai")

    mock_ensure.assert_not_called()
    mock_openai_llm.assert_called_once_with(
        model="gpt-5",
        reasoning_effort="low",
    )


def test_make_llm_openai_strips_ollama_only_kwargs(
    mock_openai_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """think/json_prompt_mode would be a TypeError inside AsyncOpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    make_llm(provider="openai", think=False, json_prompt_mode=True)

    kwargs = mock_openai_llm.call_args.kwargs
    assert "think" not in kwargs
    assert "json_prompt_mode" not in kwargs


def test_make_llm_openai_caller_reasoning_effort_wins(
    mock_openai_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied reasoning_effort overrides the "low" default."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    make_llm(provider="openai", reasoning_effort="high")

    assert mock_openai_llm.call_args.kwargs["reasoning_effort"] == "high"


def test_make_llm_openai_prompts_when_key_missing(
    mock_openai_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opted-in reader who forgot to export gets a recovery path."""
    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.getpass.getpass",
        return_value="sk-typed",
    ) as mock_getpass:
        make_llm(provider="openai")

    mock_getpass.assert_called_once_with("OPENAI_API_KEY: ")


def test_make_llm_openai_skips_prompt_when_api_key_kwarg_present(
    mock_openai_llm: MagicMock,
) -> None:
    """A caller-supplied api_key kwarg satisfies the key requirement.

    OpenAILLM forwards api_key straight to AsyncOpenAI, so a caller who
    already passed one should never see the OPENAI_API_KEY prompt, even
    with no env var set.
    """
    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.getpass.getpass",
    ) as mock_getpass:
        make_llm(provider="openai", api_key="sk-direct")

    mock_getpass.assert_not_called()
    assert mock_openai_llm.call_args.kwargs["api_key"] == "sk-direct"


def test_make_llm_does_not_prompt_when_key_present(
    mock_openai_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No redundant prompt when the key is already exported."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.getpass.getpass",
    ) as mock_getpass:
        make_llm(provider="openai")

    mock_getpass.assert_not_called()


def test_ollama_path_never_prompts(
    mock_ollama_llm: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    """The book's default reader must never see an API key prompt."""
    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.getpass.getpass",
    ) as mock_getpass:
        make_llm()

    mock_getpass.assert_not_called()


def test_make_llm_anthropic(
    mock_anthropic_llm: MagicMock,
    mock_ensure: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choosing Anthropic skips the Ollama bootstrap and builds the client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    make_llm(provider="anthropic")

    mock_ensure.assert_not_called()
    mock_anthropic_llm.assert_called_once_with(model="claude-sonnet-5")


def test_make_llm_anthropic_drops_openai_only_kwargs(
    mock_anthropic_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reasoning_effort is OpenAI-only; AsyncAnthropic would reject it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    make_llm(provider="anthropic", reasoning_effort="low", think=False)

    kwargs = mock_anthropic_llm.call_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "think" not in kwargs


def test_make_llm_anthropic_prompts_when_key_missing(
    mock_anthropic_llm: MagicMock,
) -> None:
    """An opted-in Anthropic reader who forgot to export gets the prompt."""
    with patch(
        "llm_agents_from_scratch.notebook_utils.llm.getpass.getpass",
        return_value="sk-ant-typed",
    ) as mock_getpass:
        make_llm(provider="anthropic")

    mock_getpass.assert_called_once_with("ANTHROPIC_API_KEY: ")


def test_make_llm_rejects_supported_but_unwired_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider registered in the tables but with no branch fails loudly.

    The real guard is static: `Provider` is a Literal and the dispatch ends
    in `assert_never`, so a new member without a branch is a type error.
    This pins the runtime half -- reaching the tail raises rather than
    falling through to whichever branch came last.
    """
    monkeypatch.setattr(
        llm_mod,
        "SUPPORTED_PROVIDERS",
        (*llm_mod.SUPPORTED_PROVIDERS, "gemini"),
    )
    monkeypatch.setitem(llm_mod._MODELS, "gemini", {"default": "gemini-x"})

    with pytest.raises(AssertionError, match="unreachable"):
        make_llm(provider="gemini", api_key="sk-gem")  # type: ignore[arg-type]


# -- ensure_ollama ---------------------------------------------------------


def test_ensure_ollama_noop_when_already_up() -> None:
    """An already-running service is left alone."""
    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.subprocess.Popen",
        ) as mock_popen,
    ):
        ensure_ollama()

    mock_popen.assert_not_called()


def test_ensure_ollama_raises_for_unreachable_custom_host() -> None:
    """A non-default host that isn't already up can't be started locally.

    Spawning `ollama serve` binds its usual default, not the requested
    host, so the polling loop would hang until timeout instead of
    reporting the real problem.
    """
    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
            side_effect=ConnectionError,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.subprocess.Popen",
        ) as mock_popen,
        pytest.raises(RuntimeError, match="No Ollama service responding"),
    ):
        ensure_ollama(host="http://elsewhere:11434")

    mock_popen.assert_not_called()


def test_ensure_ollama_raises_without_binary() -> None:
    """A missing binary explains how to install it."""
    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
            side_effect=ConnectionError,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.shutil.which",
            return_value=None,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.os.path.exists",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="Could not find the ollama binary"),
    ):
        ensure_ollama()


def test_ensure_ollama_starts_server_and_waits_until_up() -> None:
    """A found binary is spawned, then polled until the service answers.

    The first probe fails (nothing running), the server is started, the
    next probe still fails (not up yet, so the loop sleeps once), and the
    third succeeds.
    """
    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
            side_effect=[ConnectionError, ConnectionError, MagicMock()],
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.shutil.which",
            return_value="/opt/bin/ollama",
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.subprocess.Popen",
        ) as mock_popen,
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.time.sleep",
        ) as mock_sleep,
    ):
        ensure_ollama()

    mock_popen.assert_called_once()
    assert mock_popen.call_args.args[0] == ["/opt/bin/ollama", "serve"]
    mock_sleep.assert_called_once_with(0.5)


def test_ensure_ollama_falls_back_to_known_binary_paths() -> None:
    """When the binary isn't on PATH, the well-known locations are probed."""
    lightning = "/teamspace/studios/this_studio/.local/bin/ollama"
    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
            side_effect=[ConnectionError, MagicMock()],
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.shutil.which",
            return_value=None,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.os.path.exists",
            side_effect=lambda p: p == lightning,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.subprocess.Popen",
        ) as mock_popen,
    ):
        ensure_ollama()

    assert mock_popen.call_args.args[0] == [lightning, "serve"]


def test_ensure_ollama_raises_when_server_never_comes_up() -> None:
    """If the service never answers before the deadline, it fails loudly."""
    from itertools import chain, repeat  # noqa: PLC0415

    with (
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.urllib.request.urlopen",
            side_effect=ConnectionError,
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.shutil.which",
            return_value="/opt/bin/ollama",
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.subprocess.Popen",
        ),
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.time.sleep",
        ),
        # deadline is computed at t=0; every later check sees t=100
        patch(
            "llm_agents_from_scratch.notebook_utils.llm.time.time",
            side_effect=chain([0], repeat(100)),
        ),
        pytest.raises(RuntimeError, match="did not start within 15s"),
    ):
        ensure_ollama()
