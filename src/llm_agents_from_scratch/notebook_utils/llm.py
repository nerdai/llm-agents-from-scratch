# ruff: noqa: PLC0415
"""LLM notebook utils.

Lets a reader run any notebook in this repo against a provider other than
Ollama without editing the notebook's teaching code.

The book teaches Ollama, so Ollama is always the default. Selecting another
provider is an explicit act -- see `make_llm`.
"""

import getpass
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from llm_agents_from_scratch.base.llm import LLM
from llm_agents_from_scratch.errors import UnsupportedProviderError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CLOUD_HOST = "https://ollama.com"

#: Ollama-specific constructor kwargs, dropped for other providers.
_OLLAMA_ONLY_KWARGS = ("think", "json_prompt_mode")

#: Per-provider, per-role default models. `ollama-cloud` is not a provider a
#: caller selects; it is what `ollama` resolves to when OLLAMA_API_KEY is set.
_MODELS: dict[str, dict[str, str]] = {
    "ollama": {
        "default": "qwen3:14b",
        "small": "qwen3:8b",
        "judge": "qwen3:14b",
    },
    "ollama-cloud": {
        "default": "kimi-k2.7-code:cloud",
        "small": "kimi-k2.7-code:cloud",
        "judge": "kimi-k2.7-code:cloud",
    },
    "openai": {
        "default": "gpt-5",
        "small": "gpt-5-mini",
        "judge": "gpt-5",
    },
    "anthropic": {
        "default": "claude-sonnet-5",
        "small": "claude-haiku-4-5-20251001",
        "judge": "claude-opus-5",
    },
}

#: Providers a caller may ask for by name.
SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")


def ensure_ollama(
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: int = 15,
) -> None:
    """Start Ollama if not already running and wait until responsive.

    Args:
        host (str): Host of the Ollama service. Defaults to localhost.
        timeout (int): Seconds to wait for the service to come up.
            Defaults to 15.

    Raises:
        RuntimeError: If the ollama binary cannot be found, or the service
            does not become responsive within `timeout` seconds.
    """

    def _up() -> bool:
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=1)
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    if _up():
        print(f"✓ Ollama already running at {host}")
        return

    if host != DEFAULT_OLLAMA_HOST:
        raise RuntimeError(
            f"No Ollama service responding at {host}. Only the default "
            f"local host ({DEFAULT_OLLAMA_HOST}) can be started "
            "automatically -- spawning `ollama serve` here would bind "
            "its usual default, not the requested host. Start the "
            "service at that host yourself, or omit `host` to use the "
            "default.",
        )

    # Lightning persistent path first, then standard locations
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        for candidate in [
            "/teamspace/studios/this_studio/.local/bin/ollama",
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
        ]:
            if os.path.exists(candidate):
                ollama_path = candidate
                break
    if ollama_path is None:
        raise RuntimeError(
            "Could not find the ollama binary. Install with: "
            "curl -fsSL https://ollama.com/install.sh | sh",
        )

    print(f"Starting Ollama server ({ollama_path})...")
    subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _up():
            print(f"✓ Ollama up and running at {host}")
            return
        time.sleep(0.5)

    raise RuntimeError(f"Ollama did not start within {timeout}s")


def resolve_provider(provider: str | None = None) -> str:
    """Resolve which LLM provider to use.

    Provider selection is explicit, never inferred from which API keys happen
    to be set. A reader with OPENAI_API_KEY exported for unrelated reasons
    still gets the Ollama path the book teaches.

    Args:
        provider (str | None): Explicit provider name. Defaults to None, in
            which case the LLM_PROVIDER env var is consulted, then "ollama".

    Returns:
        str: One of SUPPORTED_PROVIDERS.

    Raises:
        UnsupportedProviderError: If the resolved name is not recognized.
    """
    resolved = provider or os.environ.get("LLM_PROVIDER") or "ollama"
    resolved = resolved.strip().lower()

    if resolved not in SUPPORTED_PROVIDERS:
        raise UnsupportedProviderError(
            f"Unknown LLM provider {resolved!r}. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}.",
        )
    return resolved


def _ensure_api_key(provider: str) -> None:
    """Make sure an API key is present, prompting for one if it is not.

    Only ever reached once a closed provider has been explicitly chosen, so
    the default Ollama reader is never prompted. Prompting via getpass keeps
    the key out of the saved notebook, and is the only workable path on
    hosted kernels where env vars cannot be set before launch.

    Args:
        provider (str): The resolved provider name.
    """
    env_var = f"{provider.upper()}_API_KEY"
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass.getpass(f"{env_var}: ")


def _resolve_model(provider: str, role: str, model: str | None) -> str:
    """Resolve the model name for a provider and role.

    Precedence: explicit `model` arg, then the <PROVIDER>_MODEL env var,
    then the per-role default.

    Args:
        provider (str): Provider key into the model table.
        role (str): The role being filled, e.g. "default", "small", "judge".
        model (str | None): Explicit model name, which wins outright.

    Returns:
        str: The model name to construct the LLM with.

    Raises:
        UnsupportedProviderError: If `role` has no default for `provider`.
    """
    if model is not None:
        return model

    # <PROVIDER>_MODEL applies to every role -- a reader who names one model
    # is asking for that model, not for a per-role table.
    env_key = provider.split("-", maxsplit=1)[0].upper()
    if env_model := os.environ.get(f"{env_key}_MODEL"):
        return env_model

    roles = _MODELS[provider]
    if role not in roles:
        raise UnsupportedProviderError(
            f"No default model for role {role!r} on provider {provider!r}. "
            f"Known roles: {', '.join(sorted(roles))}. "
            "Pass model=... to use something else.",
        )
    return roles[role]


def make_llm(
    role: str = "default",
    *,
    provider: str | None = None,
    model: str | None = None,
    host: str | None = None,
    **kwargs: Any,
) -> LLM:
    """Build an LLM for a notebook, defaulting to the Ollama path.

    With nothing configured this starts Ollama if needed and returns an
    ~OllamaLLM -- exactly what the book teaches. Readers who want a closed
    model select one explicitly, either with `provider` or by setting the
    LLM_PROVIDER env var.

    Args:
        role (str): Which model slot to fill: "default", "small", or "judge".
            Defaults to "default".
        provider (str | None): "ollama", "openai", or "anthropic". Defaults
            to None, resolving via LLM_PROVIDER then "ollama".
        model (str | None): Overrides the per-role default model. Defaults
            to None.
        host (str | None): Ollama host override. Ignored by other providers.
            Defaults to None.
        **kwargs (Any): Passed to the LLM constructor. Ollama-only kwargs
            are dropped for other providers.

    Returns:
        LLM: A ready-to-use LLM instance.

    Raises:
        UnsupportedProviderError: If the provider is unknown.
    """
    resolved = resolve_provider(provider)

    if resolved == "ollama":
        return _make_ollama_llm(role=role, model=model, host=host, **kwargs)

    # closed providers: selection already happened, so a key is now required
    for key in _OLLAMA_ONLY_KWARGS:
        kwargs.pop(key, None)
    # a caller-supplied api_key kwarg (both SDK clients accept one) already
    # satisfies the requirement, so skip the env var/prompt path
    if not kwargs.get("api_key"):
        _ensure_api_key(resolved)
    resolved_model = _resolve_model(resolved, role, model)

    if resolved == "openai":
        from llm_agents_from_scratch.llms.openai import OpenAILLM

        kwargs.setdefault("reasoning_effort", "low")
        print(f"✓ Using OpenAI ({resolved_model})")
        return OpenAILLM(model=resolved_model, **kwargs)

    from llm_agents_from_scratch.llms.anthropic import AnthropicLLM

    # OpenAI-only; AsyncAnthropic would reject it as an unexpected kwarg
    kwargs.pop("reasoning_effort", None)
    print(f"✓ Using Anthropic ({resolved_model})")
    return AnthropicLLM(model=resolved_model, **kwargs)


def _make_ollama_llm(
    role: str,
    model: str | None,
    host: str | None,
    **kwargs: Any,
) -> LLM:
    """Build an OllamaLLM, using Ollama Cloud when OLLAMA_API_KEY is set.

    Args:
        role (str): Which model slot to fill.
        model (str | None): Overrides the per-role default model.
        host (str | None): Host override.
        **kwargs (Any): Passed to the ~OllamaLLM constructor.

    Returns:
        LLM: An ~OllamaLLM instance.
    """
    from llm_agents_from_scratch.llms.ollama import OllamaLLM

    use_cloud = "OLLAMA_API_KEY" in os.environ
    provider_key = "ollama-cloud" if use_cloud else "ollama"
    resolved_model = _resolve_model(provider_key, role, model)

    if use_cloud:
        host = host or OLLAMA_CLOUD_HOST
        # cloud models ignore Ollama's `format` parameter, so structured
        # output has to be coerced at the prompt level instead
        kwargs.setdefault("json_prompt_mode", True)
        print(f"✓ Using Ollama Cloud ({resolved_model})")
    else:
        ensure_ollama(host or DEFAULT_OLLAMA_HOST)
        print(f"✓ Using Ollama ({resolved_model})")

    kwargs.setdefault("think", False)
    return OllamaLLM(model=resolved_model, host=host, **kwargs)
