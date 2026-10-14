# Running the Notebooks

I recommend running the notebooks on **Ollama Cloud**, which serves larger
open-source models on Ollama's own hardware. These models are much more
capable than what an everyday laptop can run, and Ollama keeps rotating in
the newest, best-performing ones.

## Option A — Ollama Cloud (recommended)

1. Create an API key at <https://ollama.com/settings/keys>.
2. Export it: `export OLLAMA_API_KEY=<your key>`.

Then launch Jupyter from the project root:

```sh
uv run --with jupyter jupyter lab
```

The notebooks detect the key and use the default cloud model. I pick that
model by benchmarking the available Ollama Cloud models on how well each one
runs the notebooks end to end, and an automated run checks every notebook
against it weekly (see the
[notebook status page](https://masfromscratch.com/notebook-status/)). To try
another cloud model from <https://ollama.com/search?c=cloud>, set the
`OLLAMA_MODEL` environment variable to its name.

## Option B — Local Ollama

Install [Ollama](https://ollama.com/download) and pull the model:

```sh
ollama pull qwen3:14b
```

With no `OLLAMA_API_KEY` set, the notebooks start the Ollama service for you,
so there's no need to run `ollama serve`. A 14B model needs about 16 GB of RAM
and a dedicated GPU. You can set `OLLAMA_MODEL` to a smaller model (such as
`qwen3:4b`), but smaller models may not follow tool-use instructions
reliably, especially in later chapters.

## Option C — Lightning AI (local models on a free GPU)

[Lightning AI](https://lightning.ai) offers about 22 free GPU compute hours a
month, enough for several sessions on an L4.

1. [Create a free account](https://lightning.ai) if you don't have one.
2. Open the [book template](https://lightning.ai/nerdai/templates/build-a-multi-agent-system-from-scratch).
3. Click **Clone** and make sure it's the latest version.
4. Launch the Studio with an **L4 GPU**.
5. Open any notebook. Ollama and the model are already set up.

## OpenAI or Anthropic

The notebooks also run on OpenAI and Anthropic models. Install the extra
(`uv sync --extra openai` or `uv sync --extra anthropic`), export
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and set `LLM_PROVIDER=openai` (or
`anthropic`). The "Setting up the LLM" section at the top of each notebook
has the details.
