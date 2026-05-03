# Running the Notebooks

These additional example notebooks use **Qwen3:14B** as the default model
via Ollama. Minimum hardware: ~16 GB RAM with a dedicated GPU. You can swap
in a smaller model (e.g. `qwen3:4b`), but smaller models may not follow
tool-use instructions reliably.

## Option A — Run Locally

Install [Ollama](https://ollama.com/) and pull the model:

```sh
ollama pull qwen3:14b
```

Then launch Jupyter from the project root:

```sh
uv run --with jupyter jupyter lab
```

## Option B — Lightning AI (Free)

No GPU? [Lightning AI](https://lightning.ai) offers ~22 free GPU compute
hours/month — enough for several sessions on an L4.

1. [Create a free account](https://lightning.ai) if you don't have one
2. Open the [book template](https://lightning.ai/nerdai/templates/build-a-multi-agent-system-from-scratch)
3. Click **Clone** and ensure it's the latest version
4. Launch the Studio with an **L4 GPU**
5. Open any chapter notebook — Ollama and the model are pre-configured
