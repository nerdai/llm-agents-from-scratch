---
name: benchmark-ollama-models
description: Benchmark Ollama Cloud models on the chapter notebooks' behaviour (real tool calls, skill activation, subagents finishing), not just pass/fail, to pick or re-validate the default Ollama Cloud model. Use before changing make_llm()'s Ollama Cloud default, when a cloud model is retired, or when asked to compare or test cloud models on the notebooks.
---

# Benchmark Ollama Cloud models

Pick the default Ollama Cloud model by what the chapter notebooks' agents
*do*, not whether the notebooks finish. The tool is
`_scripts/benchmark_models.py`; this skill is the procedure around it.

## Why behaviour, not pass/fail

A notebook can run to completion while the model gets the lesson wrong.
Both of these happened with `gemma4:31b`:

- **Faked tool calls:** ch04 "passed" while the agent made no tool calls
  and wrote a trace claiming it had.
- **Subagent out of steps:** ch09 "passed" because the coordinator
  handles the failure gracefully.

`benchmark_models.py` records **ran** (no errors) and **behaved** (ran and
met every check in its `EXPECTATIONS` table for that chapter). Rank by
behaved.

## When to run it

- Before changing the Ollama Cloud default in
  `src/llm_agents_from_scratch/notebook_utils/llm.py` (`_MODELS["ollama-cloud"]`).
- When the current default is retired, or Ollama Cloud adds models.
- Occasionally, to catch drift in the current default.

## Procedure

Load the keys first: `set -a && source .env && set +a`.

1. **List the models:**
   ```bash
   uv run python _scripts/benchmark_models.py models
   ```
2. **Stage 1: screen every model** on the quick chapters (ch03–ch07),
   plus ch04 two more times, since its failure mode is intermittent. Run
   models in parallel; these notebooks bind no ports:
   ```bash
   uv run python _scripts/benchmark_models.py models | xargs -P 3 -I{} \
     uv run python _scripts/benchmark_models.py run {} --out stage1.jsonl
   uv run python _scripts/benchmark_models.py models | xargs -P 3 -I{} \
     uv run python _scripts/benchmark_models.py run {} --chapters ch04 \
       --runs 2 --out stage1.jsonl
   uv run python _scripts/benchmark_models.py summary stage1.jsonl
   ```
   Keep the models with a perfect stage-1 score, usually a handful.
3. **Stage 2: the shortlist, 3 runs** on the harder chapters. ch08 is the
   approval gate; ch09 has concurrent subagents, chaining and human input,
   and is where models most often fail:
   ```bash
   printf '%s\n' <model> <model> ... | xargs -P 3 -I{} \
     uv run python _scripts/benchmark_models.py run {} \
       --chapters ch04,ch06,ch07,ch08,ch09 --runs 3 --out stage2.jsonl
   uv run python _scripts/benchmark_models.py summary stage2.jsonl
   ```
   This takes 20–30 minutes; ch09 alone is 2–5 minutes per run.
4. **Pick:** the highest behaved score, then the fastest (readers wait on
   ch09). Name a backup, the next model with a perfect score.
5. **Check the pick on what the benchmark skips**, with
   `OLLAMA_MODEL=<pick>` and an isolated HOME (see Pitfalls):
   - ch10 (A2A peer apps; the CrewAI peer uses `ollama_chat/<pick>`)
   - ch02–ch03
   - one `_scripts/run_notebook.py` pass over `more-examples/` and
     `capstones/`

## Switching the default

All in one PR:

- **Code:** `_MODELS["ollama-cloud"]` (every role) in `notebook_utils/llm.py`,
  and the two assertions in `tests/notebook_utils/test_llm.py`.
- **ch10 peers:** `extra/a2a-crewai-hailstone/main.py` (default
  `ollama_chat/<pick>`) and the three `extra/` READMEs.
- **Docs:** a CHANGELOG entry citing the benchmark, and a
  REQUIRED_BOOK_UPDATES entry (gitignored, never committed) covering
  every place the book names the model.
- **Re-record the chapter notebooks** with `run_notebook.py <the chapter
  notebooks> --provider ollama-cloud --keep-outputs` in an isolated HOME
  (`chapter-notebooks/ch*/ch*.ipynb` once each chapter has its own folder,
  #1034; `chapter-notebooks/ch*.ipynb` before). Then confirm, against `main`:
  - sources are unchanged,
  - every notebook reports the new model,
  - no API key appears in any output,
  - the agents' tool calls haven't disappeared.

## Pitfalls

- **Isolated HOME, always.** User-level skills (`~/.claude/skills`,
  `~/.agents/skills`) join every agent's skill catalog and change its
  behaviour. `benchmark_models.py run` isolates HOME by default. For
  `run_notebook.py`, set `HOME=$(mktemp -d)` and keep
  `UV_CACHE_DIR=~/.cache/uv` and
  `UV_PYTHON_INSTALL_DIR=~/.local/share/uv/python` pointing at the real
  ones.
- **One run proves little.** Gemma faked ch04's tool calls in 1 of 4
  runs. Use 3 runs for anything that decides the default.
- **ch04's agent runs in the background.** Its tool-call logs arrive after
  the starting cell finishes. The benchmark counts tool results in
  `handler.rollout` instead, so don't "simplify" the ch04 check to the
  logs.
- **`OLLAMA_MODEL` takes plain names** (`deepseek-v4.1-flash`). Only the
  CrewAI peer needs the `ollama_chat/` prefix, and its default is set in
  code.
- **Re-running ch10 re-locks `extra/*/uv.lock`.** Revert only those
  files (`git checkout -- 'extra/*/uv.lock'`), never all of `extra/`. A
  model switch edits the CrewAI peer and READMEs there, and a blanket
  revert silently undoes them.
- **The isolated HOME breaks two additional examples:** `goodnews_mcp`
  (its server lives in `~/mcp-servers/`) and `subagent_with_mcp_tool`
  (`uvx` tools and the browser extra live under HOME). Run those two with
  the real HOME; neither depends on the skill catalog.

## History

| Date | Default | Result |
|---|---|---|
| 2026-10-11 | `kimi-k2.7-code:cloud` → `gemma4:31b` → `deepseek-v4.1-flash` | Kimi missed skill activation (ch06/07). Gemma failed ch09 (0/2). DeepSeek was 15/15 and fastest; `glm-5.2` was 15/15 but ~2.5× slower, so it's the backup. |
