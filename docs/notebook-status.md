# Notebook Status

Which notebooks have been run end to end against which LLM
provider, and when. Generated from `notebook_status.yaml` by
`_scripts/render_notebook_status.py`; runs are recorded by
`_scripts/run_notebook.py`. Edit the ledger and re-render, not
this page.

| Symbol | Meaning |
|---|---|
| ✅ date `model` | Ran to completion on that date with that model |
| ❌ date | Failed; see the footnote |
| ⚪ not wired | The notebook constructs its LLM directly, so `LLM_PROVIDER` would have no effect; only run on local Ollama |
| ∅ no LLM | Nothing in the notebook builds an LLM; only run on local Ollama |
| – | Never run on that provider |

Human-in-the-loop prompts are answered by a scripted reply during
these runs, so a pass means the notebook runs to completion, not
that its prompts read well.

## Chapter notebooks

| Notebook | Ollama (local) | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|---|
| [`ch02.ipynb`](notebooks/ch02.ipynb) | ∅ no LLM | – | – | – |
| [`ch03.ipynb`](notebooks/ch03.ipynb) | ❌ 2026-09-29[^1] | – | – | – |
| [`ch04.ipynb`](notebooks/ch04.ipynb) | ❌ 2026-09-29[^2] | – | – | – |
| [`ch05.ipynb`](notebooks/ch05.ipynb) | ✅ 2026-09-29 `qwen3:14b` | – | – | – |
| [`ch06.ipynb`](notebooks/ch06.ipynb) | ✅ 2026-09-29 `qwen3:14b` | – | – | – |
| [`ch07.ipynb`](notebooks/ch07.ipynb) | ✅ 2026-09-29 `qwen3:14b` | – | – | – |
| [`ch08.ipynb`](notebooks/ch08.ipynb) | – | – | – | – |
| [`ch09.ipynb`](notebooks/ch09.ipynb) | – | – | – | – |
| [`ch10.ipynb`](notebooks/ch10.ipynb) | – | – | – | – |

[^1]: `examples/ch03.ipynb` / Ollama (local): cell 3: ModuleNotFoundError: nest_asyncio (not a declared dependency; pre-existing, unrelated to make_llm) (2026-09-29)
[^2]: `examples/ch04.ipynb` / Ollama (local): handler.result() read before the background run finished (InvalidStateError); a race only under back-to-back headless execution, reproduced identically on main (2026-09-29)

## Additional examples

| Notebook | Ollama (local) | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|---|
| [`async_tools.ipynb`](more-examples/ch02/async_tools.ipynb) | – | – | – | – |
| [`multi_tool_registry.ipynb`](more-examples/ch02/multi_tool_registry.ipynb) | – | – | – | – |
| [`pydantic_tool_validation.ipynb`](more-examples/ch02/pydantic_tool_validation.ipynb) | – | – | – | – |
| [`multi_turn_chat.ipynb`](more-examples/ch03/multi_turn_chat.ipynb) | – | – | – | – |
| [`parallel_tool_calls.ipynb`](more-examples/ch03/parallel_tool_calls.ipynb) | – | – | – | – |
| [`pdf_extraction.ipynb`](more-examples/ch03/pdf_extraction.ipynb) | – | – | – | – |
| [`jev_next_step_judge.ipynb`](more-examples/ch04/jev_next_step_judge.ipynb) | – | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | – | ✅ 2026-09-29 `claude-sonnet-5` |
| [`pokemon_comparison.ipynb`](more-examples/ch04/pokemon_comparison.ipynb) | – | – | – | – |
| [`pokemon_error_handling.ipynb`](more-examples/ch04/pokemon_error_handling.ipynb) | – | – | – | – |
| [`github_mcp.ipynb`](more-examples/ch05/github_mcp.ipynb) | – | – | – | – |
| [`goodnews_mcp.ipynb`](more-examples/ch05/goodnews_mcp.ipynb) | – | – | – | – |
| [`additional_resource_python.ipynb`](more-examples/ch06/additional_resource_python.ipynb) | – | – | – | – |
| [`skill_validation_errors.ipynb`](more-examples/ch06/skill_validation_errors.ipynb) | – | – | – | – |
| [`skills_marketplace.ipynb`](more-examples/ch06/skills_marketplace.ipynb) | – | – | – | – |
| [`user_explicit_hailstone.ipynb`](more-examples/ch06/user_explicit_hailstone.ipynb) | – | – | – | – |
| [`with_and_without_skills.ipynb`](more-examples/ch06/with_and_without_skills.ipynb) | – | – | – | – |
| [`recency_memory.ipynb`](more-examples/ch07/recency_memory.ipynb) | – | – | – | – |
| `reflective_memory.ipynb` | – | – | – | – |
| [`similarity_memory.ipynb`](more-examples/ch07/similarity_memory.ipynb) | – | – | – | – |
| [`approval_gate_in_skill.ipynb`](more-examples/ch08/approval_gate_in_skill.ipynb) | – | – | – | – |
| [`skill_with_human_input.ipynb`](more-examples/ch08/skill_with_human_input.ipynb) | – | – | – | – |
| [`supervised_trajectories.ipynb`](more-examples/ch08/supervised_trajectories.ipynb) | – | – | – | – |
| [`evaluator_pattern.ipynb`](more-examples/ch09/evaluator_pattern.ipynb) | – | – | – | – |
| [`subagent_failure.ipynb`](more-examples/ch09/subagent_failure.ipynb) | – | – | – | – |
| [`subagent_with_mcp_tool.ipynb`](more-examples/ch09/subagent_with_mcp_tool.ipynb) | – | – | – | – |
| [`cancelling_a_task_execution.ipynb`](more-examples/ch10/cancelling_a_task_execution.ipynb) | – | – | – | – |
| [`streaming_llmagent_executor.ipynb`](more-examples/ch10/streaming_llmagent_executor.ipynb) | – | – | – | – |

## Capstones

| Notebook | Ollama (local) | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|---|
| [`capstone_1.ipynb`](capstones/one/capstone_1.ipynb) | – | – | – | – |
| `capstone_2.ipynb` | – | – | – | – |
| `capstone_3.ipynb` | – | – | – | – |
| `capstone_4.ipynb` | – | – | – | – |
| `capstone_5.ipynb` | – | – | – | – |
