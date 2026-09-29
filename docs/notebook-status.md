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

| Notebook | Ollama Cloud | OpenAI | Anthropic | Ollama (local) |
|---|---|---|---|---|
| [`ch02.ipynb`](notebooks/ch02.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`ch03.ipynb`](notebooks/ch03.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ⚪ not wired | ⚪ not wired | ✅ 2026-09-29 `qwen3:14b` |
| [`ch04.ipynb`](notebooks/ch04.ipynb) | ❌ 2026-09-29[^1] | ❌ 2026-09-29[^2] | ❌ 2026-09-29[^3] | ❌ 2026-09-29[^4] |
| [`ch05.ipynb`](notebooks/ch05.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | ✅ 2026-09-29 `qwen3:14b` |
| [`ch06.ipynb`](notebooks/ch06.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | ✅ 2026-09-29 `qwen3:14b` |
| [`ch07.ipynb`](notebooks/ch07.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | ✅ 2026-09-29 `qwen3:14b` |
| [`ch08.ipynb`](notebooks/ch08.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | ❌ 2026-09-29[^5] |
| [`ch09.ipynb`](notebooks/ch09.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | – |
| [`ch10.ipynb`](notebooks/ch10.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | ✅ 2026-09-29 `qwen3:14b` |

[^1]: `examples/ch04.ipynb` / Ollama Cloud: cell 31: InvalidStateError: Exception is not set. (2026-09-29)
[^2]: `examples/ch04.ipynb` / OpenAI: cell 31: InvalidStateError: Exception is not set. (2026-09-29)
[^3]: `examples/ch04.ipynb` / Anthropic: cell 31: InvalidStateError: Exception is not set. (2026-09-29)
[^4]: `examples/ch04.ipynb` / Ollama (local): handler.result() read before the background run finished (InvalidStateError); a race only under back-to-back headless execution, reproduced identically on main (2026-09-29)
[^5]: `examples/ch08.ipynb` / Ollama (local): cell 9: cell exceeded 1800s; Example 1 was progressing correctly (7 tool steps in 30 min), qwen3:14b is just too slow here (2026-09-29)

## Additional examples

| Notebook | Ollama Cloud | OpenAI | Anthropic | Ollama (local) |
|---|---|---|---|---|
| [`async_tools.ipynb`](more-examples/ch02/async_tools.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| [`multi_tool_registry.ipynb`](more-examples/ch02/multi_tool_registry.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| [`pydantic_tool_validation.ipynb`](more-examples/ch02/pydantic_tool_validation.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| [`multi_turn_chat.ipynb`](more-examples/ch03/multi_turn_chat.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`parallel_tool_calls.ipynb`](more-examples/ch03/parallel_tool_calls.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`pdf_extraction.ipynb`](more-examples/ch03/pdf_extraction.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`jev_next_step_judge.ipynb`](more-examples/ch04/jev_next_step_judge.ipynb) | ✅ 2026-09-29 `kimi-k2.7-code:cloud` | ✅ 2026-09-29 `gpt-5` | ✅ 2026-09-29 `claude-sonnet-5` | – |
| [`pokemon_comparison.ipynb`](more-examples/ch04/pokemon_comparison.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`pokemon_error_handling.ipynb`](more-examples/ch04/pokemon_error_handling.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`github_mcp.ipynb`](more-examples/ch05/github_mcp.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`goodnews_mcp.ipynb`](more-examples/ch05/goodnews_mcp.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`additional_resource_python.ipynb`](more-examples/ch06/additional_resource_python.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`skill_validation_errors.ipynb`](more-examples/ch06/skill_validation_errors.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| [`skills_marketplace.ipynb`](more-examples/ch06/skills_marketplace.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`user_explicit_hailstone.ipynb`](more-examples/ch06/user_explicit_hailstone.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`with_and_without_skills.ipynb`](more-examples/ch06/with_and_without_skills.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`recency_memory.ipynb`](more-examples/ch07/recency_memory.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| `reflective_memory.ipynb` | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`similarity_memory.ipynb`](more-examples/ch07/similarity_memory.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`approval_gate_in_skill.ipynb`](more-examples/ch08/approval_gate_in_skill.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`skill_with_human_input.ipynb`](more-examples/ch08/skill_with_human_input.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`supervised_trajectories.ipynb`](more-examples/ch08/supervised_trajectories.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`evaluator_pattern.ipynb`](more-examples/ch09/evaluator_pattern.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`subagent_failure.ipynb`](more-examples/ch09/subagent_failure.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`subagent_with_mcp_tool.ipynb`](more-examples/ch09/subagent_with_mcp_tool.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| [`cancelling_a_task_execution.ipynb`](more-examples/ch10/cancelling_a_task_execution.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| [`streaming_llmagent_executor.ipynb`](more-examples/ch10/streaming_llmagent_executor.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |

## Capstones

| Notebook | Ollama Cloud | OpenAI | Anthropic | Ollama (local) |
|---|---|---|---|---|
| [`capstone_1.ipynb`](capstones/one/capstone_1.ipynb) | ⚪ not wired | ⚪ not wired | ⚪ not wired | – |
| `capstone_2.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| `capstone_3.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| `capstone_4.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
| `capstone_5.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM | – |
