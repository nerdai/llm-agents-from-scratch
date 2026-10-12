# Notebook Status

Which notebooks have been run end to end against which LLM
provider. Generated from `notebook_status.yaml` by
`_scripts/render_notebook_status.py`; runs are recorded by
`_scripts/run_notebook.py`. Edit the ledger and re-render, not
this page.

**Last tested:** 2026-10-12

| Symbol | Meaning |
|---|---|
| ✅ `model` | Ran to completion with that model |
| ✅ `model` + footnote | Passed with a caveat; see the footnote |
| ❌ | Failed; see the footnote, which links the run's log when the run recorded one |
| ⚪ not wired | The notebook constructs its LLM directly, so the provider setting would have no effect; not run |
| ∅ no LLM | Nothing in the notebook builds an LLM; run under Ollama Cloud only |
| – | Never run on that provider |

Human-in-the-loop prompts are answered by a scripted reply during
these runs, so a pass means the notebook runs to completion, not
that its prompts read well.

A cell tagged `await:<name>` (ch04's result cell, for example) is
run after the runner waits on that future, standing in for the
pause a reader takes between starting a task and reading its
result. Those notebooks are meant to be run cell by cell, not with
Run All.

## Chapter notebooks

<div class="notebook-status" markdown>

| Notebook | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|
| [`ch02.ipynb`](notebooks/ch02.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`ch03.ipynb`](notebooks/ch03.ipynb) | ✅ `deepseek-v4.1-flash` | ⚪ not wired | ⚪ not wired |
| [`ch04.ipynb`](notebooks/ch04.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch05.ipynb`](notebooks/ch05.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch06.ipynb`](notebooks/ch06.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch07.ipynb`](notebooks/ch07.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch08.ipynb`](notebooks/ch08.ipynb) | ✅ `deepseek-v4.1-flash`[^1] | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch09.ipynb`](notebooks/ch09.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`ch10.ipynb`](notebooks/ch10.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |

</div>

[^1]: `chapter-notebooks/ch08/ch08.ipynb` / Ollama Cloud: passed on retry; first attempt: cell 12: TaskHandlerError: Failed to get next step: 1 validation error for NextStepDecision
  Invalid JSON: EOF while parsing a string at line 3 column 148 [type=json_invalid, input_value='{\n  "kind": "next_step".... Continue calling next', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/json_invalid

## Additional examples

<div class="notebook-status" markdown>

| Notebook | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|
| [`async_tools.ipynb`](more-examples/ch02/async_tools.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`multi_tool_registry.ipynb`](more-examples/ch02/multi_tool_registry.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`pydantic_tool_validation.ipynb`](more-examples/ch02/pydantic_tool_validation.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`multi_turn_chat.ipynb`](more-examples/ch03/multi_turn_chat.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`parallel_tool_calls.ipynb`](more-examples/ch03/parallel_tool_calls.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`pdf_extraction.ipynb`](more-examples/ch03/pdf_extraction.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`jev_next_step_judge.ipynb`](more-examples/ch04/jev_next_step_judge.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`pokemon_comparison.ipynb`](more-examples/ch04/pokemon_comparison.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`pokemon_error_handling.ipynb`](more-examples/ch04/pokemon_error_handling.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`github_mcp.ipynb`](more-examples/ch05/github_mcp.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`goodnews_mcp.ipynb`](more-examples/ch05/goodnews_mcp.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`additional_resource_python.ipynb`](more-examples/ch06/additional_resource_python.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`skill_validation_errors.ipynb`](more-examples/ch06/skill_validation_errors.ipynb) | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| [`skills_marketplace.ipynb`](more-examples/ch06/skills_marketplace.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`user_explicit_hailstone.ipynb`](more-examples/ch06/user_explicit_hailstone.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`with_and_without_skills.ipynb`](more-examples/ch06/with_and_without_skills.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`recency_memory.ipynb`](more-examples/ch07/recency_memory.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| `reflective_memory.ipynb` | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`similarity_memory.ipynb`](more-examples/ch07/similarity_memory.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`approval_gate_in_skill.ipynb`](more-examples/ch08/approval_gate_in_skill.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`skill_with_human_input.ipynb`](more-examples/ch08/skill_with_human_input.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ❌[^1] |
| [`supervised_trajectories.ipynb`](more-examples/ch08/supervised_trajectories.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`evaluator_pattern.ipynb`](more-examples/ch09/evaluator_pattern.ipynb) | ✅ `deepseek-v4.1-flash`[^2] | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`subagent_failure.ipynb`](more-examples/ch09/subagent_failure.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`subagent_with_mcp_tool.ipynb`](more-examples/ch09/subagent_with_mcp_tool.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| [`cancelling_a_task_execution.ipynb`](more-examples/ch10/cancelling_a_task_execution.ipynb) | ✅ `deepseek-v4.1-flash` | ⚪ not wired | ⚪ not wired |
| [`streaming_llmagent_executor.ipynb`](more-examples/ch10/streaming_llmagent_executor.ipynb) | ✅ `deepseek-v4.1-flash` | ⚪ not wired | ⚪ not wired |

</div>

[^1]: `more-examples/ch08/skill_with_human_input.ipynb` / Anthropic: cell 8: cell exceeded 600s (not retried: budget) ([run log](https://github.com/nerdai/llm-agents-from-scratch/actions/runs/38213193723))
[^2]: `more-examples/ch09/evaluator_pattern.ipynb` / Ollama Cloud: passed on retry; first attempt: cell 10: TaskHandlerError: Failed to get next step: 1 validation error for NextStepDecision
  Invalid JSON: invalid escape at line 3 column 567 [type=json_invalid, input_value='{\n  "kind": "next_step"...er 80 characters.\'"\n}', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/json_invalid

## Capstones

<div class="notebook-status" markdown>

| Notebook | Ollama Cloud | OpenAI | Anthropic |
|---|---|---|---|
| [`capstone_1.ipynb`](capstones/one/capstone_1.ipynb) | ✅ `deepseek-v4.1-flash` | ✅ `gpt-5` | ✅ `claude-sonnet-5` |
| `capstone_2.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| `capstone_3.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| `capstone_4.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM |
| `capstone_5.ipynb` | ∅ no LLM | ∅ no LLM | ∅ no LLM |

</div>
