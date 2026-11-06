# Capstone II: Revisiting Monte Carlo Estimation of Pi

<!-- markdownlint-disable-file MD001 MD033 MD046 -->

!!! info "About this capstone"
    This capstone is a companion resource for
    *Build a Multi-Agent System — With MCP and A2A* (Manning), for readers who
    have reached Chapter 9 and its subagents. It builds directly on
    [Capstone I](../one/index.md).

!!! abstract "What this capstone covers"
    - Estimating the standard error of Capstone I's Monte Carlo algorithm
    - Context sprawl in a single agent, and isolating context with subagents
    - Measuring what a multi-agent system costs and buys: success, wall time,
      rollout size and prompt tokens

In Capstone I, an `LLMAgent` estimated π by growing a random sample until the
Monte Carlo estimate landed in `[3.1415, 3.1425)`. That's a point estimate.
This capstone asks the statistician's next question: how much do those
estimates vary? In other words, what is the **standard error** of Capstone I's
algorithm?

To answer it, we run the whole algorithm `K` times, each on its own new
sample, and compute `SE = s / √K` over the `K` final estimates. A single agent
doing this has to track `K` samples, `K` doubling schedules and `K`
convergence states in one growing rollout: the context sprawl that Chapter 9
opens with. So we build the task twice and compare.

| | System 1: single agent | System 2: coordinator + subagents |
|---|---|---|
| Who runs the algorithm | the agent itself, `K` times in one rollout | one `pi-estimator` subagent per estimate, each in a fresh context |
| Tools | Capstone I's three tools + `compute_standard_error` | subagents: Capstone I's three tools; coordinator: `compute_standard_error` |
| What the agent's context holds | every tool call for every sample | `K` dispatches and their short results |

All code listings are in the capstone's [notebook](capstone_2.ipynb). Launch
Jupyter from the root of the book's code:

```sh
uv run --with jupyter jupyter lab
```

Then open `capstones/three/capstone_2.ipynb`.

## What changes from Capstone I

- **One tool change.** `monte_carlo_estimate` also reports `in_target_range`.
  In Capstone I the agent compared the estimate against the target range
  itself, and LLMs occasionally get that comparison wrong. That mistake has
  nothing to do with context, so here the tool makes the check.
- **A new tool.** `compute_standard_error(sample_ids)` computes the mean and
  standard error from the sample registry, so the arithmetic isn't left to
  the LLM either.
- **Task success is checked against ground truth.** The final answer must
  name `K` distinct, real samples, each in range. The JSON is extracted from
  the final answer rather than required verbatim.
- **Context is measured.** Each run records wall time, steps, rollout size
  and the prompt characters sent to the LLM, via a small `MeteredLLM` wrapper.
  For the coordinator, that count includes every subagent's calls.

## Running it on a cloud model

Dispatches in the same response run concurrently, and cloud providers cap how
many requests they serve at once (Ollama Cloud answers HTTP 429 beyond its
cap). The coordinator therefore dispatches in batches of at most five, and
replications run one after another.

The notebook's metadata limits the weekly notebook-status sweep to Ollama
Cloud. A full run takes about half an hour, and on a per-token API the single
agent's sprawling runs alone send tens of millions of prompt tokens.

## Results

The recorded run uses `deepseek-v4.1-flash` on Ollama Cloud, with `K = 20`
estimates and three replications per system. These are averages over the
replications:

| | Single agent | Coordinator + subagents | Ratio |
|---|---|---|---|
| Task success | 0.67 | 1.0 | 1.5 |
| Wall time (s) | 361 | 208 | 0.58 |
| Steps | 90 | 6 | 0.06 |
| Rollout (chars) | 165K | 24K | 0.15 |
| Prompt (chars) | 34.4M | 2.9M | 0.084 |

The coordinator succeeded every time. What stands out more is how
predictable its cost is:

- **The single agent's cost depends on its strategy.** Its two successful
  replications took 87 and 180 steps, working through the samples largely
  one call at a time and re-sending a rollout of up to 330K characters on
  every step. The longer one sent 84M prompt characters on its own. The
  third replication stopped after 4 steps without a final answer, when the
  next-step decision ended the task early.
- **The coordinator's context stays small.** It holds only the dispatches and
  their short results, so every replication took 5 or 6 steps, and the
  coordinator plus all of its subagents sent about 12 times fewer prompt
  characters than the single agent.
- **Success is checked strictly.** The final answer must name 20 distinct,
  real, in-range samples and report the standard error those samples give,
  so a run can't pass by copying IDs or inventing a value.

On a per-token API, the prompt-character ratio is roughly the ratio of the
two systems' input bills. On a flat plan like Ollama Cloud, it is how much of
the plan's usage limits each system consumes. Your numbers will vary with
the model and from run to run.
