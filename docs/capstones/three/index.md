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
| Task success | 1.0 | 0.67 | 0.67 |
| Wall time (s) | 231 | 98 | 0.42 |
| Steps | 67 | 4 | 0.05 |
| Rollout (chars) | 160K | 18K | 0.11 |
| Prompt (chars) | 23.3M | 1.7M | 0.075 |

On a strong model, neither system is reliably more accurate: in this
recording one coordinator run failed, and in earlier ones a single-agent
run did. What differs is how predictable each system's cost is:

- **The single agent's cost depends on its strategy.** Two replications ran
  the 20 estimates largely in lockstep, in 21 and 23 steps. The third worked
  through the samples largely one call at a time: 157 steps, re-sending a
  rollout of up to 270K characters on every step, and 60M prompt characters
  on its own.
- **The coordinator's context stays small.** It holds only the dispatches and
  their short results, so its successful replications took 5 steps each.
  The coordinator plus all of its subagents sent about 13 times fewer prompt
  characters than the single agent. The failed replication stopped after
  its first step on a JSON validation error, with 10 of its 20 estimates
  in hand, which also pulls the coordinator's averages down.
- **Success is checked strictly.** The final answer must name 20 distinct,
  real, in-range samples and report the standard error those samples give,
  so a run can't pass by copying IDs or inventing a value.

On a per-token API, the prompt-character ratio is roughly the ratio of the
two systems' input bills. On a flat plan like Ollama Cloud, it is how much of
the plan's usage limits each system consumes. Your numbers will vary with
the model and from run to run.

## How good are the standard errors?

The capstone ends by checking the reported standard errors against the
truth. Capstone I's algorithm is only arithmetic on random draws, so the
notebook simulates it 10,000 times without an LLM. It needs no points
either: the number that land inside the quarter circle follows a binomial
distribution, so one draw stands in for each sampling call. The simulation
gives:

- **The true standard error** for 20 estimates: 6.59 × 10⁻⁵.
- **How much a 20-estimate standard error varies by chance:** the middle 95%
  falls between 4.96 × 10⁻⁵ and 7.94 × 10⁻⁵, so even a perfect agent is only
  accurate to about ±20%.

Every standard error the agents reported in the recorded run falls inside
that range:

| Replication | Single agent | Percentile | Coordinator + subagents | Percentile |
|---|---|---|---|---|
| 1 | 6.41 × 10⁻⁵ | 41 | 7.06 × 10⁻⁵ | 75 |
| 2 | 5.85 × 10⁻⁵ | 18 | 6.05 × 10⁻⁵ | 24 |
| 3 | 6.90 × 10⁻⁵ | 66 | no answer | |

The simulation also shows what the standard error doesn't tell you:

- **The stopping rule sets it, not the sample size.** Every final estimate
  has to land in a window 0.001 wide, so the estimates spread almost evenly
  across it: one estimate's spread is 2.95 × 10⁻⁴, against 0.001/√12 =
  2.89 × 10⁻⁴ for a perfectly even spread. A plain Monte Carlo estimate with
  the median final sample size (4M points) spreads about three times more,
  8.21 × 10⁻⁴. The algorithm looks this precise only because it stops once it
  is near the value of π it already knows.
- **It can't show bias.** The window `[3.1415, 3.1425)` is centred on 3.1420,
  not on π, so the estimates average 3.141890: a bias of +2.97 × 10⁻⁴, about
  4.5 standard errors. A standard error measures how much the estimates vary,
  not how close they are to the truth.
