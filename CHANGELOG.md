<!-- markdownlint-disable-file MD024 -->

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

### Changed

- refactor(ch10): `a2a_response_to_tool_call_result()` (`a2a/client/utils.py`) now wraps its own body in `try`/`except`, returning an error `ToolCallResult` for any exception while parsing a peer's response (a malformed/unexpected shape from a non-conformant peer), instead of relying on its one caller (`UseA2AAgentTool.__call__`) to catch it — mirrors the pattern its own nested `a2a_task_to_tool_call_result()` already uses for an unrecognized `TaskState`; `__call__` now just returns the helper's result directly (#875)
- refactor(ch10): `UseA2AAgentTool.__call__` — replaced the module-level `_validate_arguments()` helper (hand-rolled `isinstance` checks for `name`/`task`/`task_id`) with `validate_tool_call_arguments()` (`tools/utils.py`), mirroring #871's `UseSubAgentTool` change; safe for the same reason — `parameters_json_schema`'s `name` enum is always the exact, complete set of dispatchable peer names; an unknown peer name is now caught by that enum as a `ValidationError` listing valid names, more actionable than the removed `A2AAgentNotFoundError`'s plain "not found" message, and nothing caught it by type, so it's deleted (`A2AError`, its still-used base, stays) (#875)
- refactor(ch09): `UseSubAgentTool.__call__` — replaced two manual `isinstance` checks with `validate_tool_call_arguments()` (`tools/utils.py`, added in #851 after this tool was written); since `parameters_json_schema`'s `name` enum is always the exact, complete set of dispatchable subagent names (no `explicit_only`-style carve-out like `UseSkillTool`), an unknown subagent name is now caught by that enum as a `ValidationError` listing the valid names — more actionable for the LLM than the removed `SubAgentNotFoundError`/`SubAgentsError` classes' plain "not found" message ever was, and nothing caught them by type, so both are deleted (#871)
- fix(openai): `tool_to_openai_tool()` no longer declares `"strict": True` — OpenAI's strict mode requires `additionalProperties: false` on every object and every property listed in `required`, which neither of this library's schema generators produces (`function_signature_to_json_schema` deliberately omits defaulted parameters from `required`, which strict mode also forbids); any library-built tool on an OpenAI-backed agent turn was failing with `Invalid schema for function '...': 'additionalProperties' is required to be supplied and to be false`. Flips the flag to `False`, matching the schemas actually produced; a schema normalizer plus an opt-in `strict` flag is a fuller fix left for later (#861)
- chore: `mcp[cli]` lower bound `>=1.9.4` → `>=2.0.0` in `pyproject.toml` — the code has required mcp's 2.0.0+ API (`streamable_http_client`, renamed from `streamablehttp_client`) since #742, but the dependency pin was never bumped to match; today's resolution already lands on mcp 2.x, but nothing previously enforced it (#864)
- refactor(ch08): `_prompt_for_approval` moved out of `TaskHandler.request_approval()` (was a nested closure) to a module-level function in `agent/llm_agent.py`, mirroring `_prompt_human` in `tools/default/human_input.py`; no behavior change, no closure-captured state was needed (#855)
- refactor(ch02): `tools/utils.py` — new `validate_tool_call_arguments()`, matching the `utils.py`-per-subpackage convention already used elsewhere (`skills/`, `llms/ollama/`, `a2a/client/`); replaces the same jsonschema-validation try/except block duplicated across `SimpleFunctionTool`/`AsyncSimpleFunctionTool` (`tools/simple_function.py`) and `HumanInputTool`/`SharedConsoleHumanInputTool` (`tools/default/human_input.py`, ch08); returns `dict[str, str] | None` (an `{"error_type": ..., "message": ...}` dict on failure) rather than a full `ToolCallResult`, so each call site builds its own `ToolCallResult` from `json.dumps(validation_error_details)` (#851)
- fix(ch08): `HumanInputTool`/`SharedConsoleHumanInputTool` — `minLength: 1` on the `prompt` schema property, and the manual `if not prompt:` check in `__call__` replaced with explicit `jsonschema.validate()` enforcement (mirroring `SimpleFunctionTool`'s existing pattern); previously nothing enforced `parameters_json_schema` for these two hand-written tools at runtime, so `minLength` alone would have been purely descriptive (#849)
- refactor(ch10): `a2a/client/` cleanup following #835 — dropped the unused `from __future__ import annotations` (quoted the two self-referencing return annotations instead); renamed `CATALOG_A2A_SKILL_TEMPLATE` → `CATALOG_INDIVIDUAL_A2A_SKILL_TEMPLATE` to distinguish it from the plural `CATALOG_A2A_SKILLS_TEMPLATE`; `A2AAgentSpec.catalog()` now uses a guard clause, matching `LLMAgent`'s `_skills_catalog`/`_subagents_catalog`/`_a2a_agents_catalog` pattern instead of an inline ternary (#836)
- refactor(ch10): `A2AAgentSpec` — plain class instead of a pydantic `BaseModel`, same reasoning as `SubAgentSpec` (#833): `agent_card` is an SDK `AgentCard`, actually a protobuf message rather than a pydantic type, so `arbitrary_types_allowed=True` was the only thing pydantic bought here; same constructor signature, `from_agent_card()`/`from_url()`/`catalog()` unchanged, hand-written `__repr__` (#835)
- refactor(ch09): `SubAgentSpec` — plain class instead of a pydantic `BaseModel`, matching `Skill`/`Memory`/`LLMAgentBuilder`'s pattern for objects that wrap a live collaborator (`arbitrary_types_allowed=True` was the only thing pydantic bought here); same constructor signature, `catalog()` unchanged, hand-written `__repr__` (#833)
- refactor(ch09): `SubAgentSpec.agent: LLMAgent` → `SubAgentSpec.builder: LLMAgentBuilder` — the spec is now pure data holding a build recipe, not a live agent, mirroring `A2AAgentSpec`'s rule; `UseSubAgentTool` builds a fresh `LLMAgent` from the recipe on every dispatch, reusing whatever the builder was given directly (memory stores, MCP providers) across builds (#830)

## [0.0.23] - 2026-08-07

### Added

- feat(ch10): `a2a/server/streaming_executor.py` — `StreamingLLMAgentA2AExecutor` (streaming-capable sibling to `LLMAgentA2AExecutor`, self-contained, drives `LLMAgent.run_supervised()`'s caller-controlled `get_next_step()`/`run_step()` loop directly, publishing two `TASK_STATE_WORKING` updates per step instead of only a final terminal state) and `build_streaming_agent_card()` (mirrors `build_agent_card()`, differing only in `capabilities=AgentCapabilities(streaming=True)`); `a2a/server/__init__.py`/`a2a/__init__.py` re-export both (#817)
- feat(ch10): `a2a/server/` — `LLMAgentA2AExecutor` (bridges inbound A2A tasks to an `LLMAgent`; `cancel()` settles both `TaskHandler.background_task` and the `TaskHandler` future itself, since the latter never resolves on its own) and `build_agent_card()` (plain function returning an `a2a.types.AgentCard`, no composite wrapper class); `a2a/__init__.py` re-exports both alongside the existing client-side names (#787)
- feat(ch10): `A2AAgentSpec.timeout` (default 60.0) — configurable per-peer dispatch timeout for `UseA2AAgentTool`, passed through to `httpx.AsyncClient`; overridable via `from_agent_card()`/`from_url()` (#807)
- feat(ch10): `UseA2AAgentTool` — dispatch schema `name` + `task` + optional `task_id` to resume a peer task parked in `TASK_STATE_INPUT_REQUIRED`; creates an SDK client fresh per dispatch and always closes it; `TaskHandler._use_a2a_agent_tool`/`_a2a_agents_catalog` wired into `run_step` (#800)
- feat(ch10): `LLMAgent.a2a_agents_registry` (keyed by plain peer name, unlike `MCPToolProvider`'s prefixed tool names — A2A dispatch is one generic tool with `name` as an enum value); `LLMAgentBuilder.with_a2a_agent()`/`with_a2a_agents()` fluent methods and `build()` pass-through (#799)
- feat(ch10): `a2a/` package — `A2AAgentSpec`, the registry value type for a peer A2A agent; `from_agent_card()`/`from_url()` construction, `.catalog()` with nested `<a2a_skills>`; `errors/a2a.py` — `A2AAgentCardMissingInterfaceError` (#798)
- chore(ch10): add `a2a-sdk` core dependency; `errors/a2a.py` — `A2AError`/`A2AAgentNotFoundError`, mirroring `errors/subagents.py`'s `SubAgentsError`/`SubAgentNotFoundError` pattern (#797)

### Changed

- refactor(ch10): split `a2a/` into `a2a/client/` (`spec.py`, `tools.py`, `utils.py`, `constants.py`) ahead of #787's server-side additions; `a2a/__init__.py`'s public re-exports are unchanged (#811)

## [0.0.22] - 2026-08-04

### Added

- feat(ch09): `SubAgentSpec.skills_scopes` and `SubAgentSpec.explicit_only_skills` — per-sub-agent skill scope control, mirroring `max_steps`; `UseSubAgentTool` forwards both to `spec.agent.run()`; `None` (default) preserves prior behaviour (#781)
- feat(ch09): `current_subagent_name` — `contextvars.ContextVar` set by `UseSubAgentTool` around a dispatched sub-agent's `run()`; `ColoredFormatter` prepends `[name]` to every log line emitted while that sub-agent is in flight, so console output distinguishes coordinator logs from sub-agent logs (#775)
- feat(ch09): `subagents/recipes.py` — `general_subagent()` (`DEFAULT_TOOLS`-equipped) and `explore_subagent()` (`ReadFileTool`-only, lower `max_steps`) default subagent factories, mirroring `memory/recipes.py`; use directly with `with_subagents([general_subagent(llm), ...])`, no builder sugar (#765)
- feat(ch09): `SharedConsoleHumanInputTool` — async sibling to `HumanInputTool` safe under concurrent tool execution; class-level `asyncio.Lock` serializes all prompts to a single stdin; optional `agent_name` label rendered in the panel title; extracts `_prompt_human` module-level helper shared by both classes (#747)
- feat(ch09): `LLMAgent.subagents_registry` param, `TaskHandler._use_subagent_tool`, `TaskHandler._subagents_catalog`, and `DEFAULT_SUBAGENTS_CATALOG` template — wires sub-agent registry and `<available_subagents>` catalog into the coordinator; `LLMAgentBuilder.with_subagent()` / `with_subagents()` fluent methods (#756, #746)
- feat(ch09): `SubAgentSpec.catalog()` and `CATALOG_SPEC_TEMPLATE` — XML catalog entry for a sub-agent, mirroring `Skill.catalog()` / `CATALOG_SKILL_TEMPLATE` from Ch06 (#756)
- feat(ch09): `UseSubAgentTool` — async tool dispatching tasks to named sub-agents; catches all exceptions (incl. `MaxStepsReachedError`) as error results (#755)
- feat(ch09): `SubAgentSpec` — Pydantic model registering a named sub-agent (name, description, agent, max_steps) in the `subagents/` package (#754)

### Changed

- refactor: rename `TaskHandler.skills` → `skills_registry` and `UseSkillTool` param `skills` → `skills_registry` — consistent with the `_registry` suffix convention for dict-keyed stores; updates `examples/ch06.ipynb` and affected `more-examples/ch06/` notebooks (#758)
- refactor: rename `UseSubAgentTool` constructor param `subagents` → `subagents_registry`; rename stored attribute to `_subagents_registry` — consistent with the `_registry` suffix convention for dict-keyed stores
- refactor: rename `LLMAgent.subagents` attribute → `subagents_registry` — consistent with `tools_registry` naming convention (#756)
- fix: `TaskHandler` parameterized as `asyncio.Future[TaskResult]` — `await agent.run(task)` now resolves to `TaskResult` rather than `Any` (#755)
- feat(ch04-retro): concurrent tool execution in `run_step` via `asyncio.gather` — async tools run concurrently; sync tools wrapped in `asyncio.to_thread`; result ordering preserved (#751)

## [0.0.21] - 2026-07-23

### Added

- feat: add `json_prompt_mode` flag to `OllamaLLM` — prompt-level JSON coercion for structured output, working around cloud models that ignore the Ollama `format` parameter; not covered in the book (#735)

## [0.0.20] - 2026-07-13

### Added

- feat: add `LLMAgent.run_supervised_with_skill()` — combines skill-activation framing with caller-controlled stepwise execution; returns a `SupervisedTaskHandler` ready for `get_next_step()` / `run_step()` (#719)
- feat: `SupervisedTaskHandler.complete()` raises `TaskHandlerError` at runtime when passed a non-`TaskResult` argument (#694)
- feat: add `LLMAgent.run_supervised()` and `LLMAgent.SupervisedTaskHandler` — human-driven stepwise execution; caller controls cadence via `get_next_step()` / `run_step()` and finalises with `complete()`, `reject()`, or `abort()` (#693)
- feat: introduce `RejectedTaskResult` data structure; `get_next_step` accepts `TaskStepResult | RejectedTaskResult | None` and dispatches by type — rejection routes deterministically to a new `TaskStep` without consulting the LLM; `_process_loop` captures raw `failed_result_content` + `feedback`; rejection template now includes `<proposed-result>` alongside `<human-correction>`; `HumanInputTool` styled with yellow `Panel` (#691)
- refactor: adopt `rich.prompt.Prompt` in `HumanInputTool`; add optional `choices` parameter for constrained input (#689)
- feat: implement end-of-loop human approval gate in `_process_loop` — adds `with_approval` param to `LLMAgent.run()`, `TaskHandler.request_approval()`, `TaskHandler._prompt_for_approval()`, `ApprovalResult` data structure, three approval template keys, and `rich>=13.9.0` dependency (#688)
- feat: implement HumanInputTool (#685)

### Changed

- fix: `KeyboardInterrupt` in `request_approval()` now auto-approves (consistent with `EOFError`) instead of rejecting (#713)
- refactor: remove `approval_prompt` and `approval_rationale_prompt` from `LLMAgentTemplates` — human-facing strings rendered by `rich`, not LLM prompts; hardcoded directly in `_prompt_for_approval()`; `TaskResult` now passed directly instead of its content string (#711)
- refactor: move `_prompt_for_approval` inside `request_approval()` as a local function; it was only ever called from there via `asyncio.to_thread` (#709)
- fix: guard `HumanInputTool.__call__()` against empty or absent `prompt` — returns an error `ToolCallResult` early; `"required"` in the JSON schema guarantees the key is present but not non-empty (#706)
- fix: add `from_scratch__` prefix to `HumanInputTool.name` to match the naming convention of the other default tools (#704)

## [0.0.19] - 2026-06-23

### Added

- feat: add `qdrant_point_to_episode()` to `memory_stores/qdrant/utils.py` — inverse of `episode_to_qdrant_point_struct()`; accepts `models.Record | models.ScoredPoint`; raises `QdrantPointPayloadMissingError` when payload is `None` or `QdrantEpisodeJsonMissingError` when `episode_json` key is absent (#643)
- feat: add `memory_stores/qdrant/errors.py` with `QdrantMemoryStoreError`, `QdrantPointPayloadError`, `QdrantPointPayloadMissingError`, `QdrantEpisodeJsonMissingError` (#643)

### Changed

- feat: rename BaseMemoryStore.search() to recall() (#661)
- feat: strengthen `DEFAULT_MEMORIES_BLOCK` template in `agent/templates/defaults.py` — adds `<warning>` wrapper and explicit instruction "if a past episode covers the same task, use its result directly unless instructed otherwise" (#660)
- feat: `TaskHandler.run_step()` appends recalled memories to the system message before the skills catalog — memories are injected at `agent/llm_agent.py` after the rollout block is composed, ensuring they appear in every step's system prompt (#660)
- fix: Memory.record() works on a deep copy of Episode (#650)
- chore: use TypeAlias for MetadataFn in memory.py (#647)
- refactor: `_read_recent()` uses `scroll(order_by=OrderBy(completed_at, DESC), limit=n)` — eliminates the `count()` round-trip and Python-side sort; float payload index on `completed_at` created unconditionally in `_ensure_collection()` (new and pre-existing collections) for server-mode compatibility (#643)
- nit: rename JSONMemoryStore._rewrite() to_rewrite_jsonl() (#642)
- refactor: migrate `QdrantMemoryStore` to `AsyncQdrantClient`; collection creation moved to lazy `_ensure_collection()` with `_collection_ready` flag (#638)
- nit: rename `_collection` to `_collection_name` in `QdrantMemoryStore` (#637)
- feat: add `MaxResultsExceededWarning`; `BaseMemoryStore.search()` emits it when `_search` or `_read_recent` returns more results than `max_results` (#632)
- refactor: make _read_recent and _search internal; dispatch in BaseMemoryStore.search() (#630)
- refactor: remove key from BaseMemoryStore interface; move key_fn to QdrantMemoryStore (#629)
- refactor: replace `EpisodeAttr` Literal and `include` param with `exclude: set[str] | None` in `Episode.format()`; fields iterated from `Episode.model_fields` so new fields appear automatically; `QdrantMemoryStore` migrates to `DEFAULT_EPISODE_EXCLUDE`; `error` field is now included in embeddings (#623)
- feat: introduce FormatMode enum for Episode.format() mode param (#620)
- feat: `Episode._format_concat` now prefixes each value with its field label (e.g. `instruction: ...`, `result: ...`); metadata entries use their key as label (#616)
- feat: add delete() and update() to BaseMemoryStore (#609)

## [0.0.18] - 2026-06-03

### Added

- feat: add `RecordMemoryError` to error hierarchy (`TaskHandlerError` subclass) (#601)
- feat: add `TaskHandler.record_memory(result, error)` — builds and persists an `Episode` before resolving the Future, so `await agent.run(task)` returns only after memory is written (#601)
- feat: add `Episode.error: Exception | None` field; `Episode.result` is now optional (#601)
- feat: add `template` param to `reflective_memory()` for custom reflection prompts (#600)
- feat: add `Memory.summary()` delegating to `store.summary()` (#595)
- feat: add `memory/recipes.py` with `recency_memory()`, `similarity_memory()`, `reflective_memory()` factory functions (#593)
- feat: `Memory.key_fn` is now optional, defaulting to `lambda ep: ep.task.instruction` (#593)
- feat: add `recall_mode` (`RecallMode` enum) to `BaseMemoryStore` — `RecallMode.RECENT` delegates `search()` to `read_recent()`; `RecallMode.SEARCH` performs similarity lookup (#590)
- feat: add `RecallMode(str, Enum)` to `data_structures.memory` (#590)
- feat: add `id_` UUID field to `Episode` (#588)

### Changed

- refactor: remove `BaseMemory` ABC and legacy strategy classes (`RecencyMemory`, `SimilarityMemory`, `ReflectiveMemory`); `LLMAgent` and `LLMAgentBuilder` now accept `list[Memory]` directly (#602)
- refactor: move `BaseMemoryStore` to `base/memory_store.py`; `base/memory.py` now contains only `BaseMemory` (#587)
- refactor: move store implementations into new `memory_stores/` package — `json.py`, `qdrant/store.py`, `qdrant/utils.py` (#587)

- refactor: introduce `Memory` concrete class with `key_fn` + `metadata_fns`; replaces `BaseMemory` hierarchy (#586)
- refactor: `Episode.additional_data` renamed to `metadata` (`dict[str, str]`, non-optional, default `{}`) (#586)
- refactor: `BaseMemoryStore.write()` `embedded_text` param renamed to `key` (#586)
- refactor: `BaseMemoryStore.search()` `k` param removed; stores use `self.max_results` set at construction (#586)
- refactor: `BaseMemoryStore` gains `max_results: int = 5` init param (#586)

## [0.0.17] - 2026-05-23

### Changed

- refactor: `BaseMemoryStore.write()` accepts `embedded_text: str | None` so memory strategies own embedding text (#562)
- refactor: `episode_to_qdrant_point_struct()` takes pre-formatted `text: str`; qdrant params keyword-only (#562)
- refactor: add `DEFAULT_EPISODE_INCLUDE` to `QdrantMemoryStore`; remove init-level format params (#562)
- refactor: add `episode_to_qdrant_point_struct()` converter in `qdrant_utils.py` (#559)
- chore: migrate `QdrantMemoryStore` off deprecated `add` and `query` methods (#557)
- fix: update `recency_memory.ipynb` summary output cells to show delegated format (#556)
- refactor: `JSONMemoryStore` now takes `dir` + `filename` params instead of a single `path` (#547)

### Added

- feat: implement ReflectiveMemory in memory/reflective.py (#561)
- feat: implement `ReflectiveMemory` in `memory/reflective.py` (#527)
- feat: add `Episode.additional_data` field and `Episode.format(mode, include)` with `EpisodeAttr` Literal type (#562)
- feat: add `tqdm` to `notebooks` optional extra for async progress bars (#559)
- bonus-example: `more-examples/ch07/similarity_memory.ipynb` — SimilarityMemory + QdrantMemoryStore with REST Countries API (#554)
- feat: implement `SimilarityMemory` and add `summary()` to `BaseMemoryStore`, `JSONMemoryStore`, and `QdrantMemoryStore` (#550)
- feat: implement QdrantMemoryStore (#548)
- feat: add count() to store (#545)
- feat: implement `RecencyMemory` in `memory/recency.py` (#543)
- feat: implement `JSONMemoryStore` in `memory/json_store.py` (#542)
- feat: add `with_memory()` and `with_memories()` to `LLMAgentBuilder` (#541)
- feat: wire memory recall and record into `LLMAgent` and `TaskHandler`
- feat: add `BaseMemory` and `BaseMemoryStore` abstract base classes (#539)
- feat: add `Episode` data structure to `data_structures.memory` (#535)

## [0.0.16] - 2026-05-10

### Added

- feat: add `code` and `cwd` parameters to `PythonInterpreterTool` (#490)
- feat: add `stdin` parameter to `PythonInterpreterTool` (#487)

### Changed

- refactor: rename rollout contribution method, templates, and local vars (#502)
- refactor: rip out disable_model_invocation from LLMAgent (#456)
- refactor: outsource SKILL_SUBDIR to skills/constants (#453)
- Refactor _SKILL_SUBDIRS to use .agents/skills only (#452)
- refactor: rename info attr to frontmatter (#450)
- refactor move disable model invocation from skillinfo to skill (#448)
- refactor: rename info attr to frontmatter (#450)

## [0.0.15] - 2026-03-30

### Added

- feat: add run _with_skill() for user explicit skill activation (#441)
- refactor: replace include_default_tools with TOOLS_FOR_SKILL_RESOURCES (#440)
- feat: add ReadFileTool as default tool with opt-out (#436)
- feat: implement UseSkillTool and Skill.resources (#434)
- feat: handle `disable-model-invocation` flag and shadowing warnings (#429)
- feat: inject skills catalog into `TaskHandler` system prompt (#423)
- feat: add `skills_scopes` to `LLMAgent` and skill discovery to `TaskHandler` (#424)
- refactor: introduce `SkillScope` enum and `get_skills_path()` utility (#422)
- feat: implement `Skill.read_body()` (#419)
- feat: implement `Skill.catalog()` (#418)
- feat: implement validate_skill_dir() (#417)
- feat: skills data structures, errors and skill construct (#412)

### Removed

- chore: remove `SkillInfo.from_skill_md()` (#420)

### Changed

- fix: add clean up of session_ready_task (#378)

## [0.0.14] - 2026-02-22

### Changed

- fix: MCPToolProvider.session() fails silently (#376)
- refactor: use prev result content if final result LLMAgent.get_next_step() (#375)
- feat: add headers params for streamable http (#369)
- fix: outdated naming convetion for tools MCPTool.__call__() (#359)
- refactor: simplify MCPToolProvider session initialization (#353)
- refactor: Use persistent sessions for MCPProviders (#338)
- refactor: use same MCP tool namespacing as Claude Code (#337)

## [0.0.13] - 2026-01-24

### Added

- feat: Add LLMAgentBuilder (#329)
- feat: implement MCPTool and unit tests (#328)
- feat: MCPToolProvider and start of MCPTool (#326)

### Changed

- fix: use same error handling as simple function tool for PydanticFunctionTool (#310)
- fix: OpenAILLM.continue_chat_with_tool_results() (#304)
- fix: pass kwargs to client construction for OpenAILLM (#301)

## [0.0.12] - 2025-12-20

### Added

- feat: bonus material add OpenAILLM (#296)
- feat: add optional notebook-utils (#290)

### Changed

- feat: set_dataframe_display_options to notebook_utils api + docker adds notebook-utils extra (#291)
- feat!: no think in ollama llm structured output (#285)
- feat!: add think param to OllamaLLM (#284)
- refactor: rollouts as monologue (#280)
- fix: better default prompts and handle tool calls in final content (#279)

## [0.0.11] - 2025-11-26

### Added

- Add string representations of Task and TaskStep (#248)
- feat: Add tools param to continue_chat_with_tool_results for BaseLLM and OllamaLLM (#215)

### Changed

- refactor!: Remove SkipJsonSchema from id_ annotation for TaskStep (#259)
- refactor!: rip out TaskHandlerTemplates (#250)
- refactor!: move task_handler_templates up to __init__() (#249)
- refactor: make get_tool_json_schema() internal (#207)
- Remove __str__() impl for CompleteResult (#204)

## [0.0.10] - 2025-09-18

### Added

- Add LLM type alias for BaseLLM to match Tool alias (#132)
- Add Tool type alias for BaseTool | AsyncBaseTool (#131)
- Add host param to OllamaLLM construction to properly connect the internal AsyncClient to it. (#125)
- [Feature] Add id_to ToolCall and tool_call_id to ToolCallResult (#119)

### Changed

- [feat] More specific error handling with simple function tools (#184)
- refactor: Store custom desc in _desc for SimpleFunctionTool and AsyncSimpleFunctionTool. (#181)
- Remove return_history param in OllamaLLM.chat() (#126)
- Rename continue_conversation_with_tool_results to continue_chat_with_tool_results (#123)
- Removed `tool_call: ToolCall` attr from `ToolCallResult` (#119)

## [0.0.9] - 2025-08-09

### Changed

- Renamed a few required keys in `TaskHandlerTemplates` (#117)
- [docs] Book version of hailstone.ipynb (#111)
- [docs] Store previous runs in a separate section in `hailstone.ipynb` (#110)

### Added

- [Feature] Add LLMAgentTemplates and add as an attribute to LLMAgent (#117)
- [docs] Add Qwen2.5-72b run to hailstone.ipynb (#108)
- [docs] Add trajectory evaluation hailstone.ipynb (#106)
- [docs] Add eval report for evaluation of final result outcomes (#105)

## [0.0.8] - 2025-08-02

### Added

- Add `TaskHandler.step_counter` (#103)
- [docs] Add simple benchmark and llm as judge for `hailstone.ipynb` (#102)

### Changed

- Add task demarcation in `TaskHandler.rollout` and better tool call requests formats (#100)
- Add `max_msg_length` for log formatter (#99)
- Improved `TaskHandler.rollout` formatting (#96)
- Remove `with_task_id()` from `TaskStep` and `TaskResult` (#95)

## [0.0.7] - 2025-07-29

### Added

- Add `max_steps` to `LLMAgent.run` and set handler result to `MaxStepsReachedError` if reached (#91)

### Changed

- Remove TaskHandlerResult and use TaskResult directly (#93)
- Improve `NextStepDecision` to allow for only one next_step or task_result (#88)

## [0.0.6] - 2025-07-27

### Added

- Add templates for `_rollout_contribution_from_single_run_step` (#81)
- Add `with_task_id()` to `TaskResult` and `TaskStep` (#77)
- Add `SkipJsonSchema` annotation to `id_` for `TaskStep` (#77)

### Changed

- Update `BaseLLM.chat` and `BaseLLM.continue_conversation_with_tool_call_results` for better consistency (#84)
- Refactor: Change LLMAgent.run helper method _run name to_process_loop (#83)
- Remove TaskHandler._lock since we actually don't need it (#79)
- [Fix] Move check for previous_step_result at top of method (#76)
- Rename `llm_agents_from_scratch.agent.core` to `llm_agents_from_scratch.agent.llm_agent` (#74)

## [0.0.5] - 2025-07-24

### Changed

- Nest `TaskHandler` within `LLMAgent` (#72)
- Remove error from TaskResult and rename GetNextStep to NextStepDecision (#69)

### Added

- Add `__str__` to `TaskStepResult` (#70)
- Add ids to Task, TaskStep, and results (#67)

## [0.0.4] - 2025-07-23

### Added

- Add `tool_registry` to `LLMAgent` and raise `LLMAgentError` for duplicated tools (#65)
- Add classmethod `ChatMessage.from_tool_call_result` (#61)

### Changed

- [Fix] `LLMAgent.tools` should be list of `BaseTool | AsyncBaseTool` (#64)
- Fix: `AsyncPydanticFunctionTool` should inherit from `AsyncBaseTool` (#63)
- Use `param.kind` in instrospection for `function_signature_to_json_schema` (#62)
- Removed `llm_agents_from_scratch.llms.ollama.utils.tool_call_result_to_chat_message` (#61)

## [0.0.3] - 2025-07-10

### Changed

- Rename `llm_agents_from_scratch.core` to `llm_agents_from_scratch.agent` (#55)
- Revised `TaskHandler.get_next_step()` to return `TaskStep | TaskResult` (#54)
- Fixed bug in `OllamaLLM.chat()` where chat history was coming after user message (#51)
- Fixed bug in `TaskHandler.run_step()` where tool names were passed to `llm.chat()` (#46)

### Added

- Add `~agent.templates` module and add `TaskHandler.templates` attribute (#55)
- Add `enable_console_logging` and `disable_console_logging` to not stream logs as a library by default (#54)
- Add first working cookbook for a simple `LLMAgent` and task (#54)
- Add `data_structures.task_handler.GetNextStep` (#54)
- Add `logger.set_log_level()` and logger attribute to `TaskHandler` (#51)
- Added library logger `llm_agents_from_scratch.logger` (#50)
- Remove `OllamaLLM` from root import -- too slow! (#45)
- `OllamaLLM` and `.tools` to root import (#44)

## [0.0.2] - 2025-07-05

### Changed

- Update `TaskHandler.run_step()` to work with updated `continue_conversation_with_tool_results` (#39)
- Update return type of `continue_conversation_with_tool_results` to `list[ChatMessage]` (#38)

### Deleted

- Delete `llms.ollama.utils.tool_call_result_to_ollama_message` (#38)

### Added

- Add `llms.ollama.utils.tool_call_result_to_chat_message` (#38)
- First implementation of `TaskHandler.run_step()` (#35)
- Implement `TaskHandler.get_next_step()` (#33)
- Add `BaseLLM.structured_output()` and impl for `OllamaLLM` (#34)
- Add `AsyncPydanticFunctionTool` (#30)
- Add `PydanticFunctionTool` (#28)

## [0.0.1] - 2025-07-01

### Added

- Add `AsyncSimpleFunctionTool` (#20)
- Rename `FunctionTool` to `SimpleFunctionTool` (#19)
- Implement `__call__` for `FunctionTool` (#18)
- Add simple function tool that allows for passing as an LLM tool (#16)
- Add tools to `OllamaLLM.chat` request and required utils (#14)
- Add initial implementation of `OllamaLLM` (#11)
- Add implementation of `base.tool.BaseTool` and relevant data structures (#12)
- Add `tools` to `LLM.chat` and update relevant data structures (#8)
- Add scaffolding for `TaskHandler` (#6)
- Add `LLMAgent` and associated data structures (#6)
