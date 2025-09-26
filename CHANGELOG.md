<!-- markdownlint-disable-file MD024 -->

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

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
