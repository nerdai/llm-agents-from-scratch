<!-- markdownlint-disable-file MD024 -->

# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

- ...

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
