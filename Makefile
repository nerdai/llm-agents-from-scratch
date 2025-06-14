help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black).
	pre-commit install
	git ls-files | xargs pre-commit run black --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:
	pytest tests -v --capture=no

coverage: # for ci purposes
	pytest --cov llm_agents_from_scratch --cov-report=xml tests

coverage-report: ## Show coverage summary in terminal
	coverage report -m

coverage-html: ## Generate HTML coverage report
	coverage html
