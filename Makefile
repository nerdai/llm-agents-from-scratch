help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (ruff).
	pre-commit install
	git ls-files | xargs pre-commit run ruff-format --files

lint:	## Run linters: pre-commit (ruff, mypy)
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:
	pytest tests -v --capture=no

coverage: # for ci purposes
	pytest --cov llm_agents_from_scratch --cov-report=xml tests

coverage-report: ## Show coverage summary in terminal
	coverage report -m

coverage-html: ## Generate HTML coverage report
	coverage html

# PlantUML runner: prefer the `plantuml` binary (e.g. `brew install plantuml`),
# fall back to a jar at ~/plantuml.jar. Override with `make diagrams PLANTUML=...`
PLANTUML ?= $(shell command -v plantuml 2>/dev/null || echo "java -jar $(HOME)/plantuml.jar")

diagrams:	## Generate SVG diagrams (for web)
	@$(PLANTUML) -version > /dev/null 2>&1 || { echo "plantuml not found: install it (brew install plantuml) or set PLANTUML=..."; exit 1; }
	@echo "Generating SVG diagrams..."
	@mkdir -p uml/rendered
	@find uml -name "*.puml" -not -path "uml/common/*" -exec dirname {} \; | sed 's|^uml|uml/rendered|' | sort -u | xargs mkdir -p
	@find uml -name "*.puml" -not -path "uml/common/*" -exec sh -c '$(PLANTUML) -tsvg -o "$$(dirname "{}" | sed "s|^uml|$(PWD)/uml/rendered|")" "{}"' \;
	@echo "SVG diagrams generated in uml/rendered/ directory with chapter structure!"
