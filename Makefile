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

diagrams:	## Generate UML diagrams from PlantUML files
	@echo "Generating UML diagrams..."
	@mkdir -p uml/rendered
	@find uml -name "*.puml" -not -path "uml/common/*" -exec dirname {} \; | sed 's|^uml|uml/rendered|' | sort -u | xargs mkdir -p
	@find uml -name "*.puml" -not -path "uml/common/*" -exec sh -c 'java -jar ~/plantuml.jar -SDPI=300 -o "$$(dirname "{}" | sed "s|^uml|$(PWD)/uml/rendered|")" "{}"' \;
	@echo "Diagrams generated in uml/rendered/ directory with chapter structure!"

diagrams-svg:	## Generate SVG diagrams (for web)
	@echo "Generating SVG diagrams..."
	@mkdir -p uml/rendered
	@find uml -name "*.puml" -not -path "uml/common/*" -exec dirname {} \; | sed 's|^uml|uml/rendered|' | sort -u | xargs mkdir -p
	@find uml -name "*.puml" -not -path "uml/common/*" -exec sh -c 'java -jar ~/plantuml.jar -tsvg -o "$$(dirname "{}" | sed "s|^uml|$(PWD)/uml/rendered|")" "{}"' \;
	@echo "SVG diagrams generated in uml/rendered/ directory with chapter structure!"
