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

# PlantUML runner: prefer a jar at ~/plantuml.jar (kept current manually --
# distro packages lag badly, e.g. Ubuntu's apt plantuml is frozen at
# 1.2020.2), fall back to a PATH-installed `plantuml` binary. Override with
# `make diagrams PLANTUML=...`
PLANTUML ?= $(shell [ -f "$(HOME)/plantuml.jar" ] && echo "java -jar $(HOME)/plantuml.jar" || command -v plantuml)

diagrams:	## Generate SVG diagrams (for web)
	@$(PLANTUML) -version > /dev/null 2>&1 || { echo "plantuml not found: download a jar to ~/plantuml.jar or install it (brew install plantuml) and set PLANTUML=..."; exit 1; }
	@echo "Generating SVG diagrams..."
	@mkdir -p uml/rendered
	@find uml -name "*.puml" -not -path "uml/common/*" -exec dirname {} \; | sed 's|^uml|uml/rendered|' | sort -u | xargs mkdir -p
	@find uml -name "*.puml" -not -path "uml/common/*" -exec sh -c '$(PLANTUML) -tsvg -o "$$(dirname "{}" | sed "s|^uml|$(PWD)/uml/rendered|")" "{}"' \;
	@uv run python _scripts/fix_svg_background.py --rendered_dir uml/rendered
	@uv run python _scripts/add_svg_legend.py --rendered_dir uml/rendered
	@uv run python _scripts/set_svg_print_size.py --rendered_dir uml/rendered
	@echo "SVG diagrams generated in uml/rendered/ directory with chapter structure!"

diagrams-png:	## Also render a PNG next to every diagram SVG (needs `playwright install chromium`)
	@uv run _scripts/render_diagram_pngs.py --rendered_dir uml/rendered
