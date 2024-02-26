.PHONY: all format lint test spell_check spell_fix help clean

# Default target executed when no arguments are given to make.
all: help

test:
	poetry run python -u -m unittest discover

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES='./pyspark_ai'
lint format: PYTHON_FILES='./pyspark_ai'
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	poetry run black $(PYTHON_FILES) --check
	poetry run flake8 $(PYTHON_FILES)

format format_diff:
	poetry run black $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml $(PYTHON_FILES)

spell_fix:
	poetry run codespell --toml pyproject.toml -w $(PYTHON_FILES)

#########
# CLEAN #
#########

clean:
	rm -r spark-warehouse
	echo '' # to avoid failing builds if there is no such directory

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'spell_check                  - run spell check'
	@echo 'spell_fix                    - fix spelling errors'
	@echo 'clean                        - clean artifacts'
