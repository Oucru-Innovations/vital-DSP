# Define variables
TEST_DIR=tests
COV_DIR=cov_html
DOCS_DIR=docs
SRC_DIR=src
PANDOC_FILE=$(DOCS_DIR)/Documentation.md
PANDOC_OUTPUT=$(DOCS_DIR)/Documentation.pdf

# Default target: Run all tests
all: test coverage lint pandoc

# Run tests with pytest
test:
	PYTHONPATH=$(SRC_DIR) pytest $(TEST_DIR) -v

# Generate code coverage report
coverage:
	PYTHONPATH=$(SRC_DIR) pytest --cov=$(SRC_DIR) --cov-report=html:$(COV_DIR)

# Lint the code using flake8 with custom config
lint:
	flake8 --config=.flake8 $(SRC_DIR)

# Generate documentation using pandoc
pandoc:
	pandoc $(PANDOC_FILE) -o $(PANDOC_OUTPUT)

# Clean up the generated files
clean:
	rm -rf $(COV_DIR) $(PANDOC_OUTPUT)

# Phony targets
.PHONY: all test coverage lint pandoc clean
