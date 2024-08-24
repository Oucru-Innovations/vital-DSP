# Define variables
TEST_DIR=tests
COV_DIR=cov_html
DOCS_DIR=docs
SRC_DIR=src
SPHINXBUILD = sphinx-build
SOURCEDIR = source
BUILDDIR = $(DOCS_DIR)/_build
PANDOC_FILE=$(DOCS_DIR)/Documentation.md
PANDOC_OUTPUT=$(DOCS_DIR)/Documentation.pdf

# Default target: Run all tests
all: test coverage lint html

# Use conditional syntax to handle different OS
ifeq ($(OS),Windows_NT)
    PYTHONPATH_SET = set PYTHONPATH=$(SRC_DIR) &&
else
    PYTHONPATH_SET = PYTHONPATH=$(SRC_DIR)
endif

test:
	$(PYTHONPATH_SET) pytest $(TEST_DIR) -v

# Run tests with coverage
coverage:
	$(PYTHONPATH_SET) pytest --cov=$(SRC_DIR) --cov-report=html:$(COV_DIR)

# Lint the code using flake8 with custom config
lint:
	flake8 --config=.flake8 $(SRC_DIR)

# Build HTML documentation
html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# Generate documentation using pandoc
pandoc:
	pandoc $(PANDOC_FILE) -o $(PANDOC_OUTPUT)

# Clean up the generated files
clean:
	rm -rf $(COV_DIR) $(PANDOC_OUTPUT)
	rm -rf $(BUILDDIR)

# Phony targets
.PHONY: all test coverage lint html clean
