# Define variables
TEST_DIR=tests
COV_DIR=cov_html
DOCS_DIR=docs
SRC_DIR=src
DIST_DIR=dist
BUILD_DIR=build
SPHINXBUILD = sphinx-build
SOURCEDIR = source
DOCBUILDDIR = $(DOCS_DIR)/_build
PANDOC_FILE=$(DOCS_DIR)/Documentation.md
PANDOC_OUTPUT=$(DOCS_DIR)/Documentation.pdf

# Default target: Run all tests
all: test build coverage lint html upload

# Use conditional syntax to handle different OS
ifeq ($(OS),Windows_NT)
    # PYTHONPATH_SET = set PYTHONPATH=$(SRC_DIR) &&
	# PYTHONPATH_SET = $$env:PYTHONPATH="$(SRC_DIR)" &&
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

# Build the distribution packages
build:
	python setup.py sdist bdist_wheel

# Upload the package to PyPI
upload:
	twine upload $(DIST_DIR)/*

# Build HTML documentation
html:
	$(SPHINXBUILD) -b html $(DOCS_DIR)/$(SOURCEDIR) $(DOCBUILDDIR)/html

# Generate documentation using pandoc
pandoc:
	@echo "Pandoc target is not implemented."
# pandoc $(PANDOC_FILE) -o $(PANDOC_OUTPUT)

# Clean up the generated files
clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) $(SRC_DIR)/*.egg-info $(COV_DIR) $(SPHINX_DIR)/_build/
	rm -rf $(COV_DIR) $(PANDOC_OUTPUT)
	rm -rf $(DOCBUILDDIR)

# Phony targets
.PHONY: all test coverage lint html clean
