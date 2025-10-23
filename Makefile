# Define variables
TEST_DIR=tests
COV_DIR=cov_html
DOCS_DIR=docs
SRC_DIR=src
DIST_DIR=dist
BUILD_DIR=build
SPHINXBUILD = sphinx-build
SOURCEDIR = source
WEBAPP_DIR=$(SRC_DIR)/webapp
UPLOAD_DIR=$(WEBAPP_DIR)/data/uploads  # Upload folder
RESULTS_DIR=$(WEBAPP_DIR)/data/results  # Results folder
DOCBUILDDIR = $(DOCS_DIR)/_build
PANDOC_FILE=$(DOCS_DIR)/Documentation.md
PANDOC_OUTPUT=$(DOCS_DIR)/Documentation.pdf

# Default target: Run all tests
all: test build coverage lint html upload webapp

# Fast test targets for CI/CD
test-fast:
	python run_tests.py fast

test-ci:
	python run_tests.py ci

test-unit:
	python run_tests.py unit

test-core:
	python run_tests.py core

test-webapp:
	python run_tests.py webapp

test-coverage:
	python run_tests.py coverage

test-full:
	python run_tests.py full

test-parallel:
	python run_tests.py parallel

# Use conditional syntax to handle different OS
ifeq ($(OS),Windows_NT)
    # PYTHONPATH_SET = set PYTHONPATH=$(SRC_DIR) &&
	# PYTHONPATH_SET = $$env:PYTHONPATH="$(SRC_DIR)" &&
else
    PYTHONPATH_SET = PYTHONPATH=$(SRC_DIR)
endif

test:
	$(PYTHONPATH_SET) python -m pytest $(TEST_DIR) -v

# $(PYTHONPATH_SET) pytest --cov=$(SRC_DIR) --cov-config=.coveragerc --cov-report=html:$(COV_DIR)
# Run tests with coverage using .coveragerc
coverage:	
	python -m pytest --cov=$(SRC_DIR) --cov-config=.coveragerc --cov-report=term-missing --cov-report=html:$(COV_DIR) 

# Lint the code using flake8 with custom config
# Note: F401 (unused imports) is ignored in .flake8 config
lint:
	python -m flake8 --config=.flake8 $(SRC_DIR)

# Build the distribution packages
build:
	python setup.py sdist bdist_wheel

# Upload the package to PyPI
upload:
	twine upload $(DIST_DIR)/*

# Build HTML documentation
html:
	$(SPHINXBUILD) -b html $(DOCS_DIR)/$(SOURCEDIR) $(DOCBUILDDIR)/html

# Run the FastAPI + Dash web app using Uvicorn
webapp:
	uvicorn src.webapp.run_webapp:fastapi_app --reload --host 0.0.0.0 --port 8000


# Make sure the upload/results folders do not exceed the limit
# storage-check:
# 	@du -sh $(UPLOAD_DIR) $(RESULTS_DIR) | awk '{ if($$1 > "1G") print "Warning: Storage limit exceeded, please clean up."; }'

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
