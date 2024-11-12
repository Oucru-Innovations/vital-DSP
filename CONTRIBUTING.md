
# Contributing to vitalDSP

Thank you for considering contributing to **vitalDSP**! This document provides guidelines to help you contribute effectively, whether by reporting issues, suggesting features, or submitting code changes.

---

## Table of Contents
- [Contributing to vitalDSP](#contributing-to-vitaldsp)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Features](#suggesting-features)
    - [Improving Documentation](#improving-documentation)
    - [Contributing Code](#contributing-code)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Running Tests](#running-tests)
  - [Creating a Pull Request](#creating-a-pull-request)

---

## Code of Conduct

Please follow Python's Core Guidelines
Adhere to Python's [PEP 8](https://peps.python.org/pep-0008/) for code style, [PEP 257](https://peps.python.org/pep-0257/) for docstring conventions, and strive to write clean, maintainable code.

---

## How to Contribute

### Reporting Bugs

If you find a bug, please help us by [opening an issue](https://github.com/Oucru-Innovations/vital-DSP/issues) and including:
- A clear title and description.
- Steps to reproduce the problem.
- Expected and actual results.
- The version of vitalDSP and Python you’re using.

### Suggesting Features

We welcome feature suggestions! You can:
- Open an issue labeled **Feature Request** with a description of your suggestion.
- Share why this feature is beneficial, including any potential use cases.
- If possible, provide examples or ideas for implementation.

### Improving Documentation

Documentation improvements are always welcome! Here’s how you can contribute:
- Fix typos, improve explanations, or add missing information.
- Add examples or usage cases to help other users understand features.
- Update the README or **ReadTheDocs** documentation.

### Contributing Code

Before you start coding, consider:
1. Checking for an existing issue or creating a new one to discuss your proposed change.
2. Reviewing existing pull requests to avoid duplicating work.
3. Keeping contributions focused on one specific area to make them easier to review.

---

## Setting Up the Development Environment

1. **Fork the Repository**  
   Start by forking the [vitalDSP repository](https://github.com/Oucru-Innovations/vital-DSP) to your GitHub account.

2. **Clone Your Fork**  
   Clone your fork to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vital-DSP.git
   cd vital-DSP
   ```

3. **Install Dependencies**  
   vitalDSP requires certain Python packages for development. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Package in Editable Mode**  
   Install vitalDSP in editable mode to allow direct modifications without reinstallation:
   ```bash
   pip install -e .
   ```

---

## Running Tests

Testing is essential to ensure the reliability of vitalDSP. Before submitting changes, make sure all tests pass:

1. **Install Test Dependencies**  
   Install testing tools if not already installed:
   ```bash
   pip install pytest coverage
   ```

2. **Run Tests**  
   Run the full test suite to ensure code quality:
   ```bash
   pytest tests/
   ```

3. **Check Code Coverage**  
   Check the coverage of your tests with:
   ```bash
   coverage run -m pytest tests/
   coverage report -m
   ```

---

## Creating a Pull Request

1. **Create a New Branch**  
   Create a branch specific to your contribution. Use a descriptive name for your branch (e.g., `feature-new-filter` or `bugfix-amplitude-computation`):
   ```bash
   git checkout -b your-branch-name
   ```

2. **Make Your Changes**  
   Commit small, logical changes with meaningful commit messages.

3. **Push to Your Fork**  
   Push your branch to your forked repository:
   ```bash
   git push origin your-branch-name
   ```

4. **Submit a Pull Request (PR)**  
   Go to the [original repository](https://github.com/Oucru-Innovations/vital-DSP) and open a pull request.
   - Link to any related issues.
   - Provide a summary of your changes and why they’re necessary.
   - Be open to feedback and make any necessary changes as requested by reviewers.

---

Thank you for contributing to vitalDSP! Your help in improving this library is greatly appreciated.
