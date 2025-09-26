Contributing to VitalDSP
=========================

We welcome contributions to VitalDSP! This guide will help you get started with contributing to the project.

Contributing Overview
=====================

VitalDSP is an open-source project that benefits from community contributions. We welcome contributions in various forms:

* **Code Contributions**: Bug fixes, new features, performance improvements
* **Documentation**: Improvements to existing docs, new tutorials, examples
* **Testing**: Bug reports, test cases, validation datasets
* **Community**: Answering questions, helping other users, sharing use cases

Getting Started
===============

**Prerequisites**

* Python 3.8 or higher
* Git
* Basic understanding of signal processing
* Familiarity with Python development

**Development Setup**

1. **Fork the repository:**
   - Go to https://github.com/Oucru-Innovations/vital-DSP
   - Click "Fork" to create your own copy

2. **Clone your fork:**
   .. code-block:: bash
   
      git clone https://github.com/your-username/vital-DSP.git
      cd vital-DSP

3. **Add upstream remote:**
   .. code-block:: bash
   
      git remote add upstream https://github.com/Oucru-Innovations/vital-DSP.git

4. **Create virtual environment:**
   .. code-block:: bash
   
      python -m venv vitaldsp_dev
      source vitaldsp_dev/bin/activate  # On Windows: vitaldsp_dev\Scripts\activate

5. **Install development dependencies:**
   .. code-block:: bash
   
      pip install -r requirements.txt
      pip install -e .
      pip install -r requirements-dev.txt  # Development dependencies

6. **Run tests:**
   .. code-block:: bash
   
      pytest tests/

Development Workflow
====================

**Branch Strategy**

* **main**: Production-ready code
* **develop**: Integration branch for features
* **feature/**: Feature branches
* **bugfix/**: Bug fix branches
* **hotfix/**: Critical bug fixes

**Creating a Feature Branch**

1. **Update your local main:**
   .. code-block:: bash
   
      git checkout main
      git pull upstream main

2. **Create feature branch:**
   .. code-block:: bash
   
      git checkout -b feature/your-feature-name

3. **Make your changes:**
   - Write code
   - Add tests
   - Update documentation

4. **Commit your changes:**
   .. code-block:: bash
   
      git add .
      git commit -m "Add feature: brief description"

5. **Push to your fork:**
   .. code-block:: bash
   
      git push origin feature/your-feature-name

6. **Create pull request:**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the pull request template

**Code Style Guidelines**

**Python Style**

Follow PEP 8 style guidelines:

.. code-block:: python

   # Good
   def calculate_heart_rate(rr_intervals, sampling_rate):
       """Calculate heart rate from RR intervals.
       
       Parameters:
       -----------
       rr_intervals : array-like
           RR intervals in seconds
       sampling_rate : float
           Sampling rate in Hz
       
       Returns:
       --------
       float
           Heart rate in BPM
       """
       if len(rr_intervals) == 0:
           return 0.0
       
       mean_rr = np.mean(rr_intervals)
       heart_rate = 60.0 / mean_rr
       
       return heart_rate

   # Bad
   def calcHR(rr,sr):
       if len(rr)==0:return 0
       return 60/np.mean(rr)

**Documentation Style**

Use NumPy-style docstrings:

.. code-block:: python

   def extract_features(signal, sampling_rate):
       """Extract physiological features from signal.
       
       Parameters:
       -----------
       signal : array-like
           Input signal data
       sampling_rate : float
           Sampling rate in Hz
       
       Returns:
       --------
       dict
           Dictionary containing extracted features
       
       Raises:
       -------
       ValueError
           If signal is empty or sampling_rate is invalid
       
       Examples:
       --------
       >>> signal = np.random.randn(1000)
       >>> features = extract_features(signal, 1000)
       >>> print(features['mean'])
       """
       if len(signal) == 0:
           raise ValueError("Signal cannot be empty")
       
       if sampling_rate <= 0:
           raise ValueError("Sampling rate must be positive")
       
       # Implementation here
       pass

**Testing Guidelines**

**Test Structure**

Organize tests in the `tests/` directory:

.. code-block:: python

   # tests/test_signal_filtering.py
   import pytest
   import numpy as np
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   
   class TestSignalFiltering:
       """Test cases for SignalFiltering class."""
       
       def setup_method(self):
           """Setup test fixtures."""
           self.fs = 1000
           self.signal = np.random.randn(1000)
           self.sf = SignalFiltering(self.signal, self.fs)
       
       def test_bandpass_filter(self):
           """Test bandpass filtering."""
           filtered = self.sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
           
           assert len(filtered) == len(self.signal)
           assert not np.any(np.isnan(filtered))
           assert not np.any(np.isinf(filtered))
       
       def test_invalid_parameters(self):
           """Test invalid parameter handling."""
           with pytest.raises(ValueError):
               self.sf.bandpass_filter(low_cut=40.0, high_cut=0.5)
       
       def test_empty_signal(self):
           """Test empty signal handling."""
           empty_signal = np.array([])
           sf = SignalFiltering(empty_signal, self.fs)
           
           with pytest.raises(ValueError):
               sf.bandpass_filter(low_cut=0.5, high_cut=40.0)

**Test Coverage**

Maintain high test coverage:

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=vitalDSP tests/
   
   # Generate coverage report
   pytest --cov=vitalDSP --cov-report=html tests/

**Performance Testing**

Include performance tests for critical functions:

.. code-block:: python

   import time
   import pytest
   
   def test_filtering_performance():
       """Test filtering performance."""
       signal = np.random.randn(10000)
       sf = SignalFiltering(signal, 1000)
       
       start_time = time.time()
       filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
       end_time = time.time()
       
       processing_time = end_time - start_time
       assert processing_time < 1.0  # Should process in less than 1 second

**Integration Testing**

Test integration between modules:

.. code-block:: python

   def test_signal_processing_pipeline():
       """Test complete signal processing pipeline."""
       # Generate test signal
       signal = generate_test_signal()
       
       # Apply filtering
       sf = SignalFiltering(signal, 1000)
       filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
       
       # Extract features
       from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
       extractor = PhysiologicalFeatureExtractor(filtered, fs=1000)
       features = extractor.extract_features(signal_type="ECG")
       
       # Validate results
       assert 'qrs_duration' in features
       assert 'heart_rate' in features
       assert features['heart_rate'] is not None

Documentation Contributions
============================

**Documentation Types**

* **API Documentation**: Function and class docstrings
* **User Guides**: Tutorials and examples
* **Developer Documentation**: Architecture and design decisions
* **Release Notes**: Changes and new features

**Writing Documentation**

1. **Use clear, concise language**
2. **Include code examples**
3. **Explain the "why" not just the "what"**
4. **Keep documentation up to date**

**Documentation Structure**

.. code-block:: rst

   Module Name
   ===========
   
   Brief description of the module.
   
   Overview
   --------
   
   Detailed description of functionality.
   
   Key Features
   ~~~~~~~~~~~~~
   
   * Feature 1
   * Feature 2
   * Feature 3
   
   Usage Examples
   --------------
   
   .. code-block:: python
   
      from vitalDSP.module import Class
      
      # Example usage
      instance = Class()
      result = instance.method()
   
   API Reference
   -------------
   
   .. automodule:: vitalDSP.module
      :members:
      :undoc-members:
      :show-inheritance:

**Building Documentation**

.. code-block:: bash

   # Install documentation dependencies
   pip install sphinx sphinx-rtd-theme
   
   # Build documentation
   cd docs
   make html
   
   # View documentation
   open _build/html/index.html

Bug Reports
============

**Reporting Bugs**

When reporting bugs, include:

1. **Environment Information**
   * Python version
   * Operating system
   * VitalDSP version
   * Dependencies versions

2. **Bug Description**
   * Clear description of the issue
   * Steps to reproduce
   * Expected vs. actual behavior

3. **Code Example**
   * Minimal code that reproduces the issue
   * Error messages and stack traces

4. **Additional Context**
   * Screenshots if applicable
   * Related issues or discussions

**Bug Report Template**

.. code-block:: markdown

   **Bug Description**
   Brief description of the bug.
   
   **To Reproduce**
   Steps to reproduce the behavior:
   1. Go to '...'
   2. Click on '....'
   3. Scroll down to '....'
   4. See error
   
   **Expected Behavior**
   What you expected to happen.
   
   **Environment**
   - Python version: 3.9.7
   - OS: Ubuntu 20.04
   - VitalDSP version: 0.1.4
   
   **Code Example**
   ```python
   # Minimal code that reproduces the issue
   from vitalDSP.module import function
   result = function()
   ```
   
   **Error Message**
   ```
   Traceback (most recent call last):
     File "example.py", line 2, in <module>
       result = function()
   ValueError: Invalid parameter
   ```

Feature Requests
=================

**Requesting Features**

When requesting features, include:

1. **Use Case**
   * Why is this feature needed?
   * What problem does it solve?

2. **Proposed Solution**
   * How should the feature work?
   * What should the API look like?

3. **Alternatives**
   * What alternatives have you considered?
   * Why is this the best approach?

4. **Additional Context**
   * Related issues or discussions
   * Examples from other libraries

**Feature Request Template**

.. code-block:: markdown

   **Feature Description**
   Brief description of the feature.
   
   **Use Case**
   Describe the use case and why this feature is needed.
   
   **Proposed Solution**
   Describe how the feature should work.
   
   **Alternatives**
   Describe alternatives you've considered.
   
   **Additional Context**
   Add any other context about the feature request.

Code Review Process
====================

**Review Checklist**

Before submitting a pull request:

1. **Code Quality**
   - [ ] Code follows style guidelines
   - [ ] Functions have proper docstrings
   - [ ] Code is well-commented
   - [ ] No unused imports or variables

2. **Testing**
   - [ ] Tests are included
   - [ ] Tests pass
   - [ ] Test coverage is maintained
   - [ ] Edge cases are tested

3. **Documentation**
   - [ ] Documentation is updated
   - [ ] Examples are included
   - [ ] API documentation is complete

4. **Performance**
   - [ ] Performance is acceptable
   - [ ] Memory usage is reasonable
   - [ ] No performance regressions

**Review Process**

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code quality checks
   - Documentation builds

2. **Manual Review**
   - Code review by maintainers
   - Discussion of changes
   - Request for modifications

3. **Approval**
   - Changes approved
   - Pull request merged
   - Feature released

**Responding to Reviews**

* **Be responsive**: Address feedback promptly
* **Be open**: Consider suggestions constructively
* **Be clear**: Explain your reasoning
* **Be patient**: Review process takes time

Release Process
================

**Release Types**

* **Major Release**: Breaking changes, new features
* **Minor Release**: New features, backward compatible
* **Patch Release**: Bug fixes, minor improvements

**Release Checklist**

1. **Code Quality**
   - [ ] All tests pass
   - [ ] Code coverage maintained
   - [ ] Performance benchmarks pass

2. **Documentation**
   - [ ] Documentation updated
   - [ ] Release notes written
   - [ ] Examples updated

3. **Version Management**
   - [ ] Version number updated
   - [ ] Changelog updated
   - [ ] Git tags created

4. **Distribution**
   - [ ] Package built
   - [ ] PyPI upload
   - [ ] GitHub release created

**Creating a Release**

1. **Update version:**
   .. code-block:: bash
   
      # Update version in setup.py
      vim setup.py
      
      # Update version in __init__.py
      vim src/vitalDSP/__init__.py

2. **Update changelog:**
   .. code-block:: bash
   
      vim CHANGELOG.md

3. **Create release branch:**
   .. code-block:: bash
   
      git checkout -b release/v0.1.5
      git add .
      git commit -m "Release v0.1.5"
      git push origin release/v0.1.5

4. **Create pull request:**
   - Create PR from release branch
   - Review and merge

5. **Tag release:**
   .. code-block:: bash
   
      git tag v0.1.5
      git push origin v0.1.5

6. **Build and upload:**
   .. code-block:: bash
   
      python setup.py sdist bdist_wheel
      twine upload dist/*

Community Guidelines
=====================

**Code of Conduct**

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

* **Be respectful**: Treat everyone with respect
* **Be inclusive**: Welcome newcomers and diverse perspectives
* **Be constructive**: Provide helpful feedback
* **Be patient**: Remember that everyone is learning

**Communication Channels**

* **GitHub Issues**: Bug reports, feature requests
* **GitHub Discussions**: General discussions, questions
* **Email**: support@vitaldsp.com
* **Community Forum**: Coming soon

**Getting Help**

* **Documentation**: Check existing documentation first
* **Issues**: Search existing issues before creating new ones
* **Discussions**: Ask questions in GitHub discussions
* **Community**: Help other users when you can

**Recognition**

Contributors are recognized in:

* **CONTRIBUTORS.md**: List of all contributors
* **Release Notes**: Contributors for each release
* **GitHub**: Contributor statistics and activity

**Contributor Types**

* **Core Contributors**: Regular contributors with commit access
* **Contributors**: Occasional contributors
* **Reviewers**: Help with code reviews
* **Documenters**: Help with documentation
* **Testers**: Help with testing and bug reports

Getting Started as a Contributor
=================================

**First Contribution**

Good first contributions:

1. **Documentation**: Fix typos, improve examples
2. **Tests**: Add test cases for existing functions
3. **Bug Fixes**: Fix simple bugs
4. **Examples**: Add usage examples

**Finding Issues to Work On**

* **Good First Issue**: Labeled issues for newcomers
* **Help Wanted**: Issues needing community help
* **Bug**: Issues that need fixing
* **Enhancement**: Feature requests

**Mentorship**

* **Ask Questions**: Don't hesitate to ask for help
* **Request Reviews**: Ask for code reviews
* **Learn from Feedback**: Use feedback to improve
* **Share Knowledge**: Help others when you can

**Contributor Resources**

* **Development Guide**: This document
* **API Documentation**: Complete API reference
* **Examples**: Practical examples and tutorials
* **Community**: Connect with other contributors

**Recognition and Rewards**

* **Contributor Badges**: GitHub contributor badges
* **Release Credits**: Recognition in release notes
* **Community Status**: Recognition in community
* **Learning Opportunities**: Skill development

**Long-term Contribution**

* **Regular Contributions**: Consistent contributions
* **Code Reviews**: Help review others' code
* **Mentoring**: Help new contributors
* **Leadership**: Take on leadership roles

This contributing guide provides comprehensive information for contributing to VitalDSP. We look forward to your contributions!

For additional help or questions, please reach out to our community or support team.
