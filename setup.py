from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        self.display_message()

    @staticmethod
    def display_message():
        print("\nThank you for installing vitalDSP!")
        print("Explore the full potential of this toolkit and stay up-to-date with the latest features.")
        print("Visit our GitHub repository: https://github.com/Oucru-Innovations/vital-DSP")
        print("For updates, contributions, or to report issues, head over to GitHub!")
        print("Your feedback and contributions make vitalDSP better for everyone. Happy coding!\n")

setup(
    name='vitaldsp',
    version='0.2.3',
    author='van-koha',  
    author_email='vital.data@oucru.org', 
    description='A comprehensive toolkit for Digital Signal Processing in healthcare applications.',
    long_description=open('README.md', encoding='utf-8').read(),  # Use the README file as the long description
    long_description_content_type='text/markdown',  # The format of the long description (markdown in this case)
    url='https://github.com/Oucru-Innovations/vital-DSP',  
    packages=find_packages(where='src',include=['vitalDSP', 'vitalDSP.*', 'vitalDSP_webapp', 'vitalDSP_webapp.*']),  # Find all packages in the 'src' directory
    package_dir={'': 'src'},  # Tell setuptools that all packages are under 'src'
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta', 
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.9',  # Python version requirement
    package_data={
        'vitalDSP.health_analysis': ['feature_config.yml'],  # Include YAML file
        "vitalDSP.notebooks": ["*.csv"],  # specify the file patterns
    },
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pandas>=1.3.0',
        'plotly>=4.14.3',
        'scikit-learn>=0.24.0',
        'statsmodels>=0.12.0',
        'seaborn>=0.11.0',
        'PyYAML>=5.4',
        'Jinja2>=3.0',
        'psutil>=5.8.0',
    ],
    extras_require={
        'webapp': [
            'dash>=2.0.0',
            'fastapi>=0.68.0',
            'uvicorn>=0.15.0',
        ],
        'tensorflow': [
            'tensorflow>=2.8.0,<2.17.0; platform_system != "Darwin" and python_version < "3.13"',  # TensorFlow for Linux/Windows
            'tensorflow-macos>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine == "arm64" and python_version < "3.13"',  # Apple Silicon
            'tensorflow>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine != "arm64" and python_version < "3.13"',  # macOS Intel
            # For Python 3.13+, users need to manually install tf-nightly or use Python 3.8-3.12
        ],
        'pytorch': [
            'torch>=2.0.0',  # PyTorch for deep learning
            'torchvision>=0.15.0',  # Optional: for vision tasks
        ],
        'explainability': [
            'shap>=0.41.0',  # SHAP for model explainability
            'lime>=0.2.0',  # LIME for model interpretation
        ],
        'dl': [
            # Deep learning bundle with TensorFlow (default)
            'tensorflow>=2.8.0,<2.17.0; platform_system != "Darwin" and python_version < "3.13"',
            'tensorflow-macos>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine == "arm64" and python_version < "3.13"',
            'tensorflow>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine != "arm64" and python_version < "3.13"',
        ],
        'dl-pytorch': [
            # Deep learning bundle with PyTorch
            'torch>=2.0.0',
            'torchvision>=0.15.0',
        ],
        'dl-all': [
            # All deep learning frameworks
            'tensorflow>=2.8.0,<2.17.0; platform_system != "Darwin" and python_version < "3.13"',
            'tensorflow-macos>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine == "arm64" and python_version < "3.13"',
            'tensorflow>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine != "arm64" and python_version < "3.13"',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
        ],
        'all': [
            # Complete installation with all optional features
            'tensorflow>=2.8.0,<2.17.0; platform_system != "Darwin" and python_version < "3.13"',
            'tensorflow-macos>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine == "arm64" and python_version < "3.13"',
            'tensorflow>=2.8.0,<2.17.0; platform_system == "Darwin" and platform_machine != "arm64" and python_version < "3.13"',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'shap>=0.41.0',
            'lime>=0.2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'vitalDSP-webapp = vitalDSP.webapp.run_webapp:main',  # Entry point for the Dash web app
            # 'command_name = module:function',  # Add command-line tools here
        ],
    },
    include_package_data=True,  # Include data from MANIFEST.in
    cmdclass={'install': PostInstallCommand},
)
