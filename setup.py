from setuptools import setup, find_packages

setup(
    name='vital-DSP', 
    version='0.1.1rc1',  
    author='van-koha',  
    author_email='vital.data@oucru.org', 
    description='A comprehensive toolkit for Digital Signal Processing in healthcare applications.',
    long_description=open('README.md').read(),  # Use the README file as the long description
    long_description_content_type='text/markdown',  # The format of the long description (markdown in this case)
    url='https://github.com/Oucru-Innovations/vital-DSP',  
    packages=find_packages(where='src'),  # Find all packages in the 'src' directory
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
    install_requires=[
        'numpy>=1.19.2',
        'scipy>=1.5.2',
        'plotly>=4.14.3',
        'pywt>=1.0.10',
        'scikit-learn>=0.23.2',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            # 'command_name = module:function',  # Add command-line tools here
        ],
    },
    include_package_data=True,  # Include data from MANIFEST.in
)