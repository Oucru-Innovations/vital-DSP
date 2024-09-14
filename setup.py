from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        self.display_message()

    @staticmethod
    def display_message():
        print("\n🌟 Thank you for installing vitalDSP! 🌟")
        print("Explore the full potential of this toolkit and stay up-to-date with the latest features.")
        print("🔗 Visit our GitHub repository: https://github.com/Oucru-Innovations/vital-DSP")
        print("🚀 For updates, contributions, or to report issues, head over to GitHub!")
        print("Your feedback and contributions make vitalDSP better for everyone. Happy coding! 💻\n")

setup(
    name='vitalDSP', 
    version='0.1.1rc14',  
    author='van-koha',  
    author_email='vital.data@oucru.org', 
    description='A comprehensive toolkit for Digital Signal Processing in healthcare applications.',
    long_description=open('README.md').read(),  # Use the README file as the long description
    long_description_content_type='text/markdown',  # The format of the long description (markdown in this case)
    url='https://github.com/Oucru-Innovations/vital-DSP',  
    packages=find_packages(where='src',include=['vitalDSP', 'vitalDSP.*', 'webapp', 'webapp.*']),  # Find all packages in the 'src' directory
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
        'scikit-learn>=0.23.2',
        'dash>=2.0.0',  # Updated to include Dash
        'fastapi>=0.68.0',  # Include FastAPI for APIs
        'uvicorn>=0.15.0'   # Uvicorn for serving FastAPI
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'vitalDSP-webapp = vitalDSP.webapp.run_webapp:main',  # Entry point for the Dash web app
            # 'command_name = module:function',  # Add command-line tools here
        ],
    },
    include_package_data=True,  # Include data from MANIFEST.in
    cmdclass={'install': PostInstallCommand},
)
