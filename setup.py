# A minimal setup.py file to make a Python project installable.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['numpy', 'scipy']
    
setuptools.setup(
    name             = "pejzero",
    version          = "0.1.0",
    author           = "Whyjay Zheng",
    author_email     = "whyjz@berkeley.edu",
    description      = 'Supplmental materials presented in "Characteristics of marine-terminating glaciers vulnerable to lubricated bed"',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = setuptools.find_packages(),
    classifiers       = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>= 3.8',
    install_requires = requirements,
)

# License to be decided:        "License :: OSI Approved :: BSD License",