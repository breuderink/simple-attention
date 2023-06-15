from setuptools import setup, find_packages

setup(
    name="simple-attention",
    version="1.1.0",
    author="Boris Reuderink",
    author_email="boris@cortext.nl",
    description="Simple attention modules for PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/breuderink/simpleattention",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
