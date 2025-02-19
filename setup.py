from setuptools import setup, find_packages

setup(
    name="autospmv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pytest>=6.0.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
    ],
    author="Austin Mann",
    description="Automatic GPU kernel optimization using LLM guidance",
    python_requires=">=3.8",
)
