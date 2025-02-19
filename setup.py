from setuptools import setup, find_packages

setup(
    name="autospmv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'pytest>=7.0.0',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0'
    ]
)
