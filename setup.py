from setuptools import setup, find_packages

setup(
    name="lilgym", 
    version="0.0.1", 
    packages=find_packages(include=['lilgym', 'lilgym.*']),
    package_data={"lilgym.data": ["*.json"]}
)
