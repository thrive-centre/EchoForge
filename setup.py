from setuptools import setup, find_packages

setup(
    name="echoforge",
    version="0.1",
    packages=find_packages(),
    install_requires=["tensorflow>=2.10", "requests"],
    description="Echo2DClassifier - 47-class echocardiographic view classifier",
    author="Mayur Agrawal"
)
