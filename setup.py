from setuptools import setup, find_packages

setup(
    name="echoforge",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={"echoforge": ["config/*.json"]},
    install_requires=["tensorflow>=2.10", "requests", "huggingface_hub"],
    description="EchoForge - pretrained deep learning models for echocardiographic image analysis",
    author="Thrive",
)
