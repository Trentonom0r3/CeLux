from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ffmpy",
    version="0.2.4.1",
    author="Trenton Flanagan",
    author_email="spigonvids@gmail.com",
    description="HW accelerated video reading for ML Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Trentonom0r3/ffmpy",  # Update with the correct GitHub URL
    packages=find_packages(),
    package_data={
        "ffmpy": ["*.pyd", "*.dll"],  # Include the .pyd and .dll files
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",  # Updated license classifier
        "Operating System :: Microsoft :: Windows",  # Updated to indicate it's Windows-specific
    ],
    python_requires=">=3.11",
    include_package_data=True,
)
