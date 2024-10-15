from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ffmpy",
    version="0.2.4",
    author="Trenton Flanagan",
    author_email="spigonvids@gmail.com",
    description="HW accelerated video reading for ML Inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Trentonom0r3/ffmpy",  # Update with the correct GitHub URL
    packages=find_packages(),
    package_data={
        "ffmpy": ["*.pyd", "*.dll", "*.pyi", "*.py"],  # Include the .pyd and .dll files
    },
     classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",  # Updated license classifier
        "Operating System :: Microsoft :: Windows",  # Indicates Windows-specific package
        "Operating System :: Microsoft :: Windows :: Windows 10",  # Further specify Windows version if needed
        "Operating System :: Microsoft :: Windows :: Windows 11",  # Further specify Windows version if needed
    ],
    license="AGPL-3.0",
    license_files=("LICENSE",),
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,  # Ensures the wheel is not treated as pure Python
)
