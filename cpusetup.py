import os
from setuptools import setup
from pathlib import Path

VERSION = os.getenv("CELUX_VERSION", "0.0.1")

setup(
    name="celux",
    version=VERSION,
    author="Trenton Flanagan",
    author_email="spigonvids@gmail.com",
    description="HW accelerated video reader for ML inference (CPU version)",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Trentonom0r3/celux",
    packages=["celux"],
    package_dir={"celux": "celux"},
    package_data={
       "celux": ["*.pyd", "*.dll", "*.pyi", "*.py", "py.typed"]

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: Microsoft :: Windows",
    ],
    license="AGPL-3.0",
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
)
