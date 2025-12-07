from setuptools import setup, find_packages
import pathlib
import io
import os

HERE = pathlib.Path(__file__).parent


def read_requirements(path=HERE / "requirements.txt"):
    """Read requirements.txt and return a list suitable for install_requires.

    Ignores blank lines and comments. If the file is missing, returns an empty list.
    """
    if not path.exists():
        return []
    reqs = []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


# Read long description from README if available
long_description = ""
readme = HERE / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")


setup(
    name="nifty50_forecasting_system",
    version="0.0.1",
    description="A forecasting system for Nifty 50 stock market index",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="rkpcode",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
)