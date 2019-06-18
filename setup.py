import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evolutionary-ts-forecast",
    version="0.0.1",
    author="Ruy Brito",
    author_email="rbb3@cin.ufpe.br",
    description="Evolutionary algorithm for time-series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruybrito106/evolutionary-ts-forecast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent",
    ],
)
