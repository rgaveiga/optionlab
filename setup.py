import os.path
from setuptools import setup

from optionlab import VERSION


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="optionlab",
    version=VERSION,
    description="A Python library for evaluating option trading strategies.",
    long_description_content_type="text/markdown",
    long_description=read("Desc.MD"),
    author="Roberto Gomes de Aguiar Veiga",
    url="https://github.com/rgaveiga/optionlab",
    packages=["optionlab"],
)
