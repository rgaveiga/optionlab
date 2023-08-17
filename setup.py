import os.path
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__),fname)).read()

setup(name='optionlab',
      version='0.1.5',
      description="A Python library for evaluating option trading strategies.",
      author='Roberto Gomes de Aguiar Veiga',
      url="https://github.com/rgaveiga/optionlab",
      packages=['optionlab'])


