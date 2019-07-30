from __future__ import print_function
from setuptools import setup

description = "Simple and effective tools for the analysis of movement data"


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="movekit",
    version="0.1",
    author="Arjun Majumdar, Eren Cakmak, Jolle Jolles",
    author_email="arjun.majumdar@uni-konstanz.de, eren.cakmak@uni-konstanz.de, jjolles@orn.mpg.de",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbvis-ukon/movekit",
    packages=['movekit',
              'movekit.features',
              'movekit.io',
              'movekit.preprocessing'],
    install_requires=[
        'pandas',
        'scipy',
        'tsfresh'
    ],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"],
)
