from __future__ import print_function
from setuptools import setup

setup(
    name="movekit",
    version="0.1.1",
    author="Arjun Majumdar, Eren Cakmak, Jolle Jolles",
    author_email="arjun.majumdar@uni-konstanz.de, eren.cakmak@uni-konstanz.de, jjolles@orn.mpg.de",
    description="Simple and effective tools for the analysis of movement data",
    long_description="Movekit is a open-source software for the analysis of movement data. The package includes modules for preoprocessing, features extraction, and statistical analysis.",
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
