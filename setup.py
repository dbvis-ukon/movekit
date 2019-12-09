#!/usr/bin/env python

from setuptools import setup, find_namespace_packages
import sys

exec(open("movekit/__version__.py").read())

DESCRIPTION="""Simple and effective tools for the analysis of movement data"""
DISTNAME="movekit"
AUTHOR="Arjun Majumdar, Eren Cakmak, Jolle Jolles"
AUTHOR_EMAIL="arjun.majumdar@uni-konstanz.de, eren.cakmak@uni-konstanz.de,"\
              "j.w.jolles@gmail.com"
MAINTAINER="dbvis-ukon"
MAINTAINER_EMAIL="eren.cakmak@uni-konstanz.de"
URL="https://github.com/dbvis-ukon/movekit"
DOWNLOAD_URL="https://github.com/JolleJolles/pirecorder/archive/v2.0.0.tar.gz"

with open("README.md") as f:
    readme = f.read()

if __name__ == "__main__":

    setup(name=DISTNAME,
          author=AUTHOR,
          autor_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=long_description,
          long_description_content_type="text/markdown",
          url=URL,
          install_requires=["tsfresh>=0.12.0",
                            "pandas>=0.20.3,<=0.23.4",
                            "scipy>=1.3.1",
                            "xlrd>=1.2.0",
                            "seaborn>=0.9.0"],
        version=__version__,

        url="https://github.com/dbvis-ukon/movekit",
        license="License :: OSI Approved :: GNU General Public License v3",
        platforms=["Windows", "Linux", "Mac OS-X"],
        packages=find_namespace_packages(
            include=['movekit.*'],
            exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        python_requires='>=3.5',
        classifiers=[
                    "Development Status :: 2 - Pre-Alpha",
                    "Intended Audience :: Science/Research",
                    "Intended Audience :: Developers",
                    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 3",
                    "Programming Language :: Python :: 3.5",
                    "Programming Language :: Python :: 3.6",
                    "Programming Language :: Python :: 3.7",
                    "Topic :: Software Development",
                    "Topic :: Scientific/Engineering",
                    "Operating System :: Microsoft :: Windows",
                    "Operating System :: Unix",
                    "Operating System :: MacOS"],
    )
