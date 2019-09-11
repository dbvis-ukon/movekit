#!/usr/bin/env python
"""MOVEKIT
-Simple and effective tools for the analysis of movement data!
"""
from setuptools import setup, find_namespace_packages

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR = 0
MINOR = 1
MICRO = 3
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def setup_package():

    with open("README.md") as f:
        long_description = f.read()

    metadata = dict(
        name='movekit',
        maintainer="dbvis-ukon",
        maintainer_email="eren.cakmak@uni-konstanz.de",
        description="Simple and effective tools for the analysis of movement data",
        author="Arjun Majumdar, Eren Cakmak, Jolle Jolles",
        author_email="arjun.majumdar@uni-konstanz.de, eren.cakmak@uni-konstanz.de, jjolles@orn.mpg.de",
        version=VERSION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/dbvis-ukon/movekit",
        license='License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Mac OS-X"],
        packages=find_namespace_packages(
            include=['movekit.*'],
            exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        install_requires=['tsfresh>=0.12.0',
                          'pandas>=0.20.3,<=0.23.4',
                          'scipy>=1.3.1',
                          'xlrd>=1.2.0',
                          'seaborn>=0.9.0',
                          ],
        python_requires='>=3.5',
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
