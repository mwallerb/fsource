"""
Setup script for fsource

Copyright 2019 Markus Wallerberger.
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function, absolute_import
import io
import os.path
import re
from setuptools import setup, find_packages

HEREPATH = os.path.abspath(os.path.dirname(__file__))
VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", re.M)

def readfile(*parts):
    fullpath = os.path.join(HEREPATH, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()

def extract_version(*parts):
    initfile = readfile(*parts)
    match = VERSION_RE.search(initfile)
    return match.group(1)

LONG_DESCRIPTION = readfile('README.md')
VERSION = extract_version('fsource', '__init__.py')

setup(
    name='fsource',
    version=VERSION,

    description='Static analysis tools for Fortran, written in pure Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=' '.join([
        'fortran',
        'lexer',
        'parser',
        'analysis'
        ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        ],

    url='https://github.com/mwallerb/fsource',
    author='Markus Wallerberger',
    author_email='markus.wallerberger@tuwien.ac.at',

    python_requires='>=2.7, <4',
    install_requires=[],
    extras_require={
        'dev': ['pytest'],
        },

    packages=find_packages(exclude=['bin', 'contrib', 'doc', 'test']),
    entry_points={
        'console_scripts': [
            'fsource=fsource.__main__:main',
            ],
        },
    )
