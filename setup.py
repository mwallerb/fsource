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
DOCLINK_RE = re.compile(r"(?m)^\s*\[\s*([^\]\n\r]+)\s*\]:\s*(doc/[./\w]+)\s*$")

def readfile(*parts):
    fullpath = os.path.join(HEREPATH, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()

def extract_version(*parts):
    initfile = readfile(*parts)
    match = VERSION_RE.search(initfile)
    return match.group(1)

def rebase_links(text, base_url):
    result, nsub = DOCLINK_RE.subn(r"[\1]: %s/\2" % base_url, text)
    return result

VERSION = extract_version('src', 'fsource', '__init__.py')
REPO_URL = "https://github.com/mwallerb/fsource"

# Make sure links work on the PyPI package
_DOCTREE_URL = REPO_URL + "/tree/master"
LONG_DESCRIPTION = rebase_links(readfile('README.md'), _DOCTREE_URL)

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

    url=REPO_URL,
    author='Markus Wallerberger',
    author_email='markus.wallerberger@tuwien.ac.at',

    python_requires='>=2.7, <4',
    install_requires=[],
    extras_require={
        'dev': ['pytest', 'pylint'],
        },

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    entry_points={
        'console_scripts': [
            'fsource=fsource.__main__:main',
            ],
        },
    zip_safe=True,      # reconsider when adding data files
    )
