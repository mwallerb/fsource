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


_HEREPATH = os.path.abspath(os.path.dirname(__file__))
_VERSION_RE = re.compile(r"(?m)^__version__\s*=\s*['\"]([^'\"]*)['\"]")
_DOCLINK_RE = re.compile(r"(?m)^\s*\[\s*([^\]\n\r]+)\s*\]:\s*(doc/[./\w]+)\s*$")
_PYXFILE_RE = re.compile(r"(?i)\.pyx$")


def fullpath(path):
    """Return the full path to a file"""
    if path[0] == '/':
        raise ValueError("Do not supply absolute paths")
    return os.path.join(_HEREPATH, *path.split("/"))


def readfile(path):
    """Return contents of file with path relative to script directory"""
    return io.open(fullpath(path), 'r').read()


def extract_version(path):
    """Extract value of __version__ variable by parsing python script"""
    return _VERSION_RE.search(readfile(path)).group(1)


def rebase_links(text, base_url):
    """Rebase links to doc/ directory to ensure they work online."""
    result, nsub = _DOCLINK_RE.subn(r"[\1]: %s/\2" % base_url, text)
    return result


VERSION = extract_version('src/fsource/__init__.py')
REPO_URL = "https://github.com/mwallerb/fsource"
DOCTREE_URL = "%s/tree/v%s" % (REPO_URL, VERSION)
LONG_DESCRIPTION = rebase_links(readfile('README.md'), DOCTREE_URL)

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
        'License :: OSI Approved '
                ':: GNU Lesser General Public License v3 (LGPLv3)',
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
            'fsource=fsource._cli:main',
            ],
        },
    zip_safe=True,      # reconsider when adding data files
    )
