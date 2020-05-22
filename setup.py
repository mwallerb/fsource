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

def readfile(*parts):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()

def extract_version(*parts):
    """Extract value of __version__ variable by parsing python script"""
    initfile = readfile(*parts)
    version_re = re.compile(r"(?m)^__version__\s*=\s*['\"]([^'\"]*)['\"]")
    match = version_re.search(initfile)
    return match.group(1)

def rebase_links(text, base_url):
    """Rebase links to doc/ directory to ensure they work online."""
    doclink_re = re.compile(
                        r"(?m)^\s*\[\s*([^\]\n\r]+)\s*\]:\s*(doc/[./\w]+)\s*$")
    result, nsub = doclink_re.subn(r"[\1]: %s/\2" % base_url, text)
    return result

VERSION = extract_version('src', 'fsource', '__init__.py')
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
