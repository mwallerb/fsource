import inspect
import os.path
import sys
import warnings

import pytest

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        import pylint.lint as pylint_lint
except ImportError:
    warnings.warn("Pylint is not available - unable to do linter pass",
                  ImportWarning)
    pylint_lint = None


def pylint_no_exit():
    # pylint.lint.Run parameters exit and do_exit have a turbulent history:
    # until 2.0, you had to use exit, then exit was replaced by do_exit,
    # which was again replaced by exit in 2.5.1 (nice job of semantic
    # versioning, guys!), with do_exit then being restored and immediately
    # deprecated in favor of exit in 2.5.3.
    if sys.version_info >= (3,):
        argspec = inspect.getfullargspec(pylint_lint.Run.__init__)
    else:
        argspec = inspect.getargspec(pylint_lint.Run.__init__)
    if 'exit' in argspec.args:
        return {'exit': False}
    elif 'do_exit' in argspec.args:
        return {'do_exit': False}
    else:
        raise RuntimeError("pylint.lint.Run accepts neither exit nor do_exit")


@pytest.mark.skipif(pylint_lint is None, reason="Pylint not available")
def test_linting_errors():
    herepath = os.path.dirname(os.path.realpath(__file__))
    srcdir = os.path.join(herepath, os.pardir, "src", "fsource")
    print("running `pylint -E {}`".format(srcdir))
    run = pylint_lint.Run(['-E', srcdir], **pylint_no_exit())
    assert run.linter.msg_status == 0
