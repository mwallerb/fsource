import os.path
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


@pytest.mark.skipif(pylint_lint is None, reason="Pylint not available")
def test_linting_errors():
    herepath = os.path.dirname(os.path.realpath(__file__))
    srcdir = os.path.join(herepath, os.pardir, "src", "fsource")
    print("running `pylint -E {}`".format(srcdir))
    run = pylint_lint.Run(['-E', srcdir], exit=False)
    assert run.linter.msg_status == 0
