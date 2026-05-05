"""Setup module for building pGermlinePoly utilities."""

from distutils.core import Extension

from Cython.Build import cythonize
from setuptools import setup

extensions = [
    Extension("stahl_utils", ["cointf/stahl_utils.pyx"]),
    Extension("stahl_saddlepoint", ["cointf/stahl_saddlepoint.pyx"]),
]

setup_args = dict(
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": 3, "profile": False}
    )
)
setup(**setup_args)
