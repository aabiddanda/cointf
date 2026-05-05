"""Python package to annotate somatic sequencing VCF files to estimate the probability of germline polymorphism.

pGermlinePoly implements an EM-algorithm incoporating site-level annotations .

Modules exported are:
"""

__version__ = "0.0.1a"

from .stahl import Stahl  # noqa
