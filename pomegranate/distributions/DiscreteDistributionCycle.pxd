# DiscreteDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution
from .DiscreteDistribution cimport DiscreteDistribution

cdef class DiscreteDistributionCycle(DiscreteDistribution):
	cdef int max_components
	cdef double default_pseudocount

