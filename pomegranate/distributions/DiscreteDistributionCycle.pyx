#!python
#cython: boundscheck=False
#cython: cdivision=True
# DiscreteDistributionCycle.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import itertools as it
import numpy
import random

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state
from ..utils import _check_nan

from libc.math cimport sqrt as csqrt

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
eps = numpy.finfo(numpy.float64).eps

cdef convert_to_python(double *ptr_counts, original_dict):
	cdef int i
	d = {}
	for i, key in enumerate(original_dict.keys()):
		d[key] = ptr_counts[i]
	return d

cdef class DiscreteDistributionCycle(DiscreteDistribution):
	"""
	A discrete distribution, made up of characters and their probabilities,
	assuming that these probabilities will sum to 1.0.
	"""

	property parameters:
		def __get__(self):
			return [self.dist]
		def __set__(self, parameters):
			d = parameters[0]
			self.dist = d
			self.log_dist = {key: _log(value) for key, value in d.items()}

	property max_components:
		def __get__(self):
			return self.max_components
		def __set__(self, max_components):
			self.max_components = max_components

	property default_pseudocount:
		def __get__(self):
			return self.default_pseudocount
		def __set__(self, default_pseudocount):
			self.default_pseudocount = default_pseudocount


	def __cinit__(self, dict characters = {},bint frozen=False):
		"""
		Make a new discrete distribution with a dictionary of discrete
		characters and their probabilities, checking to see that these
		sum to 1.0. Each discrete character can be modelled as a
		Bernoulli distribution.
		"""

		self.name = "DiscreteDistributionCycle"
		self.frozen = frozen
		self.max_components = 3 # default value since we can not pass to constructor (cinit)
		self.default_pseudocount = 50 # default value since we can not pass to constructor (cinit)

		self.is_blank_= True
		self.dtype = None
		if len(characters) > 0:
			self.is_blank_ = False
			self.dtype = self._get_dtype(characters)

		self.dist = characters.copy()
		self.log_dist = { key: _log(value) for key, value in characters.items() }
		self.summaries =[{ key: 0 for key in characters.keys() }, 0]

	def _get_dtype(self, characters: dict) -> str:
		"""
		Determine dtype from characters.
		"""
		return str(type(list(characters.keys())[0])).split()[-1].strip('>').strip("'")

	def __reduce__(self):
		"""Serialize the distribution for pickle."""
		return self.__class__, (self.dist, self.frozen)

	def __len__(self):
		return len(self.dist)

	def __mul__(self, other):
		"""Multiply this by another distribution sharing the same keys."""

		self_keys = self.keys()
		other_keys = other.keys()
		distribution, total = {}, 0.0

		if isinstance(other, DiscreteDistributionCycle) and self_keys == other_keys:
			self_values = (<DiscreteDistributionCycle>self).dist.values()
			other_values = (<DiscreteDistributionCycle>other).dist.values()
			for key, x, y in zip(self_keys, self_values, other_values):
				if _check_nan(key):
					distribution[key] = (1 + eps) * (1 + eps)
				else:
					distribution[key] = (x + eps) * (y + eps)
				total += distribution[key]
		else:
			assert set(self_keys) == set(other_keys)
			self_items = (<DiscreteDistributionCycle>self).dist.items()
			for key, x in self_items:
				if _check_nan(key):
					x = 1.
				y = other.probability(key)
				distribution[key] = (x + eps) * (y + eps)
				total += distribution[key]

		for key in self_keys:
			distribution[key] /= total

			if distribution[key] <= eps / total:
				distribution[key] = 0.0
			elif distribution[key] >= 1 - eps / total:
				distribution[key] = 1.0

		return DiscreteDistributionCycle(distribution)


	def equals(self, other):
		"""Return if the keys and values are equal"""

		if not isinstance(other, DiscreteDistributionCycle):
			return False

		self_keys = self.keys()
		other_keys = other.keys()

		if self_keys == other_keys:
			self_values = (<DiscreteDistributionCycle>self).log_dist.values()
			other_values = (<DiscreteDistributionCycle>other).log_dist.values()
			for key, self_prob, other_prob in zip(self_keys, self_values, other_values):
				if _check_nan(key):
					continue
				self_prob = round(self_prob, 12)
				other_prob = round(other_prob, 12)
				if self_prob != other_prob:
					return False
		elif set(self_keys) == set(other_keys):
			self_items = (<DiscreteDistributionCycle>self).log_dist.items()
			for key, self_prob in self_items:
				if _check_nan(key):
					self_prob = 0.
				else:
					self_prob = round(self_prob, 12)
				other_prob = round(other.log_probability(key), 12)
				if self_prob != other_prob:
					return False
		else:
			return False

		return True

	def clamp(self, key):
		"""Return a distribution clamped to a particular value."""
		return DiscreteDistributionCycle({ k : 0. if k != key else 1. for k in self.keys() })

	def keys(self):
		"""Return the keys of the underlying dictionary."""
		return tuple(self.dist.keys())

	def items(self):
		"""Return items of the underlying dictionary."""
		return tuple(self.dist.items())

	def values(self):
		"""Return values of the underlying dictionary."""
		return tuple(self.dist.values())

	def mle(self):
		"""Return the maximally likely key."""

		max_key, max_value = None, 0
		for key, value in self.items():
			if value > max_value:
				max_key, max_value = key, value

		return max_key

	def probability(self, X):
		"""Return the prob of the X under this distribution."""

		return self.__probability(X)

	cdef double __probability(self, X):
		if _check_nan(X):
			return 1.
		else:
			return self.dist.get(X, 0)

	def log_probability(self, X):
		"""Return the log prob of the X under this distribution."""

		return self.__log_probability(X)

	cdef double __log_probability(self, X):
		if _check_nan(X):
			return 0.
		else:
			return self.log_dist.get(X, NEGINF)

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			elif X[i] < 0 or X[i] > self.n:
				log_probability[i] = NEGINF
			else:
				log_probability[i] = self.encoded_log_probability[<int> X[i]]

	def sample(self, n=None, random_state=None):
		random_state = check_random_state(random_state)

		keys = list(self.dist.keys())
		probabilities = list(self.dist.values())

		if n is None:
			return random_state.choice(keys, p=probabilities)
		else:
			return random_state.choice(keys, p=probabilities, size=n)

	def fit(self, items, weights=None, inertia=0.0, pseudocount=0.0):
		"""
		Set the parameters of this Distribution to maximize the likelihood of
		the given sample. Items holds some sort of sequence. If weights is
		specified, it holds a sequence of value to weight each item by.
		"""

		if self.frozen:
			return self

		self.summarize(items, weights)
		self.from_summaries(inertia, pseudocount)
		return self

	def summarize(self, items, weights=None):
		"""Reduce a set of observations to sufficient statistics."""
		if weights is None:
			weights = numpy.ones(len(items))
		else:
			weights = numpy.asarray(weights)
		items = numpy.asarray(items).flatten()
			
		for i in range(len(items)):
			x = items[i]
			if _check_nan(x) == False:
				try:
					self.summaries[0][x] += weights[i]
				except KeyError:
					self.summaries[0][x] = weights[i]
				self.summaries[1] += weights[i]

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i
		cdef double item
		self.encoded_summary = 1

		encoded_counts = <double*> calloc(self.n, sizeof(double))

		for i in range(n):
			item = items[i*d + column_idx]
			if isnan(item):
				continue

			encoded_counts[<int> item] += weights[i]

		with gil:
			for i in range(self.n):
				self.encoded_counts[i] += encoded_counts[i]
				self.summaries[1] += encoded_counts[i]
				#self.summaries[0] =  convert_to_python(self.encoded_counts, original_dict=self.summaries[0])
				# print("summary")
				# print(convert_to_python(self.summaries[1], 20))

		free(encoded_counts)




	def from_summaries(self, inertia=0.0, pseudocount=0.0):
		"""Use the summaries in order to update the distribution."""

		pseudocount = self.default_pseudocount

		if self.summaries[1] == 0 or self.frozen == True:
			#print("Cycle State will not be updated")
			return
		#print("Elements used for cycle state update ", self.summaries[1])

		if self.encoded_summary == 0:
			# print("summaries before ", self.summaries[0])
			sorted_dictionary = dict(sorted(self.summaries[0].items(), key=lambda item: item[1]))
			# print("summaries sorted ", sorted_dictionary)
			total_elements = len(sorted_dictionary)
			for key in list(sorted_dictionary.keys())[:-self.max_components]:
				self.summaries[0][key] = 0

			values = self.summaries[0].values()
			# print("summaries filtered", self.summaries[0].values())


			_sum = sum(values) + pseudocount * len(values)
			if sum(values) < pseudocount * len(values):
				print("Cycle State will not be updated actually, only forget")
			for key, value in self.summaries[0].items():
				value += pseudocount
				try:
					self.dist[key] = self.dist[key]*inertia + (1-inertia)*(value / _sum)
				except KeyError:
					self.dist[key] = value / _sum
				self.log_dist[key] = _log(self.dist[key])

			self.bake(self.encoded_keys)
		else:
			n = len(self.encoded_keys)
			# print("Encoded keys before ", self.encoded_keys)
			# print("and counts ", convert_to_python(self.encoded_counts, original_dict=self.summaries[0]))
			sorted_keys = list(sorted(self.encoded_keys, key=lambda item: self.encoded_counts[self.encoded_keys.index(item)]))
			# print("Sorted keys ", sorted_keys)

			for i in range(0,n-self.max_components):
				self.encoded_counts[self.encoded_keys.index(sorted_keys[i])] = 0

			# print("Filtered counts ", convert_to_python(self.encoded_counts, original_dict=self.summaries[0]))
			# now sum changed, need to adjust it
			self.summaries[1] = 0
			for i in range(n):
				self.summaries[1] += self.encoded_counts[i]
			for i in range(n):
				_sum = self.summaries[1] + pseudocount * n
				value = self.encoded_counts[i] + pseudocount

				key = self.encoded_keys[i]
				self.dist[key] = self.dist[key]*inertia + (1-inertia)*(value / _sum)
				self.log_dist[key] = _log(self.dist[key])

			self.bake(self.encoded_keys)

		if self.is_blank_:
			self.dtype = self._get_dtype(self.dist)
			self.is_blank_ = False

		self.clear_summaries()

	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [{ key: 0 for key in self.keys() }, 0]
		if self.encoded_summary == 1:
			for i in range(len(self.encoded_keys)):
				self.encoded_counts[i] = 0

	def to_dict(self):
		return {
			'class' : 'Distribution',
			'dtype' : self.dtype,
			'name'  : self.name,
			'parameters' : [{str(key): value for key, value in self.dist.items()}],
			'frozen' : self.frozen,
			'max_components': self.max_components,
			'default_pseudocount': self.default_pseudocount
		}

	@classmethod
	def from_samples(cls, items, weights=None, pseudocount=0, keys=None):
		"""Fit a distribution to some data without pre-specifying it."""
		key_initials = {}
		if keys is not None:
			clean_keys = tuple(key for key in keys if not _check_nan(key))
			# A priori equal probability.
			key_initials = {key: 1./len(clean_keys) for key in clean_keys}
		return cls(key_initials).fit(items, weights=weights, pseudocount=pseudocount)

	@classmethod
	def blank(cls):
		return cls()
