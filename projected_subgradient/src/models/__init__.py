# This file initializes the models module.

from .bernoulli_laplace import bernoulli_laplace_model, generate_states, get_stationary_distribution as bl_get_stationary
from .curie_weiss import curie_weiss_model, get_stationary_distribution as cw_get_stationary