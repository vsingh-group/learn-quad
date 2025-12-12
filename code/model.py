import equinox as eqx
import jax

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import pdb
import math

from jax import random
from jax import config
# config.update("jax_enable_x64", True)

import numpy as np







def tanh_layer(params, x):
	value = jnp.tanh(jnp.dot(params[0], x) + params[1])
	return value

def relu_layer(params, x):
	value = jax.nn.relu(jnp.dot(params[0], x) + params[1])
	return value

def solution_net_forward_pass(in_array, params):
	activations = in_array
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations[0]

def solution_net_forward_pass_1D(in_array, params):
	activations = jnp.array([in_array])
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations[0]

def solution_net_forward_pass_v2(in_x, in_y, params):
	activations = jnp.array([in_x, in_y])
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations[0]


def solution_net_forward_pass_v2_mod(in_x, in_y, params):
	L=2
	M=10
	w = 2.0 * jnp.pi / L
	k = jnp.arange(1, M + 1)
	out = jnp.concatenate([jnp.array([in_y, jnp.ones_like(in_y)]), jnp.cos(k * w * in_x), jnp.sin(k * w * in_y)], axis=0)
	activations = out
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations[0]


def solution_net_forward_pass_v3(inp, params):
	activations = inp
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations[0]


def modulation_net_forward_pass(input, params):
	# return jnp.array([1.0])
	activations = jnp.array([input])
	for w, b in params[:-1]:
		# print(activations.shape)
		# print(w.shape)
		# print(b.shape)
		# pdb.set_trace()
		activations = tanh_layer([w, b], activations)
		# activations = relu_layer([w, b], activations)
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	activations = jax.nn.softplus(activations)
	return activations

def generic_tanh_forward_pass(input, params):
	activations = jnp.array([input])
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations


def family_forward_pass(input, params):
	activations = jnp.array(input)
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations

def pdesolution_forward_pass(input, params):
	activations = jnp.array(input)
	for w, b in params[:-1]:
		activations = tanh_layer([w, b], activations)	
	activations = jnp.dot(params[-1][0], activations) + params[-1][1]
	return activations

class SolutionNet(eqx.Module):
	layers: list

	def __init__(self, input_dim, output_dim, depth, width, key):
		keys = jax.random.split(key, depth+2)
		self.layers = [eqx.nn.Linear(input_dim, width, key=keys[0])]
		self.layers.append(jax.nn.tanh)
		for i in range(depth):
			layer = eqx.nn.Linear(width, width, key=keys[i+1])
			self.layers.append(layer)
			self.layers.append(jax.nn.tanh)
		final_layer = eqx.nn.Linear(width, output_dim, key=keys[-1])
		self.layers.append(final_layer)

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		# return scalar value for gradient computation
		return x[0]
	
class ModulationNet(eqx.Module):
	layers: list

	def __init__(self, input_dim, depth, width, key):
		keys = jax.random.split(key, depth+2)
		self.layers = [eqx.nn.Linear(input_dim, width, key=keys[0])]
		# self.layers.append(jax.nn.tanh)
		for i in range(depth):
			layer = eqx.nn.Linear(width, width, key=keys[i+1])
			self.layers.append(layer)
			# self.layers.append(jax.nn.tanh)
		final_layer = eqx.nn.Linear(width, 1, key=keys[-1])
		self.layers.append(final_layer)
		# self.layers.append(jax.nn.softplus)            # to constrain the output to be always positive

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
			x = jax.nn.tanh(x)
		x = jax.nn.softplus(x)
		return x
		# return x[0]
	
class CoeffNet(eqx.Module):
	layers: list

	def __init__(self, input_dim, depth, width, key):
		keys = jax.random.split(key, depth+2)
		self.layers = [eqx.nn.Linear(input_dim, width, key=keys[0])]
		self.layers.append(jax.nn.tanh)
		for i in range(depth):
			layer = eqx.nn.Linear(width, width, key=keys[i+1])
			self.layers.append(layer)
			self.layers.append(jax.nn.tanh)
		final_layer = eqx.nn.Linear(width, 4, key=keys[-1])
		self.layers.append(final_layer)

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	
class EdgeNet(eqx.Module):
	layers: list

	def __init__(self, input_dim, depth, width, key):
		keys = jax.random.split(key, depth+2)
		self.layers = [eqx.nn.Linear(input_dim, width, key=keys[0])]
		self.layers.append(jax.nn.tanh)
		for i in range(depth):
			layer = eqx.nn.Linear(width, width, key=keys[i+1])
			self.layers.append(layer)
			self.layers.append(jax.nn.tanh)
		final_layer = eqx.nn.Linear(width, 6, key=keys[-1])
		self.layers.append(final_layer)

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		return x