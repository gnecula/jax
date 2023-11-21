"""

Run tests here with

"""
from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
import inspect

import typing
from typing import Union

import numpy as np

from jax._src import core
from jax import lax
from jax._src.util import safe_zip, safe_map
from jax._src import test_util as jtu
import jax
import jax.numpy as jnp

from jax.experimental.export import export

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

if typing.TYPE_CHECKING:
  f32 = typing.Annotated
else:
  class dtype:
    def __init__(self, dtype):
      self.dtype = dtype

    def __getitem__(self, dims: tuple[Union[int, str]]) -> jax.ShapeDtypeStruct:
      if type(dims) is not tuple:
        dims = (dims,)
      return jax.ShapeDtypeStruct(
          export.symbolic_shape(",".join(str(d) for d in dims)),
          self.dtype)

  f32 = dtype(jnp.dtype('float32'))


def shapecheck(f):
  try: sig = inspect.signature(f)
  except: return

  dummy_args: Sequence[jax.ShapeDtypeStruct] = [
    eval(param.annotation) for param in sig.parameters.values()]

  jaxpr = jax.make_jaxpr(f)(*dummy_args)
  computed_shape_dtype, = jaxpr.out_avals

  expected_shape_dtype: jax.ShapeDtypeStruct = eval(sig.return_annotation)
  if computed_shape_dtype.shape != expected_shape_dtype.shape:
    raise TypeError(f"Expected {expected_shape_dtype.shape}, found {computed_shape_dtype.shape}")

  return f



###

def test_simple():
  @shapecheck
  def f(x: f32[3, "n"], y:f32[3, "n"]) -> f32[3]:
    z = jnp.dot(x, y.T)
    w = jnp.tanh(z)
    return w.sum(0)

def test_batched():
  @shapecheck
  def f(x: f32["b", "n"], y:f32["b", "n"]) -> f32["b"]:
    z = jnp.dot(x, y.T)
    w = jnp.tanh(z)
    return w.sum(0)

def test_reshape():
  @shapecheck
  def f(x: f32["b", "n"], y:f32["b", "n"]) -> f32[2, "b*n"]:
    z = jnp.concatenate([x, y], axis=1)
    w = jnp.reshape(z, (2, -1))
    return w

def test_vmap():
  @shapecheck
  def f(x: f32["n"], y:f32["n"]) -> f32[2]:
    z = jnp.concatenate([x, y], axis=0)
    return z.reshape((2, -1)).sum(axis=1)

  @shapecheck
  def vf(x: f32["b", "n"], y:f32["b", "n"]) -> f32["b", 2]:
    return jax.vmap(f)(x, y)


def test_vmap_better():
  # TODO: change jax.vmap to add new axes to the shapecheck specification
  @shapecheck
  @jax.vmap
  def f(x: f32["n"], y:f32["n"]) -> f32[2]:
    z = jnp.concatenate([x, y], axis=0)
    return z.reshape((2, -1)).sum(axis=1)

def test_multiple_outputs():
  # TODO: handle multiple outputs
  @shapecheck
  def f(x: f32["b", "n"]) -> tuple[f32["b"], f32["n"]]:
    return (jnp.sum(x, axis=0),
            jnp.sum(x, axis=1))

@jtu.ignore_warning(category=DeprecationWarning, message=".* is deprecated")
def test_flax_cnn():
  from flax import linen as nn
  class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      return x

  model = CNN()
  x = np.zeros((3, 28, 28, 1), np.float32)
  variables = model.init(jax.random.key(0), x)
  prediction = model.apply(variables, x)

  x_shape = jax.ShapeDtypeStruct(
    # TODO: improve error messages if we don't use multiple of 4 for height and width
    export.symbolic_shape("b, 4*h, 4*w, c"),
    x.dtype)
  variables_shapes = jax.eval_shape(model.init,
                                    jax.random.key(0),
                                    x_shape)
  assert jax.tree_map(lambda v: str(v.shape), variables_shapes) == {
    'params': {
      'Conv_0': {'bias': '(32,)', 'kernel': '(3, 3, c, 32)'},
      'Conv_1': {'bias': '(64,)', 'kernel': '(3, 3, 32, 64)'},
      'Dense_0': {'bias': '(256,)', 'kernel': '(64*h*w, 256)'},
      'Dense_1': {'bias': '(10,)', 'kernel': '(256, 10)'}
    }
  }

  prediction_shape = jax.eval_shape(model.apply, variables_shapes, x_shape)
  assert str(prediction_shape.shape) == "(b, 10)"

@jtu.ignore_warning(category=DeprecationWarning, message=".* is deprecated")
def test_flax_cnn_parameterized():
  from flax import linen as nn
  from flax import struct
  @struct.dataclass
  class CNNConfig:
    features_1: int = 256  # Number of features in first dense layer
    features_2: int = 10   # Number of features in second dense layer

  class CNN(nn.Module):
    """A simple CNN model."""

    config: CNNConfig

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=self.config.features_1)(x)
      x = nn.relu(x)
      x = nn.Dense(features=self.config.features_2)(x)
      return x

  model_config = CNNConfig()
  model = CNN(config=model_config)
  x = np.zeros((3, 28, 28, 1), np.float32)  # : f32[b, h, w, c]
  variables = model.init(jax.random.key(0), x)
  prediction = model.apply(variables, x)

  # Now create a model with symbolic configuration
  f1, f2 = export.symbolic_shape("f1, f2")
  model_symbolic_config = CNNConfig(features_1=f1, features_2=f2)
  model = CNN(config=model_symbolic_config)
  x_shape = jax.ShapeDtypeStruct(
    # TODO: improve error messages if we don't use multiple of 4 for height and width
    export.symbolic_shape("b, 4*h, 4*w, c"),
    x.dtype)
  variables_shapes = jax.eval_shape(model.init,
                                    jax.random.key(0),
                                    x_shape)
  assert jax.tree_map(lambda v: str(v.shape), variables_shapes) == {
    'params': {
      'Conv_0': {'bias': '(32,)', 'kernel': '(3, 3, c, 32)'},
      'Conv_1': {'bias': '(64,)', 'kernel': '(3, 3, 32, 64)'},
      'Dense_0': {'bias': '(f1,)', 'kernel': '(64*h*w, f1)'},
      'Dense_1': {'bias': '(f2,)', 'kernel': '(f1, f2)'}
    }
  }

  prediction_shape = jax.eval_shape(model.apply, variables_shapes, x_shape)
  assert str(prediction_shape.shape) == "(b, f2)"


# TODO [ ] handle let-bound dynamic shapes (ie output dim vars)
# TODO [ ] handle multiple outputs
# TODO [ ] make internal error message better (dont mention tracers in msg)
# TODO [ ] clean up
# TODO [ ] mapping to python variables, set trace
# TODO [ ] editor integration of some kind
# TODO [ ] handle vmap
