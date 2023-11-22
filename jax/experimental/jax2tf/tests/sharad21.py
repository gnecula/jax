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

@jtu.ignore_warning(category=DeprecationWarning, message=".* deprecated")
def test_flax_lm1b_parameterized():
  from flax_models import transformer_lm1b as lm1b

  def _min_transformer_kwargs():
    return dict(
      vocab_size=8,
      output_vocab_size=8,
      emb_dim=4,
      num_heads=1,
      num_layers=1,
      qkv_dim=2,
      mlp_dim=2,
      max_len=2,
      dropout_rate=0.,
      attention_dropout_rate=0.)

  def _full_transformer_kwargs():
    kwargs = dict(
      decode=True,
      deterministic=True,
      logits_via_embedding=False,
      share_embeddings=False
    )
    return {**kwargs, **_min_transformer_kwargs()}

  config = lm1b.TransformerConfig(**_full_transformer_kwargs())
  model = lm1b.TransformerLM(config=config)
  x = np.zeros((2, 1), np.float32)
  rng1, rng2 = jax.random.split(jax.random.key(0))
  variables = model.init(rng1, x)

  def apply(*args):
    # Don't return the new state (containing the cache).
    output, _ = model.apply(*args, rngs={'cache': rng2}, mutable=['cache'])
    return output

  prediction = apply(variables, x)

  # Now create a model with symbolic configuration
  v, l = export.parse_shape("V, L")
  model_symbolic_config = lm1b.TransformerConfig(
    **dict(_full_transformer_kwargs(),
           vocab_size=v, output_vocab_size=v,
           max_len=l))
  model = lm1b.TransformerLM(config=model_symbolic_config)
  x_shape = jax.ShapeDtypeStruct(
    # TODO: improve error messages if we don't use multiple of 4 for height and width
    export.parse_shape("B, 1"),
    x.dtype)
  variable_shapes = jax.eval_shape(model.init, rng1, x_shape)
  assert jax.tree_map(lambda v: str(v.shape), variable_shapes) == {
    'cache': {
      'decoder': {
        'encoderdecoderblock_0': {
          'SelfAttention_0': {
            'cache_index': '()',
            'cached_key': '(B, 1, 1, 2)',
            'cached_value': '(B, 1, 1, 2)'}},
        'posembed_output': {
          'cache_index': '()'}}},
    'params': {
      'decoder': {
        'Embed_0': {'embedding': '(V, 4)'},
        'encoderdecoder_norm': {'bias': '(4,)', 'scale': '(4,)'},
        'encoderdecoderblock_0': {
          'LayerNorm_0': {'bias': '(4,)', 'scale': '(4,)'},
          'LayerNorm_1': {'bias': '(4,)', 'scale': '(4,)'},
          'MlpBlock_0': {
            'Dense_0': {'bias': '(2,)', 'kernel': '(4, 2)'},
            'Dense_1': {'bias': '(4,)', 'kernel': '(2, 4)'}},
          'SelfAttention_0': {
            'key': {
              'kernel': '(4, 1, 2)'},
            'out': {
              'kernel': '(1, 2, 4)'},
            'query': {'kernel': '(4, 1, 2)'},
            'value': {'kernel': '(4, 1, 2)'}}},
        'logitdense': {'bias': '(V,)', 'kernel': '(4, V)'}}}}
  prediction_shapes = jax.eval_shape(apply, variable_shapes, x_shape)
  assert str(prediction_shapes.shape) == '(B, 1, V)'
  stop = 1

# TODO [ ] handle let-bound dynamic shapes (ie output dim vars)
# TODO [ ] handle multiple outputs
# TODO [ ] make internal error message better (dont mention tracers in msg)
# TODO [ ] clean up
# TODO [ ] mapping to python variables, set trace
# TODO [ ] editor integration of some kind
# TODO [ ] handle vmap
