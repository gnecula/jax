from __future__ import annotations

from collections.abc import Sequence
import inspect
import types
import typing
from typing import Union

import numpy as np

from jax._src import core
from jax._src.util import safe_zip, safe_map
import jax
import jax.numpy as jnp

from jax.experimental.jax2tf import jax_export

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

jax.config.update('jax_dynamic_shapes', True)


if typing.TYPE_CHECKING:
  f32 = typing.Annotated
else:
  class dtype:
    def __init__(self, dtype):
      self.dtype = dtype

    def __getitem__(self, dims: tuple[Union[int, str]]) -> jax.ShapeDtypeStruct:
      if type(dims) is not tuple:
        dims = (dims,)
      return jax_export.poly_spec(
        (None,) * len(dims),  # not needed
        self.dtype,
        ",".join(str(d) for d in dims))

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


# TODO [ ] handle let-bound dynamic shapes (ie output dim vars)
# TODO [ ] handle multiple outputs
# TODO [ ] make internal error message better (dont mention tracers in msg)
# TODO [ ] clean up
# TODO [ ] mapping to python variables, set trace
# TODO [ ] editor integration of some kind
# TODO [ ] handle vmap
