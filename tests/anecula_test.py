# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from functools import partial
import itertools
from unittest import skip, SkipTest
import tensorflow as tf
import os
import shutil
from typing import List, Tuple, Callable, Any, TypeVar
from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr
import six

from jax import api
from jax import core
from jax import lax
from jax import ops
from jax import numpy as jnp
from jax import test_util as jtu
from jax import lax_reference
from jax.test_util import check_grads
from jax.interpreters import xla
from jax.lib import xla_bridge
from jax.lib import xla_client

from jax.config import config

config.parse_flags_with_absl()
FLAGS = config.FLAGS

Carry = TypeVar('Carry')

def set_env(test_name=''):
  output_dir = '{}/tmp/jax/dump_hlo_{}'.format(os.environ['HOME'], test_name)
  print("Dumping HLO to {}".format(output_dir))
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
  #os.environ['XLA_FLAGS'] = '--xla_dump_to={} --xla_hlo_profile'.format(output_dir)
  os.environ['JAX_LOG_COMPILES'] = 'True'
  #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def show_jaxpr_and_result(what, f, *args):
  jaxpr, consts = api.make_jaxpr(f)(*args)
  print("\n{}_jaxpr={}{}_consts=[{}]".format(what, jaxpr, what, consts))
  res = f(*args)
  print("{}_res={}".format(what, res))
  return res

def time_loop(n, f, *args):
  f(*args) # Warm the cache
  for i in range(n):
    f(*args)

def my_scan(n, init: Carry, f: Callable[[Carry, int], Carry]) -> Carry:
  """A version of scan that iterates over indices.

  Params:
    n: the number of iterations
    init: the initial carry
    f: a carry function, of type 'carry -> int_index -> carry'
  Return:
    the final carry
  """
  outc, _ = lax.scan(lambda c, i: (f(c, i), None), init, jnp.arange(n))
  return outc


def my_fori(n: int, init: Carry, f: Callable[[Carry, int], Carry]) -> Carry:
  """A version of scan that iterates over indices.

  Params:
    n: the number of iterations
    init: the initial carry
    f: a carry function, of type 'carry -> int_index -> carry'
  Return:
    the final carry
  """
  return lax.fori_loop(0, n, lambda i, c: f(c, i), init)

def opt_index(arr, index_list: List[int]):
  """Reads a scalar from a tensor.
  Params:
    arr: a n-dimensional tensor
    index_tuple: an n-tuple of integers, guaranteed to be within bounds
  Returns:
    an array of shape (1)
  """
  assert len(arr.shape) == len(index_list)
  return lax.dynamic_slice_p.bind(arr, *index_list, slice_sizes=(1,), operand_shape=arr.shape)

def opt_update_index(arr, index_list: List[int], what):
  """Updated an element of a tensor.
  Params:
    arr: an n-dimensional array
    index_list: a list of integer"""
  assert what.shape == (1,)
  assert len(arr.shape) == len(index_list)
  return lax.dynamic_update_slice_p.bind(arr, what, *index_list, update_shape=what.shape)

def matmul_xla_scan(x, y):
  """A matmul implemented using scan,

  using JAX.lax front-end.
  """
  m, p = x.shape
  p1, n = y.shape
  assert p == p1

  acc = jnp.zeros((m, n), dtype=onp.float32)

  m_range = jnp.arange(m)
  n_range = jnp.arange(n)
  p_range = jnp.arange(p)

  return lax.scan(lambda acc, i: (
    lax.scan(lambda acc, j: (
      lax.scan(lambda acc, k: (ops.index_add(acc, (i, j), x[i, k] * y[k, j]),
                               None),
               acc, p_range)),
             acc, n_range)),
                  acc, m_range)[0]


def matmul_xla_scan_1(x, y):
  """A matmul implemented using scan,

  using JAX.lax front-end.
  """
  m, p = x.shape
  p1, n = y.shape
  assert p == p1

  acc = jnp.zeros((m, n), dtype=onp.float32)

  n_range = jnp.arange(n)
  p_range = jnp.arange(p)

  return lax.scan(lambda acc, j: (
    lax.scan(lambda acc, k: (ops.index_add(acc, (0, j), x[0, k] * y[k, j]),
                             None),
             acc, p_range)),
                  acc, n_range)


class MyTests(jtu.JaxTestCase):
  def setUp(self) -> None:
    set_env(test_name=self._testMethodName)
    super(MyTests, self).setUp()

  def test_grad_fori(self):
    def loop(a):
      def for_body(i, acc):
        return acc + a[i]

      return lax.fori_loop(0, a.shape[0], for_body, 0.)

    a = jnp.ones(5)
    print("\njaxpr=\n", api.make_jaxpr(loop)(a))
    res1 = loop(a)
    res = api.grad(loop)(a)  # Not implemented
    print("grad={}".format(res))

  def test_grad_scan(self):
    def loop(a):
      def scan_body(acc, i):
        return (acc + a[i], None)

      res, _ = lax.scan(scan_body, 0., jnp.arange(a.shape[0]))
      return res

    a = jnp.ones(5)
    print("jaxpr=\n", api.make_jaxpr(loop)(a))
    res = api.grad(loop)(a)
    print("grad={}".format(res))

  def test_jaxprs(self):
    """These are jaxprs that I saw in the compilation of indexing"""

    """
      i (the index), e.g., 2
      b is the upper limit
      e is the array [1., 1., ... ]
      f is the upper limit?, 
      j = broadcast_in_dim[ shape=(1,)
                            broadcast_dimensions=() ] i
      k = concatenate[ dimension=0
                       operand_shapes=((0,), (1,)) ] f j
      l = gather[ dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
                  slice_sizes=(1,)
                  operand_shape=(5,) ] e k
      m = reshape[ new_sizes=()
                   dimensions=None
                   old_sizes=() ] l
     """

    i = 1  # The index at which we want to index
    f = jnp.array([], dtype=onp.int32)  # The upper limit?
    e = jnp.ones(5)  # The array into which we index

    def f_broadcast(i):
      # Wraps an integer as an array
      return lax.broadcast_in_dim(i, (1,), ())

    j = show_jaxpr_and_result("broadcast(j)", f_broadcast, i)

    """k = concatenate[ dimension=0
                        operand_shapes=((0,), (1,)) ] f j"""

    def f_concat(x1, x2):
      return lax.concatenate((x1, x2), 0)

    k = show_jaxpr_and_result("concatenate(k)", f_concat, f, j)

    def f_gather():
      return lax.gather()

    pass

  def test_scan_sum_vector(self):
    def sum_vector(a):
      def scan_body(c, x):
        return (c * x + x, None)

      c, bs = lax.scan(scan_body, 0., a)
      return c

    a = jnp.array([1., 2., 3.])
    res = show_jaxpr_and_result("no_transform", sum_vector, a)
    # self.assertAllClose(a * a * 4, res, check_dtypes=True)
    #
    ab = jnp.array([[1., 2., 3.], [10., 20., 30.]])
    sum_vmap = api.vmap(sum_vector)
    res = show_jaxpr_and_result("vmap", sum_vmap, ab)
    # self.assertAllClose(jnp.ones(6) * 10. * 10. * 4., res, check_dtypes=True)

    sum_grad = api.grad(sum_vector)
    res = show_jaxpr_and_result("grad", sum_grad, a)
    # self.assertAllClose(2. * a * 4, res, check_dtypes=True)

  def test_scan_sum2(self):
    def sum2(a):
      def scan_body(c, idx):
        return (c * c + a, None)

      c, bs = lax.scan(scan_body, 0., jnp.arange(0, 4))
      return c

    a = 5.
    res = show_jaxpr_and_result("sum2", sum2, a)
    # self.assertAllClose(a * a * 4, res, check_dtypes=True)
    #
    ab = jnp.ones(6) * 10.
    sum2_vmap = api.vmap(sum2)
    res = show_jaxpr_and_result("sum2_vmap", sum2_vmap, ab)
    # self.assertAllClose(jnp.ones(6) * 10. * 10. * 4., res, check_dtypes=True)

    sum2_grad = api.grad(sum2)
    res = show_jaxpr_and_result("sum1_grad", sum2_grad, a)
    # self.assertAllClose(2. * a * 4, res, check_dtypes=True)

  def test_scan_vector_mul(self):
    def vector_mul(v1, v2):
      def scan_body(c, idx):
        return (ops.index_update(c, idx, v1[idx] * v2[idx]), None)

      c, bs = lax.scan(scan_body, jnp.zeros(v1.shape[0]), jnp.arange(0, v1.shape[0]))
      return c

    v1 = jnp.array([1., 2., 3.])
    v2 = jnp.array([10., 20., 30.])
    res = show_jaxpr_and_result("vector_mul", vector_mul, v1, v2)
    self.assertAllClose(jnp.array([10., 40., 90.]), res, check_dtypes=True)

    v1b = jnp.stack([v1, v1 * 2.])
    vector_mul_vmap = api.vmap(vector_mul, in_axes=(0, None))
    res = show_jaxpr_and_result("vector_mul_vmap", vector_mul_vmap, v1b, v2)

    # self.assertAllClose(jnp.ones(6) * 10. * 10. * 4., res, check_dtypes=True)

    def vector_mul_for_grad(v1, v2):
      return jnp.sum(vector_mul(v1, v2))

    vector_mul_grad = api.grad(vector_mul_for_grad)
    res = show_jaxpr_and_result("vector_mul_grad", vector_mul_grad, v1, v2)
    # self.assertAllClose(2. * a * 4, res, check_dtypes=True)

  def test_gather1(self):
    def get_one(v, idx):
      return v[idx + 1]

    v = jnp.array([1., 2., 3.])
    res = show_jaxpr_and_result("no_transform", get_one, v, 1)
    self.assertAllClose(3., res, check_dtypes=True)

    v1b = jnp.stack([v, v * 2.])
    with_vmap = api.vmap(get_one, in_axes=(0, 0))
    res = show_jaxpr_and_result("vmap", with_vmap, v1b, jnp.array([0, 1]))
    self.assertAllClose(jnp.array([2., 6.]), res, check_dtypes=True)

    with_grad = api.grad(get_one)
    res = show_jaxpr_and_result("grad", with_grad, v, 1)
    self.assertAllClose(jnp.array([0., 0., 1.]), res, check_dtypes=True)

  def test_gather_non_lin(self):
    def get_one(v, idx):
      return v[idx] ** 3

    v = jnp.array([1., 2., 3.])
    res = show_jaxpr_and_result("no_transform", get_one, v, 1)
    self.assertAllClose(8., res, check_dtypes=True)

    with_grad = api.grad(get_one)
    res = show_jaxpr_and_result("grad", with_grad, v, 1)
    self.assertAllClose(jnp.array([0., 12., 0.]), res, check_dtypes=True)

  def test_gather_2dim_non_lin(self):
    def get_one(v, idx, idy):
      return v[idx, idy] ** 3

    v = jnp.array([[1., 2., 3.], [10., 20., 30.]])
    res = show_jaxpr_and_result("no_transform", get_one, v, 1, 1)
    self.assertAllClose(8000., res, check_dtypes=True)

    with_grad = api.grad(get_one)
    res = show_jaxpr_and_result("grad", with_grad, v, 1, 1)
    self.assertAllClose(jnp.array([[0., 0., 0.], [0., 1200., 0.]]), res, check_dtypes=True)

  def test_scather1(self):
    def set_one(v, idx, val):
      return ops.index_update(v, idx + 1, val * val)

    v = jnp.array([1., 2., 3.])
    res = show_jaxpr_and_result("no_transform", set_one, v, 0, 3.)
    self.assertAllClose(jnp.array([1., 9., 3.]), res, check_dtypes=True)

    v1b = jnp.stack([v, v * 2.])
    with_vmap = api.vmap(set_one, in_axes=(0, 0, 0))
    res = show_jaxpr_and_result("vmap", with_vmap, v1b, jnp.array([0, 1]), jnp.array([3., 4.]))
    self.assertAllClose(jnp.array([[1., 9., 3.],
                                   [2., 4., 16.]]), res, check_dtypes=True)

    with_grad = api.grad(lambda v, i, val: jnp.sum(set_one(v, i, val)), argnums=0)
    res = show_jaxpr_and_result("grad", with_grad, v, 0, 3.)
    self.assertAllClose(jnp.array([1., 0., 1.]), res, check_dtypes=True)

  def test_scather2(self):
    def set_one(v, idx, val):
      return ops.index_update(v, idx, val * val)

    v = jnp.array([1., 2., 3.])
    res = show_jaxpr_and_result("no_transform", set_one, v, 0, 3.)

    with_grad = api.grad(lambda v, i, val: jnp.sum(set_one(v, i, val)), argnums=0)
    res = show_jaxpr_and_result("grad", with_grad, v, 0, 3.)
    self.assertAllClose(jnp.array([1., 0., 1.]), res, check_dtypes=True)

  def test_scather_read(self):
    def set_one(v, idx, ina):
      return ops.index_update(v, idx, ina[idx] ** 3)

    v = jnp.array([1., 2., 3.])
    ina = jnp.array([5., 6., 7., 8.])
    res = show_jaxpr_and_result("no_transform", set_one, v, 0, ina)

    with_grad = api.grad(lambda v, i, ina: jnp.sum(set_one(v, i, ina)),
                         argnums=(0, 2))
    res = show_jaxpr_and_result("grad", with_grad, v, 0, ina)
    self.assertAllClose(jnp.array([0., 1., 1.]), res, check_dtypes=True)

  def test_scan_write(self):
    def f(ina):
      def body(c, idx):
        outa, ina = c
        return ((ops.index_update(outa, idx, ina[idx] ** 3), ina),
                None)

      outa = jnp.zeros(5)
      (outa, outina), _ = lax.scan(body, (outa, ina), jnp.arange(3))
      return jnp.sum(outa)

    ina = jnp.array([5., 6., 7., 8.])

    res = api.jit(f)(ina)
    # res = show_jaxpr_and_result("no_transform", api.jit(f), ina)

    # with_grad = api.jit(api.grad(f))
    # res = show_jaxpr_and_result("grad", with_grad, ina)
    # self.assertAllClose(jnp.array([0., 1., 1.]), res, check_dtypes=True)

  def test_scan_write(self):
    def f(ina):
      def body(c, idx):
        outa, ina = c
        return ((ops.index_update(outa, idx, ina[idx] ** 3), ina),
                None)

      outa = jnp.zeros(5)
      (outa, outina), _ = lax.scan(body, (outa, ina), jnp.arange(3))
      return jnp.sum(outa)

    ina = jnp.array([5., 6., 7., 8.])

    res = api.jit(f)(ina)

  def test_matmul_scan(self):
    n = 32
    a = jnp.ones((3, 4))
    b = jnp.ones((4, 5))

    f = api.jit(matmul_xla_scan_1)
    # res = show_jaxpr_and_result("no_transform", f, a, b)
    res = f(a, b)

  def test_matmul_scan_grad(self):
    n = 32
    a = jnp.ones((n, n))
    b = jnp.ones((n, n))

    f = api.jit(api.grad(lambda a, b: jnp.sum(matmul_xla_scan(a, b))))
    # res = show_jaxpr_and_result("no_transform", f, a, b)
    res = f(a, b)

  def test_stupid(self):
    p_range = jnp.arange(5)
    a = jnp.ones((5, 5), dtype=onp.float32)
    acc = onp.zeros((3, 3), dtype=onp.float32)

    def f(x):
      return a[1, x]

    print(api.make_jaxpr(f)(1))

    def body(acc, k):
      return a[1, k], None

    res = lax.scan(body,
                   1.,
                   p_range)

  def test_one_loop(self):
    """for i in 0..2:
          out[i] = a[i] ** 2

    This allocates 8500 bytes, which includes two copies of the "out" buffer.
    - one is for the input parameter of the while
    - the other is a copy made right away, and then reused throughout scan

    With optimized HLO we get the same allocation.
    """
    opt_hlo = True
    def f(a):
      out = jnp.zeros(1000, onp.float32)  # So we recognize the allocation
      def body_fun(c, i):
        if opt_hlo:
          a_i = opt_index(a, [i])
          a_i_square = lax.pow(a_i, 2.)
          return opt_update_index(c, [i], a_i_square)
        else:
          return ops.index_update(c, i, a[i] ** 2)
      return my_scan(2, out, body_fun)

    a = lax.iota(onp.float32, 100)  # Use some large numbers so we recognize the allocations
    print(api.make_jaxpr(f)(a))
    print(api.jit(f)(a))

  def test_two_loops_sequence(self):
    """for i in 0..2:
          out[i] = a[i] ** 2
       for i in 0..3:
          out[i + 1] = a[i]

    This allocates 8500 bytes, which includes two copies of the "out" buffer.
    - one is for the input parameter of the function
    - the other is a copy made right away, and then reused throughout both scans.
    """

    def f(a):
      out = jnp.zeros(1000, onp.float32)  # So we recognize the allocation
      out1 = my_scan(2, out, lambda c, i: ops.index_update(c, i, a[i] ** 2))
      out2 = my_scan(3, out1, lambda c, i: ops.index_update(c, i + 1, a[i]))
      return out2

    a = lax.iota(onp.float32, 100)  # Use some large numbers so we recognize the allocations
    print(api.make_jaxpr(f)(a))
    print(f(a))

  def test_one_loop_read_output(self):
    """for i in 0..1000:
          out[i] = out[n - i] + a[i % 100]

    We read the output array while writing it.

    Without opt_hlo this allocates 12.5k bytes. There are 3 allocations of 4000.
    - one is the input parameter
    - one is a copy of the input parameter, made at the start of the module
    - there seems to be a copy in the loop body!!!

    With opt_hlo allocates only 8k bytes.
    """
    opt_hlo = True
    def f(a):
      out = jnp.zeros(1000, onp.float32)  # So we recognize the allocation
      def body_fun(c, i):
        if opt_hlo:
          return opt_update_index(c, [i], opt_index(c, [999 - i]) + opt_index(a, [i % 100]))
        else:
          return ops.index_update(c, i, c[999 - i] + a[i % 100])
      return my_fori(1000, out, body_fun)

    a = lax.iota(onp.float32, 100)  # Use some large numbers so we recognize the allocations
    print(api.make_jaxpr(f)(a))
    print(api.jit(f)(a))

  def test_power_strange(self):
    """When raising an integer to a power, we get lots of xla_call computations.

    Seems to be because we use binary exponentiation for integers. And also because
    the lax.where is jitted (to avoid some constant materialization!!?)
    """

    def f(a):
      return a ** 2

    print(api.make_jaxpr(f)(4))
    print(f(4))

###
### Patches second moment
###
def get_shape(tensor):
  """Returns list of dimensions using ints only for statically known ones."""

  if tensor.shape.dims is None:
    raise ValueError("Unknown rank for tensor {}.".format(tensor))

  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.shape(tensor)
  return tuple(elt if elt is not None else dynamic_shape[idx]
               for idx, elt in enumerate(static_shape))

def num_conv_locations(input_shape, filter_shape, strides, padding):
  """Returns the number of spatial locations a conv kernel is applied to.

  Args:
    input_shape: List of ints representing shape of inputs to
      tf.nn.convolution().
    filter_shape: List of ints representing shape of filter to
      tf.nn.convolution().
    strides: List of ints representing strides along spatial dimensions as
      passed in to tf.nn.convolution().
    padding: string representing the padding method, either 'VALID' or 'SAME'.

  Returns:
    A scalar |T| denoting the number of spatial locations for the Conv layer.

  Raises:
    ValueError: If input_shape, filter_shape don't represent a 1-D or 2-D
      convolution.
  """
  if len(input_shape) != 4 and len(input_shape) != 3:
    raise ValueError("input_shape must be length 4, corresponding to a Conv2D,"
                     " or length 3, corresponding to a Conv1D.")
  if len(input_shape) != len(filter_shape):
    raise ValueError("Inconsistent number of dimensions between input and "
                     "filter for convolution")

  if strides is None:
    if len(input_shape) == 4:
      strides = [1, 1, 1, 1]
    else:
      strides = [1, 1, 1]

  # Use negative integer division to implement 'rounding up'.
  # Formula for convolution shape taken from:
  # http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
  if len(input_shape) == 3:
    if padding is not None and padding.lower() == "valid":
      out_width = -(-(input_shape[1] - filter_shape[0] + 1) // strides[1])
    else:
      out_width = -(-input_shape[1] // strides[1])

    return out_width
  else:
    if padding is not None and padding.lower() == "valid":
      out_height = -(-(input_shape[1] - filter_shape[0] + 1) // strides[1])
      out_width = -(-(input_shape[2] - filter_shape[1] + 1) // strides[2])
    else:
      out_height = -(-input_shape[1] // strides[1])
      out_width = -(-input_shape[2] // strides[2])

    return out_height * out_width

def append_homog(tensor, homog_value=None):
  """Appends a homogeneous coordinate to the last dimension of a Tensor.

  Args:
    tensor: A Tensor.
    homog_value: Value to append as homogeneous coordinate to the last dimension
      of `tensor`.  If None 1.0 is used. (Default: None)

  Returns:
    A Tensor identical to the input but one larger in the last dimension.  The
    new entries are filled with ones.
  """
  shape = tensor.shape.as_list()
  rank = len(shape)
  if any(elt is None for elt in shape):
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
  else:
    shape[-1] = 1
  if homog_value is not None:
    appendage = homog_value * tf.ones(shape, dtype=tensor.dtype)
  else:
    appendage = tf.ones(shape, dtype=tensor.dtype)
  return tf.concat([tensor, appendage], axis=-1)

def compute_cov(tensor, tensor_right=None, normalizer=None):
  """Compute the empirical second moment of the rows of a 2D Tensor.

  This function is meant to be applied to random matrices for which the true row
  mean is zero, so that the true second moment equals the true covariance.

  Args:
    tensor: A 2D Tensor.
    tensor_right: An optional 2D Tensor. If provided, this function computes
      the matrix product tensor^T * tensor_right instead of tensor^T * tensor.
    normalizer: optional scalar for the estimator (by default, the normalizer is
        the number of rows of tensor).

  Returns:
    A square 2D Tensor with as many rows/cols as the number of input columns.
  """
  if normalizer is None:
    normalizer = get_shape(tensor)[0]
  if tensor_right is None:
    cov = (
        tf.matmul(tensor, tensor, transpose_a=True) / tf.cast(
      normalizer, tensor.dtype))
    return (cov + tf.transpose(cov)) / tf.cast(2.0, cov.dtype)
  else:
    return (tf.matmul(tensor, tensor_right, transpose_a=True) /
            tf.cast(normalizer, tensor.dtype))

def psm_standard(inputs, filter_shape, strides, padding):
  patches = tf.image.extract_patches(
    inputs,
    sizes=[1] + list(filter_shape[0:-2]) + [1],
    strides=strides,
    rates=[1, 1, 1, 1],
    padding=padding)

  flatten_size = onp.prod(filter_shape[0:-1])
  # patches_flat below is the matrix [[A_l]] from the KFC paper (tilde
  # omitted over A for clarity). It has shape M|T| x J|Delta| (eq. 14),
  # where M = minibatch size, |T| = number of spatial locations,
  # |Delta| = number of spatial offsets, and J = number of input maps
  # for convolutional layer l.
  patches_flat = tf.reshape(patches, [-1, flatten_size])
  # We append a homogenous coordinate to patches_flat if the layer has
  # bias parameters. This gives us [[A_l]]_H from the paper.

  patches_flat = append_homog(patches_flat)
  # We call compute_cov without passing in a normalizer. compute_cov uses
  # the first dimension of patches_flat i.e. M|T| as the normalizer by
  # default. Hence we end up computing 1/M|T| * [[A_l]]^T [[A_l]], with
  # shape J|Delta| x J|Delta|. This is related to hat{Omega}_l from
  # the paper but has a different scale here for consistency with
  # ConvOutputKroneckerFactor.
  # (Tilde omitted over A for clarity.)
  return compute_cov(patches_flat)

N = 1  # 32
C = 3
W = 64 # 224
H = 64 # 224
FILTER_SHAPE = (3, 3, 3, -1)
STRIDES =  (1, 1, 1, 1)
PADDING = "VALID"

def patches_second_moment_1(image, kH=FILTER_SHAPE[0], kW=FILTER_SHAPE[1], stride=STRIDES[1], padding=PADDING):
  """The first implementation from Dominik's doc."""
  image_patches = tf.image.extract_patches(
      image, sizes=[1, kH, kW, 1], strides=[1, stride, stride, 1],
      rates=[1, 1, 1, 1], padding=padding)
  image_patches_flat = tf.reshape(
      image_patches, [-1, image_patches.shape[3]])

  output_matrix = tf.matmul(
      image_patches_flat, image_patches_flat, transpose_a=True)
  output_vector = tf.reduce_sum(image_patches_flat, axis=0)

  return output_matrix, output_vector


def patches_second_moment_2(image, kH=FILTER_SHAPE[0], kW=FILTER_SHAPE[1]):
  """Second implementation from Dominik's doc, pure nested loops and numpy."""
  batch_size, height, width, num_channels = image.shape
  output_size = kH * kW * num_channels
  output_matrix = onp.zeros([output_size, output_size])
  output_vector = onp.zeros([output_size])

  for x in range(height - kH + 1):
    for y in range(width - kW + 1):
      patch = image[:, x:x+kH, y:y+kW, :]
      patch = onp.reshape(patch, [batch_size, -1])
      for b in range(batch_size):
        for i in range(output_size):
          for j in range(output_size):
            output_matrix[i, j] += patch[b, i] * patch[b, j]
          output_vector[i] += patch[b, i]

  return output_matrix, output_vector

def patches_second_moment_2_jax(image, kH=FILTER_SHAPE[0], kW=FILTER_SHAPE[1]):
  """patches_second_moment_2 with JAX."""
  batch_size, height, width, num_channels = image.shape
  output_size = kH * kW * num_channels
  output_matrix = onp.zeros([output_size, output_size])
  output_vector = onp.zeros([output_size])

  def body_for_x(x, c):
    def body_for_y(y, c):
      # patch = image[:, x:x + kH, y:y + kW, :]
      patch = lax.dynamic_slice(image, (0, x, y, 0), (batch_size, kH, kW, num_channels))
      patch = onp.reshape(patch, [batch_size, -1])
      def body_for_b(b, c):
        def body_for_i(i, c):
          def body_for_j(j, c):
            output_matrix, output_vector = c
            output_matrix = ops.index_add(output_matrix, (i, j), patch[b, i] * patch[b, j])
            return output_matrix, output_vector
          output_matrix, output_vector = lax.fori_loop(0, output_size, body_for_j, c)
          output_vector = ops.index_add(output_vector, i, patch[b, i])
          return output_matrix, output_vector
        return lax.fori_loop(0, output_size, body_for_i, c)
      return lax.fori_loop(0, batch_size, body_for_b, c)
    return lax.fori_loop(0, width - kW + 1, body_for_y, c)

  output_matrix, output_vector = lax.fori_loop(0, height - kH + 1, body_for_x,
                                               (output_matrix, output_vector))
  return output_matrix, output_vector

def patches_second_moment_opt(image, kH=FILTER_SHAPE[0], kW=FILTER_SHAPE[1]):
  """From Tamas's implementation, last variant.
  https://cs.corp.google.com/piper///depot/google3/experimental/users/tberghammer/tensorflow/patches_second_moment_loop_nest.py
  """
  batch_size, height, width, num_channels = image.shape
  output_size = kH * kW * num_channels
  oH, oW = height - kH + 1, width - kW + 1
  output_matrix = onp.zeros([kH, kW, num_channels, kH, kW, num_channels])
  output_vector = onp.zeros([kH, kW, num_channels])

  for i0 in range(kH):
    for i1 in range(kW):
      chunk0 = image[:, i0:i0 + oH, i1:i1 + oW, :]
      chunk0 = onp.reshape(chunk0, [batch_size * oH * oW, num_channels])
      output_vector[i0, i1, :] += onp.sum(chunk0, axis=0)  # Sum over batches
      for j0 in range(kH):
        for j1 in range(kW):
          chunk1 = image[:, j0:j0 + oH, j1:j1 + oW, :]
          chunk1 = onp.reshape(chunk1, [batch_size*oH*oW, num_channels])
          update = onp.tensordot(chunk0, chunk1, axes=[[0], [0]])
          output_matrix[i0, i1, :, j0, j1, :] += update

  output_matrix = onp.reshape(output_matrix, [output_size, output_size])
  output_vector = onp.reshape(output_vector, [output_size])
  return output_matrix, output_vector

def patches_second_moment_opt_jax(image, kH=FILTER_SHAPE[0], kW=FILTER_SHAPE[1]):
  """patches_second_moment_opt with JAX."""
  batch_size, height, width, num_channels = image.shape
  output_size = kH * kW * num_channels
  oH, oW = height - kH + 1, width - kW + 1
  output_matrix = jnp.zeros([kH, kW, num_channels, kH, kW, num_channels])
  output_vector = jnp.zeros([kH, kW, num_channels])

  def body_for_i0(i0, c):
    def body_for_i1(i1, c):
      # chunk0 = image[:, i0:i0 + oH, i1:i1 + oW, :]
      chunk0 = lax.dynamic_slice(image, (0, i0, i1, 0), (batch_size, oH, oW, num_channels))
      chunk0 = jnp.reshape(chunk0, [batch_size * oH * oW, num_channels])

      output_matrix, output_vector = c
      # output_vector[i0, i1, :] += onp.sum(chunk0, axis=0)
      output_vector = ops.index_add(output_vector, ops.index[i0, i1, :],
                                    jnp.sum(chunk0, axis=0))
      c = output_matrix, output_vector

      def body_for_j0(j0, c):
        def body_for_j1(j1, c):
          # chunk1 = image[:, j0:j0 + oH, j1:j1 + oW, :]
          chunk1 = lax.dynamic_slice(image, (0, j0, j1, 0),
                                     (batch_size, oH, oW, num_channels))
          chunk1 = jnp.reshape(chunk1, [batch_size * oH * oW, num_channels])
          update = jnp.tensordot(chunk0, chunk1, axes=[[0], [0]])

          output_matrix, output_vector = c
          # output_matrix[i0, i1, :, j0, j1, :] += update
          output_matrix = ops.index_add(output_matrix, ops.index[i0, i1, :, j0, j1, :],
                                        update)
          return output_matrix, output_vector
        return lax.fori_loop(0, kW, body_for_j1, c)
      return lax.fori_loop(0, kH, body_for_j0, c)
    return lax.fori_loop(0, kW, body_for_i1, c)

  c = output_matrix, output_vector
  output_matrix, output_vector = lax.fori_loop(0, kH, body_for_i0, c)

  output_matrix = jnp.reshape(output_matrix, [output_size, output_size])
  output_vector = jnp.reshape(output_vector, [output_size])
  return output_matrix, output_vector


class PSMTests(jtu.JaxTestCase):
  def setUp(self) -> None:
    set_env(test_name=self._testMethodName)
    super(PSMTests, self).setUp()

  def test_tf0(self):
    inputs = onp.ones((N, W, H, C), dtype=onp.float32)
    inputs = onp.random.normal(size=(N, W, H, C))
    res_standard = psm_standard(inputs, FILTER_SHAPE, STRIDES, PADDING)
    res_python_1 = patches_second_moment_1(inputs)

    # Baseline
    res_python_2 = patches_second_moment_2(inputs)

    res_python_opt = patches_second_moment_opt(inputs)
    self.assertAllClose(res_python_2[0], res_python_opt[0], check_dtypes=True)
    self.assertAllClose(res_python_2[1], res_python_opt[1], check_dtypes=True)

    res_jax_2 = patches_second_moment_2_jax(inputs)
    self.assertAllClose(res_python_2[0], res_jax_2[0], check_dtypes=True)
    self.assertAllClose(res_python_2[1], res_jax_2[1], check_dtypes=True)

    res_jax_opt = patches_second_moment_opt_jax(inputs)
    self.assertAllClose(res_python_2[0], res_jax_opt[0], check_dtypes=True)
    self.assertAllClose(res_python_2[1], res_jax_opt[1], check_dtypes=True)

  def test_explore(self):
    inputs = jnp.ones((3, 3, 3, 3), dtype=onp.float32)
    upd = jnp.ones((3, 3), dtype=onp.float32) * 2.
    upd1 = upd[:, jnp.newaxis, :, jnp.newaxis]

    v = lax.dynamic_slice(inputs, (0, 1, 0, 1), (3, 1, 3, 1))
    v1 = v + upd1
    a = lax.dynamic_update_slice(inputs, v1, (0, 1, 0, 1))
    # print(a)

    def f(idx):
      return ops.index_add(inputs, ops.index[:, idx:idx+1, :, 1], upd)
    print(api.make_jaxpr(f)(1))
    # print(a1)

  def test_explore_1(self):
    class MyRange(object):
      def __init__(self, upper):
        self.count = -1
        self.upper = upper

      def __iter__(self):
        print("__iter__")
        return self

      def __next__(self):
        self.count += 1
        if self.count >= self.upper:
          print("Stop")
          raise StopIteration
        print("__next__ : {}".format(self.count))
        return self.count

    for i in MyRange(3):
      print(i)
    else:
      print("Here")
    print("After")


def make_jaxpr_and_consts(fun, *args):
  jaxpr, consts = api.make_jaxpr(fun)(*args)
  return core.JaxprAndConsts(jaxpr, consts)


def pp_jaxpr(fun, **pp_flags):
  def wrapped(*args, **kwargs):
    jaxpr, consts = api.make_jaxpr(fun)(*args, **kwargs)
    printer = core.JaxprPrinter(**pp_flags)
    return printer.main(jaxpr, consts)
  return wrapped


class JaxprPrintTests(jtu.JaxTestCase):

  def test_0(self):
    """No constants"""

    def func1(first, second):
      temp = first + jnp.sin(second) * 3.
      return jnp.sum(temp)

    print(api.make_jaxpr(func1)(jnp.zeros(8), jnp.ones(8)))

    print("Use on scalars")
    arg_first = onp.ones(1)
    arg_second = onp.ones(1)
    print(api.make_jaxpr(func1)(arg_first, arg_second))
    print(pp_jaxpr(func1, raw=True)(arg_first, arg_second))
    print(pp_jaxpr(func1)(arg_first, arg_second))

    print("Use on vectors")
    arg_first = onp.ones(16)
    arg_second = arg_first
    print(api.make_jaxpr(func)(arg_first, arg_second))
    print(pp_jaxpr(func, raw=True)(arg_first, arg_second))
    print(pp_jaxpr(func)(arg_first, arg_second))


  def test_trace_through(self):
    """Some trace-through control-flow and higher-order functions"""

    def func2(inner, first, second):
      temp = inner(first) + jnp.sin(second) * 3.
      return jnp.sum(temp)

    def inner(first):
      if first.shape[0] > 4:
        return jnp.cos(first)
      else:
        assert False

    def func3(first, second):
      return func2(inner, first, second)

    print(api.make_jaxpr(func3)(jnp.zeros(8), jnp.ones(8)))

  def test_pytree(self):
    """No constants, with a PyTree"""

    def func4(arg):
      temp = arg[0] + jnp.sin(arg[1]) * 3.
      return jnp.sum(temp)

    print(api.make_jaxpr(func4)((jnp.zeros(8), jnp.ones(8))))


  def test_consts(self):
    """Some constants"""

    def func5(first, second):
      temp = first + jnp.sin(second) * 3. - jnp.ones(first.shape)
      return temp

    def func6(first):
      return func5(first, jnp.ones(first.shape))

    print(api.make_jaxpr(func6)(jnp.ones(2)))
    print(pp_jaxpr(func6)(jnp.ones(2)))

    print(api.make_jaxpr(func6)(jnp.ones(16)))
    print(pp_jaxpr(func6)(jnp.ones(16)))


  def test_cond_0(self):
    def func7(arg):
      return lax.cond(arg >= 0.,
                      arg,
                      lambda xtrue: xtrue + 3.,
                      arg,
                      lambda xfalse: xfalse - 3.)

    print(api.make_jaxpr(func7)(5.))
    print(pp_jaxpr(func7, raw=True)(5.))
    print(pp_jaxpr(func7)(5.))

  def test_cond_pytree_const(self):
    def func8(arg1, arg2):  # arg2 is a pair
      return lax.cond(arg1 >= 0.,
                      arg2,
                      lambda xtrue: xtrue[0] + 3.,
                      arg2,
                      lambda xfalse: xfalse[1] + jnp.ones(1))

    print(api.make_jaxpr(func8)(5., (1., 2.)))
    print(pp_jaxpr(func8, sugar_primitives=False, inline_consts=False)(5., (1., 2.)))
    print(pp_jaxpr(func8)(5., (1., 2.)))


  def test_while(self):
    def func10(arg, n):
      ones = jnp.ones(arg.shape)  # A constant
      return lax.fori_loop(0, n,
                           lambda i, carry: carry + ones * 3. + arg,
                           arg + ones)

    print(api.make_jaxpr(func10)(onp.ones(16), 5))
    print(pp_jaxpr(func10, raw=True)(onp.ones(16), 5))
    print(pp_jaxpr(func10)(onp.ones(16), 5))

  def test_scan(self):
    def func11(arr, extra):
      ones = jnp.ones(arr.shape)  # A constant
      def body(carry, aelems):
        # carry: running dot-product of the two arrays
        # aelems: a pair with corresponding elements from the two arrays
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)

      return lax.scan(body, 0., (arr, ones))


    print(api.make_jaxpr(func11)(onp.ones(16), 5.))
    print(pp_jaxpr(func11)(onp.ones(16), 5.))


  def test_jit_0(self):

    def func12(arg):
      @api.jit
      def inner(x):
        return x + arg * jnp.ones(1)  # Include a constant in the inner function
      return arg + inner(arg - 2.)

    print(api.make_jaxpr(func12)(1.))

  def test_jit_1(self):

    def func13(arg):
      @api.jit
      def inner(x):
        return x + arg * jnp.ones(1)  # Include a constant in the inner function
      return arg + inner(arg - 2.)

    print(api.make_jaxpr(api.jit(func13))(1.))
    print(pp_jaxpr(api.jit(func13))(1.))


  def test_10(self):
    def f(first, second):
      sum = first + jnp.ones(first.shape) + jnp.sin(second) * 3.
      if sum.shape[0] == 1:
        return sum
      else:
        return jnp.sum(sum)

    print("Use on scalars")
    arg_first = onp.ones(1)
    arg_second = onp.ones(1)
    print(api.make_jaxpr(f)(arg_first, arg_second))
    print(pp_jaxpr(f, raw=True)(arg_first, arg_second))
    print(pp_jaxpr(f)(arg_first, arg_second))

    print("Use on vectors")
    arg_first = onp.ones(16)
    print(api.make_jaxpr(f)(arg_first, arg_second))
    print(pp_jaxpr(f, raw=True)(arg_first, arg_second))
    print(pp_jaxpr(f)(arg_first, arg_second))

  def test_cond2(self):
    def f(arg):
      ones = onp.ones(arg.shape, dtype=arg.dtype)
      return lax.cond(arg.shape[0] > 10,
                      ones * 2.,
                      lambda xtrue: xtrue + ones * 3.,
                      arg + ones * 4.,
                      lambda xfalse: xfalse + arg)

    arg_big = onp.ones(10, dtype=onp.float32)
    print(api.make_jaxpr(f)(arg_big))
    print(pp_jaxpr(f, raw=True)(arg_big))
    print(pp_jaxpr(f)(arg_big))


  def test_scan_2(self):

    def f(aa):
      def scan_body(c, a):
        return (c + a + onp.ones(3), 2.)

      return lax.scan(scan_body, onp.zeros(3, dtype=onp.float32), aa)

    aa = onp.ones((3, 3), dtype=onp.float32)

    print(pp_jaxpr(f, raw=True)(aa))
    print(pp_jaxpr(f)(aa))


  def test_freevar2(self):

    def jvp_unlinearized(primal, tangent):
      out, jvp = api.linearize(api.jit(jnp.sin), primal)
      return out, jvp(tangent)

    a = onp.ones((3, 2))
    a_tan = a * 0.5
    f_jit = api.jit(jvp_unlinearized)

    print("api.jit(jnp.sin)")
    print(api.make_jaxpr(api.jit(jnp.sin))(a))
    print(pp_jaxpr(api.jit(jnp.sin))(a))

    print("api.linearize(api.jit(jnp.sin))")
    print(api.make_jaxpr(lambda a: api.linearize(api.jit(jnp.sin), a)(a)))
    print("f_jit")
    print(api.make_jaxpr(f_jit)(a, a_tan))
    print(pp_jaxpr(f_jit)(a, a_tan))

  def testWhileWithClosureJit(self):

    def loop(init, local_limit, inc):

      def loop_cond(state):
        pos, _ = state
        return lax.lt(pos, local_limit)

      def loop_body(state):
        pos, count = state
        f = lambda pos, inc: (lax.add(pos, 1), lax.add(count, inc))
        return api.jit(f)(pos, inc)

      result = lax.while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)
    print(api.make_jaxpr(cloop)(2, 10, 1))
    print(pp_jaxpr(cloop)(2, 10, 1))

  def testWhileWithClosureJit2(self):

    def loop(init, local_limit, inc):

      def loop_cond(state):
        pos, _ = state
        return lax.lt(pos, local_limit)

      def loop_body(state):
        pos, count = state
        f = lambda pos, inc: (lax.add(pos, 1), lax.add(count, inc))
        return f(pos, inc)

      result = lax.while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)
    print(api.make_jaxpr(cloop)(2, 10, 1))
    print(pp_jaxpr(cloop)(2, 10, 1))

  def testNestedJitFreeVar(self):
    def f(x):
      def nested_f(y):
        return jnp.sin(x + y)
      return lax.cond(True,
                      1.,
                      lambda xt: api.jit(nested_f)(xt + x),
                      x + 3.,
                      lambda xf: xf
                      )
    ff = api.jit(f)

    print(api.make_jaxpr(ff)(10.))
    print(pp_jaxpr(ff)(10.))

  def test_pp(self):
    from jax import pprint_util as ppu
    p1 = ppu.PrettyPrint([(1, "first"), (3, "second")])
    print(p1 >> p1)



if __name__ == '__main__':
  absltest.main()
