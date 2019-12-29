# Copyright 2019 Google LLC
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
"""
Transformation: count-flops
---------------------------

Transform functions into FLOPS counters (count of floating point operations
that are used to compute the result, and are visible to tracing). This relies
on a simple cumulative performance model for the operations:
 * use of a variable or constant costs 0.
 * most arithmetic operations cost 1.
 * exponentiation costs ceil(log2(power)).
 * conditional costs 1 + the cost of the taken branch
 * a call into a jit scope costs 1 + number of arguments

Only the flops for the computations that are actually used to compute the
result of the function are included.

The flops may depend on the actual values (e.g., for conditionals). If the flops
in the body of a JIT are a constant then they are lifted outside the JIT,
otherwise they are done inside the JIT. If the flops of the conditional branches
are the same constants, they are lifted outside the conditional.

Concrete examples are in `tests/mini_jax_flops_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
from typing import Dict, Callable, Tuple, Sequence, List

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value
)
from jax.experimental.mini_jax.mini_jax_util import map_list

FlopsVal = Value  # In principle, we could count more things


class Flops(object):
  """Methods related to counting flops."""

  def eval_function(self, func: Function, *args_v: Sequence[Value]
                    ) -> Sequence[Value]:
    # Prepare an expression evaluator for when flops are data-dependent
    eval_std_expr = func.make_evaluator(args_v,
                                        eval_operator=Expr.eval_std_operator)

    # The Expr.visit_expr will only visit an expression once, but the result
    # of the expression may be used multiple times in parent expressions. To
    # ensure we only count once, we pass a mutable accumulator to the visitor,
    # instead of carrying the flops with the expression values.
    accum_flops = [0.]  # type: List[Value]  - flops accumulator
    eval_flops_expr = func.make_evaluator(args_v, eval_expr=self.eval_expr,
                  eval_std_expr=eval_std_expr,
                  accum_flops=accum_flops)
    _ = [eval_flops_expr(result) for result in func.results]
    return accum_flops[0]

  def eval_expr(self, e: Expr, args_v: Sequence[FlopsVal],
                eval_std_expr: Callable[[Expr], Value] = None,
                accum_flops: List[FlopsVal] = None) -> FlopsVal:
    """Computes the flops, excluding the flops of the arguments.

    Will be called one per distinct Expr object.
    Args:
      e: the expression whose flops cost to calculate.
      eval_expr: in the cases when the flops are data-dependent, an expression
        evaluator.
    Returns:
      cost of flops, not including the flops of `e.args`.
    """

    def accum(flops):
      accum_flops[0] += flops
      return flops

    if e.operator in (Operator.VAR, Operator.LITERAL, Operator.PROJECTION):
      return accum(0.)
    if e.operator in (Operator.ADD, Operator.SUB, Operator.MUL):
      return accum(1.)
    if e.operator == Operator.POW:
      return accum(float(math.ceil(math.log2(e.params["pow"]))))

    if e.operator == Operator.JIT_CALL:
      func = e.params["func"]
      flops_func = func.transform_function(self.eval_function)
      # Perhaps the flops of the function is data-independent
      if flops_func.results[-1].operator == Operator.LITERAL:
        # Add in the cost of the call itself
        return accum(1. + float(len(func.invars)) +
                     flops_func.results[-1].params["val"])
      else:
        e_args_v = map_list(eval_std_expr, e.args)
        res = Expr.eval_std_operator(Operator.JIT_CALL,
                                     dict(func=flops_func),
                                     e_args_v)
        return accum(1. + float(len(func.invars)) + res)

    if e.operator == Operator.COND_GE:
      true_func_f = e.params["true_func"]
      true_func_flops = true_func_f.transform_function(self.eval_function)
      false_func_f = e.params["false_func"]
      false_func_flops = false_func_f.transform_function(self.eval_function)
      # If both branches have the same flops count, lift it out
      if (true_func_flops.results[-1].operator == Operator.LITERAL and
          false_func_flops.results[-1].operator == Operator.LITERAL and
          true_func_flops.results[-1].params["val"] ==
          false_func_flops.results[-1].params["val"]):
        return accum(1. + true_func_flops.results[-1].params["val"])
      else:
        res_cond = Expr.eval_std_operator(
          Operator.COND_GE,
          dict(true_func=true_func_flops,
               false_func=false_func_flops),
          map_list(eval_std_expr, e.args))
        return accum(1. + res_cond)


def count_flops(func: Callable, abstract: bool = True) -> Callable:
  """Wrap a function into a flops counter.

  The counting of flops is done as much as possible statically.

  Params:
    func: a traceable function
    abstract: whether to force arguments to be abstract
  Returns:
    a function that when applied to arguments will return a counter of the
    flops performed.
  """

  def wrapped_flops(*args: Sequence[Value]):
    func_f, func_f_env = Function.trace_user_function(func, args,
                                                      abstract=abstract)
    res_flops = Flops().eval_function(func_f, *args, *func_f_env)
    return res_flops

  return wrapped_flops
