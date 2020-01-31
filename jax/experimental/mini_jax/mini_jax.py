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
"""Mini-JAX: a pedagogical model of JAX tracing and transformations.

See README.md for high-level description.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import itertools
import numpy as np
import typing
from typing import Any, Dict, Callable, Tuple, Sequence, List, Optional, Union

from jax.pprint_util import PrettyPrint, pp_kv_pairs
from jax.experimental.mini_jax.mini_jax_util import (
  map_tuple, map_list, unzip,
  pp_str, pp_seq
)

TA = typing.TypeVar('TA')
TB = typing.TypeVar('TB')

Value = Union[Any, 'Tracer']  # Either a Python concrete value, or a Tracer
Shape = Tuple[int, ...]

class Globals(object):
  variable_id = itertools.count()  # Unique ids for variables
  function_id = itertools.count()  # Unique ids for function names (for JIT)
  scope_nesting_depth = 0

  @staticmethod
  def reset():
    """Reset counters, for deterministic testing."""
    Globals.variable_id = itertools.count()
    Globals.function_id = itertools.count()
    Globals.scope_nesting_depth = 0


class ExprType(object):
  """Types for symbolic expressions and tracers."""

  def __init__(self, shape: Shape, dtype: type):
    self.shape = shape  # May be () for scalars
    self.dtype = dtype

  @staticmethod
  def val_to_type(v: Value) -> 'ExprType':
    """Given a Value, get its type."""
    if isinstance(v, Tracer):
      return v.expr.etype
    # Must be a constant
    if isinstance(v, int):
      raise NotImplementedError("integer constants not yet implemented")
    elif isinstance(v, float):
      return ExprType((), float)
    elif isinstance(v, np.ndarray):
      assert v.dtype == np.float32 or v.dtype == np.float64
      return ExprType(v.shape, float)
    else:
      assert False, "{}".format(v)

  def __repr__(self):
    if not self.shape:
      return self.dtype.__name__  # print "float" instead of "float[]"
    else:
      shape_str = ",".join([str(d) for d in self.shape])
      return "{}[{}]".format(self.dtype.__name__, shape_str)

  __str__ = __repr__

  # we will use ExprType as hash keys for caching tracing results.
  def __hash__(self):
    return hash((self.dtype.__name__, self.shape))

  def __eq__(self, other):
    return (self.dtype.__name__ == other.dtype.__name__ and
            self.shape == other.shape)


class Operator(enum.Enum):
  """Operators are applied to `Expr` arguments and may have parameters (a Dict)."""
  LITERAL = "literal"
  """Represents a literal, no arguments. params = {var : constant}"""

  VAR = "var"  # 0-ary, params = {'id' : int}
  """Represents a variable, no arguments, params = {id: int}. Variables get
  assigned unique integers.
  """

  ADD = "add"  # binary, no params
  SUB = "sub"  # binary, no params
  MUL = "mul"  # binary, no params
  POW = "pow"
  """Exponentiation, unary, has one param {pow: int}."""

  GE = "ge"  # binary, no params
  GT = "gt"

  JIT_CALL = "jit_call"
  """A jit-wrapped Function. One parameter {func: Function}. Has as many arguments
  as there are func.invars. If the result is a tuple, it must be a tuple with 
  at least 2 elements. In that case, `e.etype` is a tuple, and 
  `len(e.etype) = len(func.results)`.
  """

  COND_GE = "cond_ge"
  """Conditional for greater-equal to 0. Has the following parameters
    func_true: a Function, for the true  branch
    func_false: a Function, for the false branch
    
  The arguments are in order:
    1 argument representing the value to be compared >= 0
    `len(func_true.invars) == len(func_false.invars)` arguments to be passed to 
    the `true_func` and `false_func`.
    
  When pretty-printing, the arguments are actually shown as part of 
  parameters:
    pred_arg:
    args: arguments to be passed to the `true_func` and `false_func`.
  """

  PROJECTION = "proj"
  """Used to extract a single value from an expression that denotes a tuple. 
  There is one argument, whose `e.args[0].etype` is a sequence (can be a
  JIT_CALL or COND_GE). The parameters are {idx: int}.
  """

  def __repr__(self):
    return self.value

  __str__ = __repr__


class Expr(object):
  """Symbolic typed expressions."""

  def __init__(self,
               operator: Operator,
               args: Sequence['Expr'],
               etype: Optional[ExprType] = None,
               **params):
    self.operator = operator
    self.params = params
    self.args = args
    if etype is not None:  # Type check on construction
      self.etype = etype
    else:
      self.etype = Expr.type_check(self.operator, params,
                                   map_tuple(lambda e: e.etype, args))

  def __repr__(self):
    if self.operator == Operator.VAR:
      return "v{}".format(self.params['id'])
    elif self.operator == Operator.LITERAL:
      return repr(self.params['val'])
    else:
      # This is for debugging only, we normally pretty print entire functions
      bindings, names = Expr.three_address_code([self])
      return str(Expr.pp_three_address_code(bindings, names))

  __str__ = __repr__

  @staticmethod
  def type_check(op: Operator, params: Dict,
                 args_et: Sequence[ExprType]) -> ExprType:
    """Computes the type for an operator application given types for arguments."""
    if op in (Operator.ADD, Operator.MUL, Operator.SUB):
      arg1_et, arg2_et = args_et
      if (arg1_et.dtype is float and arg2_et.dtype is float and
          arg1_et.shape == arg2_et.shape):
        return arg1_et
      raise TypeError("{} {} {}".format(op, arg1_et, arg2_et))
    if op == Operator.POW:
      if args_et[0].dtype is float:
        return args_et[0]
      raise TypeError("pow {}".format(args_et[0]))
    if op in (Operator.GE, Operator.GT):
      arg1_et, arg2_et = args_et
      if (arg1_et.dtype is float and arg2_et.dtype is float and
        arg1_et.shape == arg2_et.shape):
        return ExprType((), bool)
      raise TypeError("{} {} {}".format(op, arg1_et, arg2_et))
    raise NotImplementedError

  @staticmethod
  def eval_std_operator(op: Operator, params: Dict,
                        args_v: Sequence[Value]) -> Value:
    """Standard evaluation of an application of an operator,
       given values or tracers for arguments.

    If any of the arguments is a `Tracer`, the result is a `Tracer`, with the
    corresponding symbolic expression. If all arguments are Python concrete values,
    then the expression is evaluated and the result is a Python value.

    Args:
      op, params: the operator and its parameters.
      args_v: the evaluated args
    """
    if op == Operator.VAR:
      assert False  # The VARs are added to the memo table before evaluation
    if op == Operator.LITERAL:
      return params["val"]
    if op == Operator.ADD:
      return args_v[0] + args_v[1]
    if op == Operator.SUB:
      return args_v[0] - args_v[1]
    if op == Operator.MUL:
      return args_v[0] * args_v[1]
    if op == Operator.POW:
      return args_v[0] ** params["pow"]
    if op == Operator.GE:
      return args_v[0] >= args_v[1]
    if op == Operator.GT:
      return args_v[0] > args_v[1]
    if op == Operator.PROJECTION:
      return args_v[0][params["idx"]]  # type: ignore[index]

    if op == Operator.JIT_CALL:
      func = params["func"]
      assert len(args_v) == len(func.invars)
      all_concrete = all([not isinstance(a, Tracer) for a in args_v])
      if all_concrete:  # Arguments are all Python values => JIT and execute
        return Jit.compile_and_execute(func, args_v)

      call_res = Tracer.build(Operator.JIT_CALL, dict(func=func),
                              map_list(Tracer.val_to_tracer, args_v),
                              etype=func.result_type())
      return Tracer.handle_tuple_result(call_res)

    if op == Operator.COND_GE:
      true_func_f = params["true_func"]
      false_func_f = params["false_func"]
      assert len(true_func_f.invars) == len(false_func_f.invars)
      assert len(true_func_f.invars) + 1 == len(args_v)
      assert len(true_func_f.results) == len(false_func_f.results)
      all_concrete = all([not isinstance(a, Tracer) for a in args_v])
      if all_concrete:  # Arguments are all Python values => evaluate now
        branch_args = args_v[1:1 + len(true_func_f.invars)]
        if args_v[0] >= 0.:
          # We use the jitter to evaluate the branch
          return Jit.compile_and_execute(true_func_f, branch_args)
        else:
          return Jit.compile_and_execute(
            false_func_f, branch_args)

      cond_t = Tracer.build(Operator.COND_GE, params,
                            map_tuple(Tracer.val_to_tracer, args_v),
                            etype=true_func_f.result_type())
      return Tracer.handle_tuple_result(cond_t)

    raise NotImplementedError

  def visit_expr(self,
                 visitor: Callable[['Expr', Sequence[TA]], TA],
                 memo: Dict[int, TA]) -> TA:
    """Visit an expression, applying a visitor to all sub-expressions, with memoization.

    Args:
      visitor: a function called with the expression being visited, and
        the result of visiting the `args`. The results of visiting expressions
        should not be mutated; they are stored in a memo table.
      memo: a memo table for storing the results of visiting expressions,
        indexed by id(e).

    Returns: the result of visiting the expression
    """
    sentinel = object()

    def do_visit(e: 'Expr') -> TA:
      res = memo.get(id(e), sentinel)  # type: ignore[arg-type]
      if res is not sentinel:
        return res
      args_v = map_tuple(do_visit, e.args)
      memo[id(e)] = res = visitor(e, args_v)
      return res

    return do_visit(self)

  @staticmethod
  def three_address_code(elst: Sequence['Expr']) -> Tuple[Sequence[Tuple[str, 'Expr']],
                                                          Sequence['Expr']]:
    """Converts a list of `Expr` to 3-address-codes.

    All sub-expressions that are not literals or `VAR` are turned into::

      n = op arg1 ...

    where `n` are new names, `arg1` are all *simple* expressions (literal, VAR,
    or one of the previously defined names. Shared sub-expressions will reuse
    the same name.)

    Args:
      elst: a list of expressions to convert to 3-address-codes. The sub-expressions
        can be shared across them.
    Returns:
      a pair with: a list of bound sub-expressions (each with a name), and
      a list of simple expressions corresponding to the input `elst`.
    """
    bindings = []  # List[Tuple(str, Expr)] list of bound non-simple Expr.
    memo: Dict[int, Any] = dict()  # Share the memo across all expressions in the list
    name_id = itertools.count()  # New names local to this expression list

    def visitor_simplify(e: 'Expr', args_v: Sequence[Expr]) -> Union[Expr, str]:
      """Simplification visitor.
      Returns:
        a simple expression.
      """
      if e.operator in [Operator.VAR, Operator.LITERAL]:
        return e
      else:
        name = "n{}".format(next(name_id))
        binding = (name, Expr(e.operator, args_v, etype=e.etype, **e.params))
        bindings.append(binding)
        return name

    names = map_tuple(lambda e: e.visit_expr(visitor_simplify, memo=memo), elst)
    return (bindings, names)

  @staticmethod
  def pp_three_address_code(bindings: Sequence[Tuple[str, 'Expr']],
                            names: Sequence[str]):
    """Pretty-prints 3-address-code."""

    def pp_binding(binding: Tuple[str, Expr]):
      name, e = binding
      # Special case the printing of some operators
      args, params = e.args, e.params
      if e.operator == Operator.COND_GE:
        # Put the true_args and false_args among parameters, make it easier to read
        params = dict(**params)
        params["pred_arg"] = args[0]
        params["args"] = args[1:1 + len(params["true_func"].invars)]
        args = []
      pp_args = pp_str(" ".join(map_list(str, args)))
      return (pp_str("{} = {}".format(name, e.operator)) >>
              pp_kv_pairs(sorted(params.items())) >>
              pp_str(" ") >> pp_args)

    if len(names) > 1:
      result = pp_str("in (") >> pp_seq(names) >> pp_str(",)")
    else:
      result = pp_str("in {}".format(names[0]))
    return (pp_seq(map_list(pp_binding, bindings), vertical=True) +
            result)

# An environment is a mapping of variables to the Tracers they stand for
# from shallower scope depths.
Environment = Dict[Expr, 'Tracer']


class Tracer(object):
  """A value to be used in lieu of actual Python values for tracing.

  A Tracer implements many Python operators. It carries a symbolic expression
  representing the tracing value. It also carries information necessary for
  separating tracer values from outer scopes.
  """
  __array_priority__ = 1000  # Take priority when overloading over numpy.ndarray

  def __init__(self, expr: Expr,
               scope_nesting_depth: int,
               env: Environment,
               concrete: Optional[Value] = None):
    """
    Args:
      expr: the symbolic expression representing the traced value.
      scope_nesting_depth: the scope depth at which it was built. May be `None`
        for literals.
      env: the environment for the expression: additional free variables
        that appear in `expr` along with the tracers they were introduced for.
      concrete: an optional Python constant representing the actual concrete
        Python value for this Tracer. It is used for deciding conditionals.
        This may be None for values computed from abstract function arguments
        (e.g., for jit, or for other transformations with "abstract=True").
    """
    self.expr = expr
    self.scope_nesting_depth = scope_nesting_depth
    self.env = env
    self.concrete = concrete
    # We export a shape attribute
    self.shape = expr.etype.shape

  def __repr__(self):
    op = self.expr.operator  # Special-case some operators, simplifies debugging
    if op in (Operator.VAR, Operator.LITERAL):
      fmt = str(self.expr)
    else:
      fmt = op
    if op != Operator.LITERAL and self.concrete is not None:
      fmt = "{}={}".format(fmt, self.concrete)
    return str("Tr[{}/{}]".format(fmt, self.scope_nesting_depth))

  __str__ = __repr__

  @staticmethod
  def val_to_tracer(v: Value) -> 'Tracer':
    """Make a Tracer from a Value."""
    if isinstance(v, Tracer):
      return v
    v_et = ExprType.val_to_type(v)
    return Tracer.build(Operator.LITERAL, dict(val=v), (), etype=v_et,
                        concrete=v)

  @staticmethod
  def new_var_tracer_from_type(etype: ExprType) -> 'Tracer':
    """Creates a new Tracer VAR with a given type."""
    return Tracer.build(Operator.VAR,
                        dict(id=next(Globals.variable_id)),
                        (), etype=etype)

  @staticmethod
  def new_var_tracer_from_val(v: Value, abstract=False) -> 'Tracer':
    """Creates a new Tracer VAR from a Value.
    Args:
      v: either a Python constant, or a Tracer
      abstract: whether to force it to be abstract (drop the concrete value).
    """
    v_typ = ExprType.val_to_type(v)
    if abstract:
      concrete = None
    elif isinstance(v, Tracer):
      concrete = v.concrete
    else:
      concrete = v
    return Tracer.build(Operator.VAR,
                        dict(id=next(Globals.variable_id)),
                        (), etype=v_typ,
                        concrete=concrete)

  @staticmethod
  def closure_convert(args_t: Sequence['Tracer'],
                      scope_nesting_depth: int
                      ) -> Tuple[Sequence[Expr], Environment]:
    """Prepares a list of tracing values for use at a given scope depth.

    If there are tracers from shallower scope depths, they are replaced
    with new variables, and an environment is constructed for these variables
    along with the expressions of the shallow tracers they represent.
    """
    env = {}  # Dict[Expr, Tracer]  - map VAR to a Tracer
    args_e = []  # Sequence[Expr]
    for a_t in args_t:
      if a_t.scope_nesting_depth is None or a_t.scope_nesting_depth == scope_nesting_depth:
        # Duplicate VARs in the environment refer to the same Tracer
        env.update(a_t.env)
        args_e.append(a_t.expr)
      else:
        assert a_t.scope_nesting_depth < scope_nesting_depth
        if a_t.expr.operator == Operator.VAR:  # Reuse variables, for convenience
          new_v = a_t.expr
        else:
          new_v = Expr(Operator.VAR, (), etype=a_t.expr.etype,
                       id=next(Globals.variable_id))
        env[new_v] = a_t
        args_e.append(new_v)

    return (args_e, env)

  @staticmethod
  def build(op: Operator, params: Dict,
            args_t: Sequence['Tracer'],
            etype: Optional[ExprType] = None,
            concrete: Optional[Value] = None
            ) -> 'Tracer':
    """Builds a Tracer for an operator applied to arguments.

    Args:
      op: the operator.
      params: the operator parameters.
      args_t: the `Tracer`s for the arguments.
      etype: the optional `ExprType` for the `Expr` being built. If not given,
        then the typeis calculated through type checking.
      concrete: an optional concrete Python value to carry with this Tracer.
    Returns:
      a `Tracer` at the current scope nesting depth.
    """
    args_conc = [arg_t.concrete for arg_t in args_t]
    args_e, args_env = Tracer.closure_convert(args_t,
                                              Globals.scope_nesting_depth)
    expr = Expr(op, tuple(args_e), etype=etype, **params)
    # If all arguments are concrete, and want concrete tracing
    if concrete is None and args_t and all([ct is not None for ct in args_conc]):
      concrete = Expr.eval_std_operator(op, params, args_conc)
    return Tracer(expr, Globals.scope_nesting_depth, args_env,
                  concrete=concrete)

  @staticmethod
  def handle_tuple_result(res: 'Tracer'
                          ) -> Union['Tracer', Sequence['Tracer']]:
    """Wrap tracing val in projections, if we need to return a tuple result.
    Args:
      res: a tracing value, that has a tuple `res.expr.etype`
    """
    if isinstance(res.expr.etype, (tuple, list)):
      assert len(res.expr.etype) > 1
      res_tuple = map_tuple(
        lambda idx: Tracer.build(Operator.PROJECTION,
                                 dict(idx=idx), (res,),
                                 etype=res.expr.etype[idx]),  # type: ignore[index]
        range(len(res.expr.etype)))
      return res_tuple
    else:
      return res

  ############## Overload Python operators
  def __add__(self, other: Value) -> 'Tracer':
    if is_zero(other): return self
    return Tracer.build(Operator.ADD, {}, (self,
                                           self.val_to_tracer(other)))

  def __radd__(self, other: Value) -> 'Tracer':
    if is_zero(other): return self
    return Tracer.build(Operator.ADD, {}, (self.val_to_tracer(other),
                                           self))

  def __mul__(self, other: Value) -> 'Tracer':
    if is_zero(other): return zero_like(self)
    if is_const(other, 1.): return self
    return Tracer.build(Operator.MUL, {}, (self,
                                           self.val_to_tracer(other)))

  def __rmul__(self, other: Value) -> 'Tracer':
    if is_zero(other): return zero_like(self)
    if is_const(other, 1.): return self
    return Tracer.build(Operator.MUL, {}, (self.val_to_tracer(other),
                                           self))

  def __sub__(self, other: Value) -> 'Tracer':
    if is_zero(other): return self
    return Tracer.build(Operator.SUB, {}, (self,
                                           self.val_to_tracer(other)))

  def __rsub__(self, other: Value) -> 'Tracer':
    return Tracer.build(Operator.SUB, {}, (self.val_to_tracer(other),
                                           self))

  def __pow__(self, power: int) -> 'Tracer':
    assert isinstance(power, int)
    if power == 0: return const_like(1., self)
    if power == 1: return self
    return Tracer.build(Operator.POW, dict(pow=power), (self,))

  # Comparison
  def __ge__(self, other: Value):
    return Tracer.build(Operator.GE, {}, (self, self.val_to_tracer(other)))

  def __gt__(self, other: Value):
    return Tracer.build(Operator.GT, {}, (self, self.val_to_tracer(other)))

  def __nonzero__(self):
    # TODO: when is this exactly called?
    if self.concrete is not None:
      assert self.expr.etype.dtype is bool
      return self.concrete
    else:
      msg = "Boolean test not supported on abstract values"
      raise TypeError(msg)

  __bool__ = __nonzero__


class Function(object):
  """Denotes a closed function in the symbolic expression language."""

  def __init__(self,
               invars: Sequence[Expr],
               results: Sequence[Expr]):
    """
    If there are more than 1 results, then and only then the function returns
    a tuple.

    Args:
      invars: the variables that occur in the results, including free variables.
        These are VAR expressions.
      results: a list of `Expr` that depend only on the VARs in `invars`.
    """
    self.invars = invars
    self.results = results

  def __repr__(self):
    return str(self.pp())

  __str__ = __repr__

  def pp(self):
    """Pretty prints a function definition, using 3-address-codes."""
    bindings, names = Expr.three_address_code(self.results)
    return ((pp_str("{lambda ") >> pp_seq(self.invars) >> pp_str(".")) +
            (pp_str("  # ") >> pp_seq(["{}: {}".format(v, v.etype)
                                       for v in self.invars], hsep=", ")) +
            Expr.pp_three_address_code(bindings, names).indent(2)
            >> pp_str("}"))

  @staticmethod
  def trace_user_function(func: Callable,
                          args_v: Sequence[Value],
                          abstract: bool = True,
                          cache: bool = True,
                          ) -> Tuple['Function', Sequence['Tracer']]:
    """Traces a user function given values for the arguments.

    Watches for usage (capture) of tracers from outer tracing, introduces new
    variables for them, and returns them as extra arguments to be passed to
    the closed Function.

    Args:
      func: a Python user function to trace using Tracer arguments.
      args_v: the values corresponding to the `func` arguments. May be Python
        concrete values, or Tracer values.
      abstract: whether to force arguments to be abstract (no constants carried
        in the Tracer, cheaper, but cannot handle data-dependent control-flow).
      cache: whether to allow caching of the resulting Function. The result is
        cached only if `abstract` and if there are no captured tracers.

    Returns: a pair of a closed `Function` along with a list of tracers
      captured from outer scopes used. The tail of the `invars` of the function
      correspond to the captured tracers.
    """
    if cache:
      cache_key = ("trace", map_tuple(ExprType.val_to_type, args_v))
      # Do not cache when we do concolic testing, because the path may
      # depend on actual concrete values.
      result = Cache.get(func, cache_key) if abstract else None
      if result is not None:
        return result

    Globals.scope_nesting_depth += 1
    try:
      scope_nesting_depth = Globals.scope_nesting_depth
      args_t = [Tracer.new_var_tracer_from_val(arg_v, abstract=abstract)
                for arg_v in args_v]
      res = func(*args_t)
      if not isinstance(res, tuple):
        res = (res,)

      # res may contain literals, turn them into Tracer
      res_t = tuple(Tracer.val_to_tracer(r) for r in res)
      res_e, res_env = Tracer.closure_convert(res_t, scope_nesting_depth)

      # The freevars from res_env. Sort by VAR id, for deterministic results
      freevars = sorted(res_env.keys(), key=lambda v: v.params["id"])
      freevars_env = [res_env[fv] for fv in freevars]

      func_f = Function([v_t.expr for v_t in args_t] + freevars, res_e)
      result = func_f, freevars_env
      # Do not cache if we captured tracers
      if cache and abstract and not freevars_env:
        Cache.set(func, cache_key, result)
      return result
    finally:
      Globals.scope_nesting_depth -= 1

  def transform_function(
      self,
      transform_key: Any,
      evaluator: Callable[..., Any],  # Takes a Function and a sequence of Tracer
      extra_args_typ: Sequence[ExprType] = None
  ) -> 'Function':
    """Runs a traceable evaluator over a Function to produce transformed Function.

    This is the workhorse of composable transformations. Given an evaluator
    that evaluates a `Function` with special semantics for operators and uses
    only overloaded operations on the arguments, returns the transformed
    `Function`.

    Args:
      transform_key: a hashable value to be used for caching the result
        of the transformation (by `self`).
      evaluator: a Python traceable function, that given a `Function` and
        a set of Tracers evaluates the Function according to the transformed
        semantics.
      extra_args_typ: optional, the list of types of additional arguments for
        the resulting function.

    Returns:
      a transformed Function.
    """
    result = Cache.get(self, transform_key)
    if result is not None:
      return result

    # Start with the current arg types
    args_typ = [iv.etype for iv in self.invars]
    if extra_args_typ is not None:
      args_typ += extra_args_typ
    args_t = map_tuple(Tracer.new_var_tracer_from_type, args_typ)
    res_t = evaluator(self, *args_t)
    if not isinstance(res_t, tuple):
      res_t = (res_t,)
    # res may contain literals, turn them into Tracer
    res_t = [Tracer.val_to_tracer(r) for r in res_t]
    result = Function([v_t.expr for v_t in args_t], [r_t.expr for r_t in res_t])
    Cache.set(self, transform_key, result)
    return result

  def result_type(self):
    """The result type of the function.
    The result type is a tuple iff there are more than 1 results.
    """
    res = [result.etype for result in self.results]
    return res[0] if len(res) == 1 else res

  def make_evaluator(
      self,
      in_args_v: Sequence[TA],
      eval_expr: Optional[Callable[[Expr, Sequence[TA]], TA]] = None,
      eval_operator: Optional[Callable[[Operator, Dict, Sequence[TA]], TA]] = None,
      **eval_params):
    """Make a memoized expression evaluator for this Function.

    Args:
      in_args_v: the values for the Function invars.
      eval_expr: an evaluator to be called (once) for each sub-expression,
        given values for its arguments.
      eval_operator: an evaluator to be called (once) for each operator,
        given values for its arguments. Exactly one of `eval_expr` and
        `eval_operator` must be given.
      eval_params: keyword parameters to be passed to the eval functions.

    Returns:
      the list of values corresponding to the function results
    """
    assert len(self.invars) == len(in_args_v)
    memo: Dict[int, TA] = {}
    for iv, v in zip(self.invars, in_args_v):
      memo[id(iv)] = v

    def eval_expr_visitor(e, args_v):
      if eval_expr is None:
        return eval_operator(e.operator, e.params, args_v, **eval_params)
      else:
        return eval_expr(e, args_v, **eval_params)

    return (lambda e: e.visit_expr(eval_expr_visitor, memo=memo))


class Ops(object):
  """Special primitive operations that operate either on values of tracers"""

  @staticmethod
  def cond_ge(to_test: Value, true_func: Callable,
              false_func: Callable, args: Sequence[Value]
              ) -> Value:
    # Collect the branch functions
    if (not isinstance(args, (list, tuple))):
      raise TypeError("args must be tuples for cond_ge")

    to_test_type = ExprType.val_to_type(to_test)
    assert to_test_type.dtype is float and not to_test_type.shape, (
      "guard of cond_ge must be a float: {}".format(to_test_type))
    true_func_f, true_func_env = (
      Function.trace_user_function(true_func, args, abstract=False))
    false_func_f, false_func_env = (
      Function.trace_user_function(false_func, args, abstract=False))
    assert str(true_func_f.result_type()) == str(false_func_f.result_type()), (
      "{} != {}".format(true_func_f.result_type(), false_func_f.result_type()))
    # Merge the true_func_env and false_func_env
    merged_func_env: Sequence[Tracer] = true_func_env + false_func_env
    def mk_new_var(t: Tracer) -> Expr:
      return Expr(Operator.VAR, (), etype=t.expr.etype,
                  id=next(Globals.variable_id))
    true_func_f.invars = true_func_f.invars + [mk_new_var(t) for t in
                                               false_func_env]
    false_func_f.invars = (false_func_f.invars[0:len(args)] +
                           [mk_new_var(t) for t in true_func_env] +
                           false_func_f.invars[len(args):])

    # Update the functions
    return Expr.eval_std_operator(
      Operator.COND_GE,
      dict(true_func=true_func_f, false_func=false_func_f),
      tuple(itertools.chain([to_test], args, merged_func_env)))


class Cache(object):
  sentinel = object()

  @staticmethod
  def get(obj, key):
    cache = Cache._get_cache(obj)
    res = cache.get(key, Cache.sentinel)
    if res is Cache.sentinel:
      cache["__misses"] += 1
      return None
    else:
      cache["__hits"] += 1
      return res

  @staticmethod
  def set(obj, key, val):
    cache = Cache._get_cache(obj)
    cache[key] = val

  @staticmethod
  def get_info(obj):
    cache = Cache._get_cache(obj)
    return dict(misses=cache["__misses"], hits=cache["__hits"])

  @staticmethod
  def _get_cache(obj):
    cache = getattr(obj, "_cache", None)
    if cache is None:
      cache = {"__misses": 0, "__hits": 0}
      setattr(obj, "_cache", cache)
    return cache

def const_like(const: float, v: Value, shape: Optional[Shape] = None):
  """A constant with the given shape, or the shape of the value"""
  shape = val_to_shape(v) if shape is None else shape
  if shape is None:
    return const
  else:
    return const * np.ones(shape, dtype=np.float32)

def zero_like(v: Value, shape: Optional[Shape] = None) -> Value:
  """Zeros in the shape of a Value"""
  return const_like(0., v, shape=shape)

def zero_like_shape(shape: Optional[Shape]) -> Value:
  """Zeros in the shape of a Value"""
  return const_like(0., None, shape=shape)

def is_const(v: Value, const: float) -> bool:
  """Checks if all elements of the tensor are equal to `const`."""
  return not isinstance(v, Tracer) and np.allclose(v, const_like(const, v))

def is_zero(v: Value) -> bool:
  """Checks if all elements of the tensor are 0."""
  return is_const(v, 0.)

def val_to_shape(v: Value):
  if isinstance(v, np.ndarray):
    return v.shape
  elif isinstance(v, float):
    return ()
  elif isinstance(v, Tracer):
    return v.expr.etype.shape
  else:
    assert False

############ TRACE ##############
def trace(func: Callable, abstract: bool = True,
          cache: bool = True) -> Callable[..., Function]:
  """
  Returns: a function that when applied to `func` arguments traces the function
    and returns the `Function`. If 'abstract' then force arguments to be
    abstract during tracing.
  """

  def do_trace(*args: Sequence[Value]) -> Function:
    assert Globals.scope_nesting_depth == 0, "Outside any other tracers"
    Globals.reset()  # Produces more deterministic results
    func_f, func_f_env = Function.trace_user_function(func, args,
                                                      abstract=abstract,
                                                      cache=cache)
    assert not func_f_env  # We usually trace only outside other transformations
    return func_f

  return do_trace


# Circular dependency Jit<->Mini-jax core
from jax.experimental.mini_jax.mini_jax_jit import Jit
