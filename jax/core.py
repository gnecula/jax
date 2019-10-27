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

from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple, Counter, defaultdict
import itertools as it
from weakref import ref
import threading
import types
import numpy as onp

import six

from . import linear_util as lu
from .util import safe_zip, safe_map, partial, curry, split_dict, split_list
from .pprint_util import PrettyPrint, pp, vcat, hcat, pp_kv_pairs

# TODO(dougalm): the trace cache breaks the leak detector. Consisder solving.
check_leaks = False
# TODO(dougalm): put this behind a flag that's enabled during testing
skip_checks = True  # not __debug__  # google doesn't use -O

zip = safe_zip
map = safe_map


# -------------------- jaxprs --------------------

class Jaxpr(object):
  def __init__(self, constvars, freevars, invars, outvars, eqns):
    self.constvars = list(constvars)
    self.freevars = list(freevars)
    self.invars = list(invars)
    self.outvars = list(outvars)
    self.eqns = list(eqns)

  def __str__(self):
    return str(pp_jaxpr(self))
  __repr__ = __str__

class TypedJaxpr(object):
  def __init__(self, jaxpr, literals, in_avals, out_avals):
    assert type(jaxpr) is Jaxpr
    assert len(literals) == len(jaxpr.constvars)
    assert len(in_avals) == len(jaxpr.invars)
    assert all(isinstance(aval, AbstractValue) for aval in in_avals)
    assert all(isinstance(aval, AbstractValue) for aval in out_avals)
    assert not jaxpr.freevars

    self.jaxpr = jaxpr
    self.literals = list(literals)
    self.in_avals = list(in_avals)
    self.out_avals = list(out_avals)

  def __iter__(self):
    return iter((self.jaxpr, self.literals, self.in_avals, self.out_avals))

  def __str__(self):
    # TODO(mattjj): improve this with type annotations?
    return str(pp_jaxpr(self.jaxpr))
  __repr__ = __str__

@curry
def jaxpr_as_fun(typed_jaxpr, *args):
  return eval_jaxpr(typed_jaxpr.jaxpr, typed_jaxpr.literals, (), *args)


JaxprEqn = namedtuple('JaxprEqn', ['invars', 'outvars', 'primitive',
                                   'bound_subjaxprs', 'params'])
JaxprEqn.__repr__ = JaxprEqn.__str__ = lambda eqn: str(pp_eqn(eqn)).rstrip()
new_jaxpr_eqn = JaxprEqn


class Var(object):
  def __init__(self, count, suffix):
    self.count = count
    self.suffix = suffix

  def __repr__(self):
    rem = self.count
    s = ''
    while True:
      rem, i = rem // 26, rem % 26
      s = chr(97 + i % 26) + s
      if not rem:
        break
    return s + self.suffix

def gensym(suffix):
  counter = it.count()
  return lambda: Var(next(counter), suffix)

class Literal(object):
  __slots__ = ["val", "hash"]

  def __init__(self, val):
    self.val = val
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError):
          self.hash = None

  def __hash__(self):
    return id(self.val) if self.hash is None else self.hash

  def __eq__(self, other):
    return self.val is other.val if self.hash is None else self.val == other.val

  def __repr__(self):
    if self.hash is None:
      return 'Literal(val={}, hashable={})'.format(self.val, self.hashable)
    else:
      return '{}'.format(self.val)

literalable_types = set()

class Primitive(object):
  multiple_results = False  # override for multi-output primitives

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return '{}'.format(self.name)

  def bind(self, *args, **kwargs):
    assert skip_checks or all(isinstance(arg, Tracer)
                              or valid_jaxtype(arg) for arg in args), args
    top_trace = find_top_trace(args)
    if top_trace is None:
      return self.impl(*args, **kwargs)

    tracers = map(top_trace.full_raise, args)
    out_tracer = top_trace.process_primitive(self, tracers, kwargs)
    if self.multiple_results:
      return map(full_lower, out_tracer)
    else:
      return full_lower(out_tracer)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = abstract_eval
    return abstract_eval

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **kwargs):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **kwargs):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))


# -------------------- lifting --------------------


def eval_jaxpr(jaxpr, consts, freevar_vals, *args):
  def read(v):
    if type(v) is Literal:
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  map(write, jaxpr.freevars, freevar_vals)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    subfuns = [partial(eval_jaxpr, subjaxpr, map(read, const_bindings),
                                             map(read, freevar_bindings))
               for subjaxpr, const_bindings, freevar_bindings
               in eqn.bound_subjaxprs]
    subfuns = map(lu.wrap_init, subfuns)
    ans = eqn.primitive.bind(*(subfuns + in_vals), **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)


def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val


def find_top_trace(xs):
 try:
   top_trace = max((x.trace for x in xs if isinstance(x, Tracer)),
                   key=attrgetter('level'))
 except ValueError:
   return None
 else:
   return type(top_trace)(top_trace.master, cur_sublevel())


# -------------------- tracing --------------------


class Trace(object):
  def __init__(self, master, sublevel):
    self.master = master
    self.level = master.level
    self.sublevel = sublevel

  def full_raise(self, val):
    if not isinstance(val, Tracer):
      return self.pure(val)
    level = self.level
    sublevel = self.sublevel
    if val.trace.master is self.master:
      if val.trace.sublevel == sublevel:
        return val
      elif val.trace.sublevel < sublevel:
        return self.sublift(val)
      else:
        raise Exception("Can't lift sublevels {} to {}"
                        .format(val.trace.sublevel, sublevel))
    elif val.trace.level < level:
      if val.trace.sublevel > sublevel:
        raise Exception("Incompatible sublevel: {}, {}"
                        .format(val.trace, (level, sublevel)))
      return self.lift(val)
    elif val.trace.level > level:
      raise Exception("Can't lift {} to {}".format(val, self))
    elif val.trace.level == self.level:
      raise Exception("Different traces at same level: {}, {}".format(val, self))
    else:
      raise Exception("Can't lift {} to {}".format(val, self))


  def pure(self, val):
    assert False

  def lift(self, tracer):
    assert False

  def sublift(self, tracer):
    assert False

  def __repr__(self):
    return '{}(level={}/{})'.format(
        self.__class__.__name__, self.level, self.sublevel)


class Tracer(object):
  __array_priority__ = 1000
  __slots__ = ['trace', '__weakref__']

  def __array__(self, *args, **kw):
    raise Exception("Tracer can't be used with raw numpy functions. "
                    "You might have\n  import numpy as np\ninstead of\n  import jax.numpy as np")

  def __init__(self, trace):
    self.trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    return self.aval._len(self)

  @property
  def aval(self):
    assert False

  def __neg__(self): return self.aval._neg(self)
  def __pos__(self): return self.aval._pos(self)
  def __eq__(self, other): return self.aval._eq(self, other)
  def __ne__(self, other): return self.aval._ne(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __le__(self, other): return self.aval._le(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __ge__(self, other): return self.aval._ge(self, other)
  def __abs__(self): return self.aval._abs(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __sub__(self, other): return self.aval._sub(self, other)
  def __rsub__(self, other): return self.aval._rsub(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __div__(self, other): return self.aval._div(self, other)
  def __rdiv__(self, other): return self.aval._rdiv(self, other)
  def __truediv__(self, other): return self.aval._truediv(self, other)
  def __rtruediv__(self, other): return self.aval._rtruediv(self, other)
  def __floordiv__(self, other): return self.aval._floordiv(self, other)
  def __rfloordiv__(self, other): return self.aval._rfloordiv(self, other)
  def __divmod__(self, other): return self.aval._divmod(self, other)
  def __rdivmod__(self, other): return self.aval._rdivmod(self, other)
  def __mod__(self, other): return self.aval._mod(self, other)
  def __rmod__(self, other): return self.aval._rmod(self, other)
  def __pow__(self, other): return self.aval._pow(self, other)
  def __rpow__(self, other): return self.aval._rpow(self, other)
  def __matmul__(self, other): return self.aval._matmul(self, other)
  def __rmatmul__(self, other): return self.aval._rmatmul(self, other)
  def __and__(self, other): return self.aval._and(self, other)
  def __rand__(self, other): return self.aval._rand(self, other)
  def __or__(self, other): return self.aval._or(self, other)
  def __ror__(self, other): return self.aval._ror(self, other)
  def __xor__(self, other): return self.aval._xor(self, other)
  def __rxor__(self, other): return self.aval._rxor(self, other)
  def __invert__(self): return self.aval._invert(self)
  def __lshift__(self, other): return self.aval._lshift(self, other)
  def __rshift__(self, other): return self.aval._rshift(self, other)
  def __getitem__(self, idx): return self.aval._getitem(self, idx)
  def __nonzero__(self): return self.aval._nonzero(self)
  def __bool__(self): return self.aval._bool(self)
  def __float__(self): return self.aval._float(self)
  def __int__(self): return self.aval._int(self)
  def __long__(self): return self.aval._long(self)
  def __complex__(self): return self.aval._complex(self)
  def __hex__(self): return self.aval._hex(self)
  def __oct__(self): return self.aval._oct(self)

  def __setitem__(self, idx, val):
    raise TypeError("JAX 'Tracer' objects do not support item assignment")

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert skip_checks or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except KeyError:
      raise AttributeError(
          "{} has no attribute {}".format(self.__class__.__name__, name))
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        if six.PY3:
          return types.MethodType(attr.fun, self)
        else:
          return types.MethodType(attr.fun, self, None)
      else:
        return attr

  def __repr__(self):
    return 'Traced<{}>with<{}>'.format(self.aval, self.trace)


# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class MasterTrace(object):
  def __init__(self, level, trace_type):
    self.level = level
    self.trace_type = trace_type

  def __repr__(self):
    return "MasterTrace({},{})".format(self.level, self.trace_type.__name__)

  def __hash__(self):
    return hash((self.level, self.trace_type))

  def __eq__(self, other):
    return self.level == other.level and self.trace_type == other.trace_type


class TraceStack(object):
  def __init__(self):
    self.upward = []
    self.downward = []

  def next_level(self, bottom):
    if bottom:
      return - (len(self.downward) + 1)
    else:
      return len(self.upward)

  def push(self, val, bottom):
    if bottom:
      self.downward.append(val)
    else:
      self.upward.append(val)

  def pop(self, bottom):
    if bottom:
      self.downward.pop()
    else:
      self.upward.pop()

  def __repr__(self):
    return  'Trace stack\n{} ---\n{}'.format(
      map('  {}\n'.format, self.upward[::-1]),
      map('  {}\n'.format, self.downward))


class Sublevel(int): pass

# The global state of the tracer is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class TraceState(threading.local):
  def __init__(self):
    self.trace_stack = TraceStack()
    self.substack = [Sublevel(0)]

trace_state = TraceState()


def cur_sublevel():
  return trace_state.substack[-1]


@contextmanager
def new_master(trace_type, bottom=False):
  level = trace_state.trace_stack.next_level(bottom)
  master = MasterTrace(level, trace_type)
  trace_state.trace_stack.push(master, bottom)

  try:
    yield master
  finally:
    trace_state.trace_stack.pop(bottom)

  if check_leaks:
    t = ref(master)
    del master
    if t() is not None:
      print(trace_state.trace_stack)
      raise Exception('Leaked trace {}'.format(t()))


@contextmanager
def new_sublevel():
  sublevel = Sublevel(len(trace_state.substack))
  trace_state.substack.append(sublevel)
  try:
    yield
  finally:
    trace_state.substack.pop()

  if check_leaks:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      raise Exception('Leaked sublevel {}'.format(t()))

# -------------------- abstract values --------------------


class AbstractValue(object):
  __slots__ = []

  def at_least_vspace(self):
    assert False

  def __repr__(self):
    try:
      kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def strip_weak_type(self):
    return self

class Bot(AbstractValue): pass

bot = Bot()

class AbstractUnit(AbstractValue):
  def join(self, other): return self
  def _eq(self, self_traced, other): return get_aval(other) is self

abstract_unit = AbstractUnit()

def lattice_join(x, y):
  if x is None:
    return y
  elif y is None:
    return x
  elif isinstance(x, type(y)):
    return y.join(x)
  elif isinstance(y, type(x)):
    return x.join(y)
  else:
    raise TypeError((x, y))


def valid_jaxtype(x):
  try:
    concrete_aval(x)
  except TypeError:
    return False
  else:
    return True


def concrete_aval(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("{} is not a valid Jax type".format(type(x)))


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return concrete_aval(x)


pytype_aval_mappings = {}


class Unit(object):
  def __repr__(self): return '*'
unit = Unit()
literalable_types.add(Unit)

class UnitVar(object):
  def __repr__(self): return '*'
unitvar = UnitVar()

pytype_aval_mappings[Unit] = lambda _: abstract_unit

identity_p = Primitive('id')
identity_p.def_impl(lambda x: x)
identity_p.def_custom_bind(lambda x: x)

# ------------------- Call -------------------


def apply_todos(todos, outs):
  while todos:
    outs = map(full_lower, todos.pop()(outs))
  return outs

@lu.transformation_with_aux
def process_env_traces(primitive, level, params_tuple, *args):
  outs = yield args, {}
  params = dict(params_tuple)
  todo = []
  while True:
    tracers = [x for x in outs if isinstance(x, Tracer) and x.trace.level > level]
    if tracers:
      ans = max(tracers, key=lambda x: x.trace.level)
    else:
      break
    trace = type(ans.trace)(ans.trace.master, cur_sublevel())
    outs = map(trace.full_raise, outs)
    outs, cur_todo = trace.post_process_call(primitive, outs, params)
    todo.append(cur_todo)
  yield outs, todo

def call_bind(primitive, f, *args, **params):
  top_trace = find_top_trace(args)
  level = trace_state.trace_stack.next_level(True) if top_trace is None else top_trace.level
  params_tuple = tuple(params.items())
  f, env_trace_todo = process_env_traces(f, primitive, level, params_tuple)
  if top_trace is None:
    with new_sublevel():
      outs = primitive.impl(f, *args, **params)
  else:
    tracers = map(top_trace.full_raise, args)
    outs = map(full_lower, top_trace.process_call(primitive, f, tracers, params))
  return apply_todos(env_trace_todo(), outs)


def call_impl(f, *args, **params):
  return f.call_wrapped(*args, **params)


call_p = Primitive('call')
call = partial(call_bind, call_p)
call_p.def_custom_bind(call)
call_p.def_impl(call_impl)


# ------------------- Jaxpr printed representation -------------------

def check_jaxpr(jaxpr):
  def context():
    return "\njaxpr:\n{}\n".format(jaxpr)

  def read_env(env, v):
    if v not in env and type(v) is not Literal:
      raise Exception("Variable '{}' not defined".format(v) + context())

  def write_env(env, v):
    if v in env:
      raise Exception("Variable {} already bound".format(v) + context())
    env.add(v)

  env = set()
  read = partial(read_env, env)
  write = partial(write_env, env)

  write(unitvar)
  map(write, jaxpr.constvars)
  map(write, jaxpr.freevars)
  map(write, jaxpr.invars)
  for eqn in jaxpr.eqns:
    map(read, eqn.invars)
    for subjaxpr, constvars, freevars in eqn.bound_subjaxprs:
      map(read, freevars)
      map(read, constvars)
      check_jaxpr(subjaxpr)
    map(write, eqn.outvars)
  map(read, jaxpr.outvars)


def pp_vars(vs):
    return ' '.join(map(str, vs))

def pp_eqn(eqn):
  lhs = pp_vars(eqn.outvars)
  pp_subexpr = pp('')
  if eqn.bound_subjaxprs:
    for subjaxpr, const_vars, bound_vars in eqn.bound_subjaxprs:
      pp_subexpr = pp_subexpr + (
          pp_jaxpr(subjaxpr).indent(2)
          >> pp(' [ {} ; {} ]'.format(pp_vars(const_vars),
                                      pp_vars(bound_vars))))
  return (pp('{} = '.format(lhs)) >>
          pp(eqn.primitive.name) >> pp_kv_pairs(eqn.params.items())
          >> pp(' ') >> pp(pp_vars(eqn.invars))) + pp_subexpr

def pp_jaxpr(jaxpr):
  return (pp('{{ lambda {} ; {} ; {}.'.format(pp_vars(jaxpr.constvars),
                                              pp_vars(jaxpr.freevars),
                                              pp_vars(jaxpr.invars))) +
          ((pp('let ') >>
            vcat(map(pp_eqn, jaxpr.eqns))) +
           pp('in {} }}'.format(jaxpr.outvars))).indent(2))


class JaxprPrinter(object):

  def __init__(self, raw=False, inline_consts=True,
               sugar_primitives=True):
    """Configures a pretty printer.

    Args:
      raw: if True then sets all the other configuration flags to show
        raw jaxpr.
      inline_consts: inline small constants.
      sugar_primitives: tries to simplify the primitives
    """
    self.inline_consts = inline_consts
    self.sugar_primitives = sugar_primitives
    if raw:
      self.inline_consts = False
      self.sugar_primitives = False
    self.inline_const_max_size = 8
    self.const_counter = it.count()

    self.var_rewrites = {}  # Map Var from jaxpr to a string
    self.next_var_count = 0
    # Stack of scopes, last is current. Each scope is a pair with
    # value of next_var_count on entering, and the list of orig_vars
    # defined in the scope.
    self.scopes = []

  def push_scope(self):
    """Starts a new scope."""
    self.scopes.append((self.next_var_count, []))

  def pop_scope(self, pp):
    """Pops the current scope.
    Args:
      pp: a PrettyPrinter to return after poping the scope
    """
    curr_scope = self.scopes.pop()
    self.next_var_count = curr_scope[0]
    for v in curr_scope[1]:
      del self.var_rewrites[v]
    return pp

  def define_vars(self, orig_vars, vals):
    """Defines some Vars in the current scope.
    Args:
      orig_vars: list of Vars
      vals: optional string for the values of the var, if known
    Returns:
      a string with the new Vars separated by spaces. Only the vars that
        do not have a "val" are included.
    """
    assert len(orig_vars) == len(vals)
    new_vars = []
    for orig_var, val_pp in zip(orig_vars, vals):
      self.scopes[-1][1].append(orig_var)
      if val_pp is not None:
        if not isinstance(val_pp, str):
          val_pp = str(val_pp)
        self.var_rewrites[orig_var] = val_pp
      else:
        new_var = Var(self.next_var_count, orig_var.suffix)
        self.next_var_count += 1
        self.var_rewrites[orig_var] = str(new_var)
        new_vars.append(str(new_var))

    return " ".join(new_vars)

  def use_var(self, orig_var):
    """A string for a Var or Literal usage."""
    if isinstance(orig_var, Var):
      return self.var_rewrites[orig_var]
    else:
      return str(orig_var)

  def use_vars(self, orig_vars):
    """A list of string for a list of Var usages."""
    return map(self.use_var, orig_vars)

  def main(self, jaxpr, consts):
    """Entry point for pretty printing a jaxpr and the constant values."""
    assert len(jaxpr.constvars) == len(consts)
    self.push_scope()
    # Pre-process the constants and build the legend
    legend_pp = []
    use_constvals = []  # Strings with values
    old_print_options = onp.get_printoptions()
    onp.set_printoptions(threshold=8)
    for const in consts:
      const_size = onp.size(const)
      if self.inline_consts and const_size <= self.inline_const_max_size:
        use_constvals.append(pp(str(const)))
      else:
        # TODO: shorten the printed constant for the legend
        use_constvar = "const_{}".format(next(self.const_counter))
        use_constvals.append(use_constvar)
        legend_pp.append(pp(use_constvar + ' = ') >> pp(str(const)))
    onp.set_printoptions(threshold=old_print_options['threshold'])

    pjaxpr = self.pp_jaxpr(jaxpr, use_constvals, [],
                           [None] * len(jaxpr.invars))
    return self.pop_scope(pjaxpr + vcat(legend_pp))

  def pp_jaxpr(self, jaxpr, constvar_values, freevar_values, invar_values):
    """Pretty print a jaxpr, inlining constvars, freevars and invars
    Args:
      the values are lists of string, or of None
    """
    self.push_scope()
    constvars = self.define_vars(jaxpr.constvars, constvar_values)
    freevars = self.define_vars(jaxpr.freevars, freevar_values)
    invars = self.define_vars(jaxpr.invars, invar_values)

    if self.inline_consts and not constvars and not freevars and not invars:
      lambda_header = pp('{ lambda .')
    else:
      lambda_header = pp('{{ lambda {} ; {} ; {}.'.format(constvars,
                                                          freevars,
                                                          invars))
    eqns_pp = map(self.pp_eqn, jaxpr.eqns)
    use_outvars = hcat(map(pp, self.use_vars(jaxpr.outvars)), sep=pp(' '))
    return self.pop_scope(
      lambda_header +
      ((pp('let ') >> vcat(eqns_pp)) +
       pp('in [') >> use_outvars >> pp('] }')).indent(2))

  def pp_eqn(self, eqn):
    """Pretty print one equation."""
    # Do this first, to reserve the names for the outvars
    lhs = self.define_vars(eqn.outvars, [None] * len(eqn.outvars))
    invars_pp = self.use_vars(eqn.invars)
    preprocessor = getattr(self, 'preprocess_'+eqn.primitive.name, None) if self.sugar_primitives else None
    if preprocessor is not None:
      params = preprocessor(eqn, invars_pp)
      invars_pp = []
      bound_subjaxprs = []
    else:
      params = eqn.params.items()
      bound_subjaxprs = eqn.bound_subjaxprs

    pp_subexpr = pp('')
    if bound_subjaxprs:
      for subjaxpr, const_vars, bound_vars in bound_subjaxprs:
        pp_subexpr = pp_subexpr + (
            self.pp_jaxpr(subjaxpr, const_vars, bound_vars, [None] * len(subjaxpr.invars)).indent(2))
    return (pp('{} = '.format(lhs)) >>
            pp(eqn.primitive.name) >> pp_kv_pairs(params)
            >> pp(' ') >> hcat(map(pp, invars_pp), sep=pp(' '))) + pp_subexpr

  def preprocess_cond(self, eqn, invars_pp):
    true_jaxpr = eqn.params['true_jaxpr']
    false_jaxpr = eqn.params['false_jaxpr']
    pred, true_consts_ops, false_consts_ops = (
      split_list(invars_pp, [1, len(true_jaxpr.in_avals)])
    )
    return [
      ('pred', pp(pred[0])),
      ('true_jaxpr',
       self.pp_jaxpr(true_jaxpr.jaxpr, true_jaxpr.literals, [],
                     true_consts_ops)),
      ('false_jaxpr',
       self.pp_jaxpr(false_jaxpr.jaxpr, false_jaxpr.literals, [],
                     false_consts_ops))
    ]

  def preprocess_while(self, eqn, invars_pp):
    cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts = (
      split_dict(eqn.params,
                 ['cond_jaxpr', 'cond_nconsts', 'body_jaxpr', 'body_nconsts']))
    cond_consts, body_consts, carry_init = (
      split_list(invars_pp, [cond_nconsts, body_nconsts])
    )
    return [
      ('cond_jaxpr',
       self.pp_jaxpr(cond_jaxpr.jaxpr, self.use_vars(cond_jaxpr.literals), [],
                     cond_consts + [None] * len(carry_init))),
      ('body_jaxpr',
       self.pp_jaxpr(body_jaxpr.jaxpr, self.use_vars(body_jaxpr.literals), [],
                     body_consts + [None] * len(carry_init))),
      ('init_carry',
       hcat(map(pp, carry_init), sep=pp(' ')))
    ]

  def preprocess_xla_call(self, eqn, invars_pp):
    assert len(eqn.bound_subjaxprs) == 1
    subjaxpr, consts, freevals = eqn.bound_subjaxprs[0]
    assert len(subjaxpr.invars) == len(invars_pp)
    return [
      ('device', eqn.params['device']),
      ('backend', eqn.params['backend']),
      ('body_jaxpr',
       self.pp_jaxpr(subjaxpr, self.use_vars(consts),
                     self.use_vars(freevals), invars_pp))
    ]

  def preprocess_scan(self, eqn, invars_repl):
    assert len(eqn.bound_subjaxprs) == 0
    assert len(eqn.params['linear']) == len(invars_repl)

    jaxpr = eqn.params['jaxpr']
    body_consts, carry_init, inputs = (
      split_list(invars_repl, [eqn.params['num_consts'], eqn.params['num_carry']])
    )
    const_linear, carry_linear, input_linear = (
      split_list(eqn.params['linear'], [eqn.params['num_consts'], eqn.params['num_carry']])
    )
    # Drop from the inputs the '*'
    filtered_inputs = []
    filtered_inputs_linear = []
    inputs_repl = []
    for i, il in zip(inputs, input_linear):
      if i != '*':
        filtered_inputs.append(i)
        filtered_inputs_linear.append(il)
        inputs_repl.append(None)
      else:
        inputs_repl.append('')

    return [
      ('forward', eqn.params['forward']),
      ('length', eqn.params['length']),
      ('num_carry', eqn.params['num_carry']),
      ('carry_init', hcat(map(pp, carry_init), sep=pp(' '))),
      ('inputs', hcat(map(pp, filtered_inputs), sep=pp(' '))),
      ('carry_linear', carry_linear),
      ('inputs_linear', filtered_inputs_linear),
      ('body_jaxpr',
       self.pp_jaxpr(jaxpr.jaxpr, self.use_vars(jaxpr.literals),
                     [], body_consts + [None] * len(carry_init) + inputs_repl)),

    ]