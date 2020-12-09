import os
import sys

import torch
import inspect
from functools import reduce, wraps
from collections.abc import Iterable


def is_iterable(x):
    return isinstance(x, Iterable)


def tensor_repr(t, assert_all=False):
    exception_encountered = False
    info = []
    shape = tuple(t.shape)
    if shape == () or shape == (1,):
        info.append(f"[{t.item():.4f}]")
    else:
        info.append(f"({', '.join(map(repr, shape))})")
    invalid_sum = (~t.isfinite()).sum().item()
    if invalid_sum:
        info.append(f"{invalid_sum} INVALID ENTRIES")
        exception_encountered = True
    if t.requires_grad:
        info.append("req_grad")
    if t.is_leaf and t.grad is not None:
        grad_invalid_sum = (~t.grad.isfinite()).sum().item()
        if grad_invalid_sum:
            info.append(
                f"GRAD {(~ t.grad.isfinite()).sum().item()} INVALID ENTRIES")
            exception_encountered = True
    if hasattr(debug, 'verbose') and debug.verbose:
        info.append(f"|x|={t.float().norm():.1f}")
        if t.numel():
            info.append(f"x in [{t.min():.1f}, {t.max():.1f}]")
        if t.is_leaf and t.grad is not None:
            info.append(f"|grad|={t.grad.float().norm()}")
    if t.dtype != torch.float:
        info.append(f"dtype={str(t.dtype).split('.')[-1]}")
    if t.device.type != 'cpu':
        info.append(f"device={t.device.type}")
    output = f"tensor({', '.join(info)})"
    if assert_all:
        assert_val = t.all()
        if not assert_val:
            exception_encountered = True
    if exception_encountered and (not hasattr(debug, 'raise_exception') or debug.raise_exception):
        if debug.restore_defaults_on_exception:
            debug.raise_exception = False
            debug.silent = False
        debug.x = t
        stack = output + ('\nSTACK:' + debug._stack +
                          output) if debug._stack else ''
        if debug._indent:
            debug.args = debug._last_args
            debug.func = debug._last_call

            @wraps(debug.func)
            def _recall(*args, **kwargs):
                call_args = {**debug.args, **kwargs,
                             **dict(zip(debug._last_args_sig, args))}
                return debug(debug.func)(**call_args)

            def print_stack():
                print(stack)
            debug.stack = print_stack

            debug.recall = _recall
        debug._indent = 0
        if assert_all and not assert_val:
            assert False, "Assert did not pass on " + stack
        assert False, "Invalid entries encountered in " + stack
    return output


def _debug_log(output, var=None, indent='', assert_true=False):
    debug._stack += indent + output
    if not debug.silent:
        print(indent + output, end='')
    if var is not None:
        if isinstance(var, str):
            _debug_log(f"'{var}'")
        elif isinstance(var, torch.Tensor):
            _debug_log(tensor_repr(var, assert_true))
        elif is_iterable(var):
            if debug.expand:
                _debug_log(f"{type(var).__name__} {{")
                if isinstance(var, dict):
                    for k, v in var.items():
                        _debug_log(f"'{k}': ", v, indent + 6 * ' ',
                                   assert_true)
                else:
                    for e in var:
                        _debug_log('- ', e, indent + 6 * ' ',
                                   assert_true)
                _debug_log(indent + 4 * ' ' + '}')
            else:
                _debug_log(f"{type(var).__name__}[{len(list(var))}]")
        else:
            _debug_log(str(var))
    else:
        debug._stack += '\n'
        if not debug.silent:
            print()


def debug(arg, assert_true=False):

    if not hasattr(arg, '__call__'):
        if debug._indent == 0:
            debug._stack = ""
        line = ''.join(inspect.stack()[1][4])
        argname = ')'.join('('.join(line.split('(')[1:]).split(')')[:-1])
        _debug_log(f"{{{argname}}}  =  ", arg, ' ' * 4 *
                   debug._indent, assert_true)
        return

    func = arg
    sig_parameters = inspect.signature(func).parameters
    sig_argnames = [p.name for p in sig_parameters.values()]
    sig_defaults = {
        k: v.default
        for k, v in sig_parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    @wraps(func)
    def _func(*args, **kwargs):
        if debug._indent == 0:
            debug._stack = ""
        indent = ' ' * 4 * debug._indent
        debug._indent += 1

        _debug_log('\n')
        _debug_log(f"@{func.__name__}()", indent=indent)

        args_kw = dict(zip(sig_argnames, args))
        defaults = {k: v for k, v in sig_defaults.items()
                    if k not in kwargs
                    if k not in args_kw}
        debug._last_call = func
        debug._last_args = {**args_kw, **defaults}
        debug._last_args_sig = sig_argnames

        for argtype, params in [("args", args_kw.items()),
                                ("kwargs", kwargs.items()),
                                ("defaults", defaults.items())]:
            if params:
                _debug_log(f"{argtype}:", indent=indent + ' ' * 6)
            for argname, arg in params:
                _debug_log(f"- {argname}:  ", arg,
                           indent + ' ' * 8, assert_true)
        out = func(*args, **kwargs)
        debug.out = out
        debug._indent -= 1
        if out is not None:
            _debug_log("returned:  ", out,
                       indent, assert_true)
        return out
    return _func


def debug_init():
    debug._stack = ""
    debug._indent = 0
    debug.verbose = True
    debug.silent = False
    debug.expand = True
    debug.raise_exception = True
    interactive_notebook = 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ
    debug.restore_defaults_on_exception = not interactive_notebook


debug_init()
