#######################################################################################################################
#######################################################################################################################
# Title:        Thermal Model Order Reduction and Simulation (TherMOS)
# Topic:        Power Electronics, Model Order Reduction
# File:         calcEq
# Date:         07.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function evaluates a user equation (e.g. the losses of the PINN residual, '3*Rs*x4**2') on arrays or PyTorch
tensors. The equation is parsed and checked before it is evaluated: only numbers, the given variable and constant
names, the operators + - * / ** % and the functions abs, sqrt, exp, log, sin, cos, tan, tanh, minimum, maximum and
where are allowed (no attributes, indexing, keywords or other names), so no arbitrary code can be executed. With
tensors the PyTorch functions are used, so gradients flow through the equation.
Inputs:     1) expr:    equation (string)
            2) var:     dictionary of variables {name: array or tensor}
            3) const:   dictionary of constants {name: float}
Outputs:    1) out:     value of the equation (array or tensor, broadcast of the variables)
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import ast
import numpy as np

#######################################################################################################################
# Settings
#######################################################################################################################
FNC = ('abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'tanh', 'minimum', 'maximum', 'where')
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub,
          ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE)


#######################################################################################################################
# Additional Functions
#######################################################################################################################
def _check(expr, names):
    """Parsed and checked expression; raises ValueError with a readable message."""
    try:
        tree = ast.parse(str(expr).strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError("syntax error in '%s': %s" % (expr, e.msg))
    for n in ast.walk(tree):
        if not isinstance(n, _NODES):
            raise ValueError("'%s' is not allowed in '%s'" % (type(n).__name__, expr))
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError("only numbers are allowed as constants in '%s'" % expr)
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in FNC or n.keywords):
            raise ValueError("unknown function in '%s' (allowed: %s)" % (expr, ', '.join(FNC)))
        if isinstance(n, ast.Name) and n.id not in names and n.id not in FNC:
            raise ValueError("unknown name '%s' in '%s' (known: %s)" % (n.id, expr, ', '.join(sorted(names))))
    return compile(tree, '<equation>', 'eval')


def _funcs(var):
    """Functions of the namespace: PyTorch for tensors, NumPy otherwise."""
    if any(type(v).__module__.startswith('torch') for v in var.values()):
        import torch
        return {'abs': torch.abs, 'sqrt': torch.sqrt, 'exp': torch.exp, 'log': torch.log, 'sin': torch.sin,
                'cos': torch.cos, 'tan': torch.tan, 'tanh': torch.tanh,
                'minimum': lambda a, b: torch.minimum(torch.as_tensor(a), torch.as_tensor(b)),
                'maximum': lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)),
                'where': torch.where}
    return {'abs': np.abs, 'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
            'tan': np.tan, 'tanh': np.tanh, 'minimum': np.minimum, 'maximum': np.maximum, 'where': np.where}


#######################################################################################################################
# Main Function
#######################################################################################################################
def calcEq(expr, var, const=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    const = {k: float(v) for k, v in (const or {}).items()}
    names = set(var) | set(const)

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    code = _check(expr, names)
    ns = {'__builtins__': {}, **_funcs(var), **const, **var}
    out = eval(code, ns)

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return out


#######################################################################################################################
# References
#######################################################################################################################
