from sympy.printing.latex import latex
import re


def md_math(expr):
    return '$$\n' + latex(expr) + '\n$$\n\n'


# def replace_symbol(file: str):
#     for lines in file:

