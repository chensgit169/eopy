import sympy
from sympy import cos, pi, exp, sqrt, diff, simplify, ceiling, series, fraction
from sympy.abc import x, y, z
from sympy.printing.latex import latex
from sympy.matrices import diag, det
from numpy import inf


def md_math(expr):
    return '$$\n' + latex(expr) + '\n$$\n\n'


Ax = sympy.Function('Ax')(x, y, z)
Ay = sympy.Function('Ay')(x, y, z)
Az = sympy.Function('Az')(x, y, z)
g = sympy.Function('g')(x, y, z)
term = g.series(x, n=None)
# print([next(term) for i in range(2)])

x = sympy.Function('x')(z)
y = sympy.Function('y')(z)
t = sympy.Function('t')(z)
xp = diff(x, z)
yp = diff(y, z)

print(g)
f = g
print(f)
g = g.xreplace({x: 0})
print(g)
L = sqrt(1+xp**2+yp**2) * g + Ax*xp + Ay*yp + Az
# print(L)
eq_x = fraction(simplify(diff(L, xp, z) - diff(L, x)))[0]
eq_y = fraction(simplify(diff(L, yp, z) - diff(L, y)))[0]


g_approx = g.subs([(x, 0), (y, 0), (z, 0)])
print(g_approx)
eq_x = eq_x.subs(g, g_approx)


# with open('expression_latex.md', 'w') as f:
#     f.write(md_math(eq_x))
#     f.write(md_math(simplify(eq_x)))



