import sympy as sp

# 定义符号变量
x, y, z, t = sp.symbols('x y z t')
E_x, E_y, E_z = sp.symbols('E_x E_y E_z', function=True)
B_x, B_y, B_z = sp.symbols('B_x B_y B_z', function=True)
rho = sp.Symbol('rho', real=True)
J_x, J_y, J_z = sp.symbols('J_x J_y J_z', function=True)

# 定义Maxwell方程组
gauss_eq = sp.Eq(sp.divergence(sp.Matrix([E_x(x, y, z, t), E_y(x, y, z, t), E_z(x, y, z, t)])), rho)
ampere_eq = sp.Eq(sp.divergence(sp.Matrix([B_x(x, y, z, t), B_y(x, y, z, t), B_z(x, y, z, t)])), 0)
faraday_eq1 = sp.Eq(sp.curl(sp.Matrix([E_x(x, y, z, t), E_y(x, y, z, t), E_z(x, y, z, t)])), -sp.diff(sp.Matrix([B_x(x, y, z, t), B_y(x, y, z, t), B_z(x, y, z, t)]), t))
faraday_eq2 = sp.Eq(sp.curl(sp.Matrix([B_x(x, y, z, t), B_y(x, y, z, t), B_z(x, y, z, t)])), sp.Matrix([J_x(x, y, z, t), J_y(x, y, z, t), J_z(x, y, z, t)]))

# 将表达式转换为Markdown格式的LaTeX字符串
gauss_eq_latex = sp.latex(gauss_eq)
ampere_eq_latex = sp.latex(ampere_eq)
faraday_eq1_latex = sp.latex(faraday_eq1)
faraday_eq2_latex = sp.latex(faraday_eq2)

# 输出Markdown格式
markdown_content = f"""
Maxwell's Equations:

1. Gauss's Law:
   {gauss_eq_latex}

2. Gauss's Law for Magnetism:
   {ampere_eq_latex}

3. Faraday's Law of Induction:
   {faraday_eq1_latex}

4. Ampère's Law with Maxwell's Addition:
   {faraday_eq2_latex}

Where:
- \(\mathbf{{E}} = (E_x, E_y, E_z)\) is the electric field vector.
- \(\mathbf{{B}} = (B_x, B_y, B_z)\) is the magnetic field vector.
- \(\rho\) is the charge density.
- \(\mathbf{{J}} = (J_x, J_y, J_z)\) is the current density.

Note: The equations are given in natural units, where \(\epsilon_0 = \mu_0 = c = 1\).
"""

# 输出Markdown内容
print(markdown_content)
