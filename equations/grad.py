from sympy.vector import CoordSys3D, divergence, curl, gradient


r = CoordSys3D('r')
f = r.x * r.y * r.z

grad = gradient(f)
print(grad.i)
print(type(grad))
