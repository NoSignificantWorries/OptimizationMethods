
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def brute_force_method(func, a, b, eps):
    x_star = a
    x_bar = a
    f_star = func(a)

    call_count = 1
    while x_bar <= b:
        x_bar = x_bar + eps
        f = func(x_bar)
        call_count += 1
        
        if f < f_star:
            x_star = x_bar
            f_star = f
    
    return x_star, f_star, call_count


def dihotomy(func, a, b, eps, delta=None):
    if delta is None:
        delta = eps / 4.0
    if delta <= 0 or delta >= eps / 2.0:
        raise ValueError("delta must satisfy 0 < delta < eps/2")

    alpha = a
    beta = b
    call_count = 0

    while (beta - alpha) > 2.0 * eps:
        mid = (alpha + beta) / 2.0
        x1 = mid - delta
        x2 = mid + delta
        y1 = func(x1)
        y2 = func(x2)
        call_count += 2

        if y1 < y2:
            beta = x2
        elif y1 > y2:
            alpha = x1
        else:
            alpha = x1
            beta = x2

    x_star = (alpha + beta) / 2.0
    f_star = func(x_star)
    call_count += 1
    return x_star, f_star, call_count


x = sp.symbols('x')

# str_expr = "exp(x) + x ** 2"
# str_expr = "sin(x) + 0.1 * x"
str_expr = "-x"
# a, b = -1, 1
a, b = 0, 5
eps = 10 ** (-3)
delta = 10 ** (-7)
samples = round((b - a) / eps)

expr = sp.sympify(str_expr)
f = sp.lambdify(x, expr, modules=['numpy', 'math'])


X = np.linspace(a, b, samples)
Y = f(X)

bfX, bfY, bf_cnt = brute_force_method(f, a, b, eps)
print(f"Brute Force Method Result: f_min = f({bfX}) = {bfY}, count of calculations = {bf_cnt}")

dhX, dhY, dh_cnt = dihotomy(f, a, b, eps, delta)
print(f"Dihotomy Method Result: f_min = f({dhX}) = {dhY}, count of calculations = {dh_cnt}")

plt.plot(X, Y, color="green")
plt.axvline(bfX, color="red", linestyle="--", linewidth=2, label="Brute force")
plt.axhline(bfY, color="red", linestyle="--", linewidth=2, label="Brute force")
plt.axvline(dhX, color="blue", linestyle="--", linewidth=1, label="Dihotomy")
plt.axhline(dhY, color="blue", linestyle="--", linewidth=1, label="Dihotomy")

plt.savefig("res.png", dpi=300)
