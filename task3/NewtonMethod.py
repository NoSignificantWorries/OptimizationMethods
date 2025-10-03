import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def newton_method(f, f_prime, f_prime2, a, b, eps1, eps2):
    x_0 = (a + b) / 2

    pointsX = []
    pointsY = []
    pointsY_prime = []
    pointsY_prime2 = []

    counts = 0
    counts_prime = 0
    counts_prime2 = 0
    
    while True:
        y_0 = f_prime(x_0)
        counts_prime += 1
        
        pointsX.append(x_0)
        pointsY.append(f(x_0))
        pointsY_prime.append(f_prime(x_0))
        pointsY_prime2.append(f_prime2(x_0))

        if y_0 <= eps1:
            x_star = x_0
            f_star = f(x_0)
            counts += 1
            break

        x_0 -= y_0 / f_prime2(x_0)
        counts_prime2 += 1
    
    return pointsX, pointsY, pointsY_prime, pointsY_prime2, x_star, f_star, counts, counts_prime, counts_prime2


if __name__ == "__main__":
    x = sp.symbols('x')

    str_expr = "exp(x) + x ** 2"
    # str_expr = "cos(x)+(10*x-x^2)/50"
    a, b = -1, 1
    # eps1 = 10 ** (-3)
    eps1 = 10 ** (-4)
    eps2 = 2 * eps1
    samples = round((b - a) / eps1)

    expr = sp.sympify(str_expr)
    f = sp.lambdify(x, expr, modules=['numpy', 'math'])
    f_prime_expr = sp.diff(expr, x)
    f_prime = sp.lambdify(x, f_prime_expr, modules=['numpy', 'math'])
    f_prime2 = sp.lambdify(x, sp.diff(f_prime_expr, x), modules=['numpy', 'math'])
    
    X = np.linspace(a, b, samples)
    Y = f(X)
    Y_prime = f_prime(X)
    Y_prime2 = f_prime2(X)

    pX, pY, ppY, pp2Y, chX, chY, ch_cnt, chp_cnt, chp2_cnt = newton_method(f, f_prime, f_prime2, a, b, eps1, eps2)
    print(f"Newton's Method Result: f_min = f({chX}) = {chY}\ncount of calculations = {ch_cnt}\ncount of calculations (prime) = {chp_cnt}\ncount of calculations (prime 2) = {chp2_cnt}")

    plt.plot(X, Y, color="green", label="func")
    plt.plot(X, Y_prime, color="blue", label="func'")
    plt.plot(X, Y_prime2, color="red", label="func''")
    plt.plot(pX, pY, "go")
    plt.plot(pX, ppY, "bo")
    plt.plot(pX, pp2Y, "ro")
    plt.axvline(chX, color="red", linestyle="--", linewidth=2, label="x*")
    plt.axhline(chY, color="red", linestyle="--", linewidth=2, label="f*")

    plt.legend()
    plt.savefig("task3/newton_method.png", dpi=300)
