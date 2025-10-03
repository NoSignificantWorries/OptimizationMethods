import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def chords_method(f, f_prime, a, b, eps1, eps2):
    fp_a = f_prime(a)
    if fp_a >= 0:
        f_a = f(a)
        return [a], [f_a], [fp_a], a, f_a, 1, 1
    
    fp_b = f_prime(b)
    if fp_b <= 0:
        f_b = f(b)
        return [b], [f_b], [fp_b], b, f_b, 1, 1

    pointsX = []
    pointsY = []
    pointsY_prime = []
    
    x_star = float("inf")
    f_star = float("inf")
    
    counts = 0
    counts_prime = 2
    while (b - a) >= eps2:
        c = a - (b - a) * fp_a / (fp_b - fp_a)
        fp_c = f_prime(c)
        counts_prime += 1
        pointsX.append(c)
        pointsY.append(f(c))
        pointsY_prime.append(fp_c)

        if np.abs(fp_c) <= eps1:
            x_star = c
            f_star = f(x_star)
            counts += 1
            break
        
        if fp_c < 0:
            a = c
            fp_a = fp_c
        else:
            b = c
            fp_b = fp_c
    
    return pointsX, pointsY, pointsY_prime, x_star, f_star, counts, counts_prime


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
    f_prime = sp.lambdify(x, sp.diff(expr, x), modules=['numpy', 'math'])
    
    X = np.linspace(a, b, samples)
    Y = f(X)
    Y_prime = f_prime(X)

    pX, pY, ppY, chX, chY, ch_cnt, chp_cnt = chords_method(f, f_prime, a, b, eps1, eps2)
    print(f"Chord Method Result: f_min = f({chX}) = {chY}, count of calculations = {ch_cnt}, count of calculations (prime) = {chp_cnt}")

    plt.plot(X, Y, color="green", label="func")
    plt.plot(X, Y_prime, color="blue", label="func'")
    plt.plot(pX, pY, "go")
    plt.plot(pX, ppY, "bo")
    plt.axvline(chX, color="red", linestyle="--", linewidth=2, label="x*")
    plt.axhline(chY, color="red", linestyle="--", linewidth=2, label="f*")

    plt.legend()
    plt.savefig("task3/chord_method.png", dpi=300)
