
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def golden_ratio(func, a, b, eps):
    a_in, b_in = a, b
    f_a, f_b = func(a), func(b)

    x_1 = b - (np.sqrt(5) - 1) / 2 * (b - a)
    x_2 = a + b - x_1

    f_1, f_2 = func(x_1), func(x_2)
    
    pointsX = [a, b]
    pointsY = [f_a, f_b]
    
    call_times = 4
    while (b - a) > eps:
        pointsX += [x_1, x_2]
        pointsY += [f_1, f_2]
        if f_1 < f_2:
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + b - x_2
            f_1 = func(x_1)
        else:
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + b - x_1
            f_2 = func(x_2)
        call_times += 1

    if f_1 < f_2:
        f_star = f_1
        x_star = x_1
    else:
        f_star = f_2
        x_star = x_2
    
    if f_a < f_star:
        f_star = f_a
        x_star = a_in
    elif f_b < f_star:
        f_star = f_b
        x_star = b_in

    return pointsX, pointsY, x_star, f_star, call_times


if __name__ == "__main__":
    x = sp.symbols('x')

    str_expr = "exp(x) + x ** 2"
    # str_expr = "sin(x) + 0.1 * x"
    a, b = -1, 1
    # a, b = 0, 5
    eps = 10 ** (-3)
    delta = 10 ** (-7)
    samples = round((b - a) / eps)

    expr = sp.sympify(str_expr)
    f = sp.lambdify(x, expr, modules=['numpy', 'math'])


    X = np.linspace(a, b, samples)
    Y = f(X)

    pX, pY, grX, grY, gr_cnt = golden_ratio(f, a, b, eps)
    print(f"Golden Ratio Method Result: f_min = f({grX}) = {grY}, count of calculations = {gr_cnt}")

    plt.plot(X, Y, color="green")
    plt.plot(pX, pY, "bo")
    plt.axvline(grX, color="red", linestyle="--", linewidth=2)
    plt.axhline(grY, color="red", linestyle="--", linewidth=2)

    plt.savefig("task2/golden_ratio.png", dpi=300)

