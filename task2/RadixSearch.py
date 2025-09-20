
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def radix_search(func, a, b, eps):
    x_star = a
    f_star = func(a)
    x = a

    pointsX = [a]
    pointsY = [f_star]

    call_times = 1    
    while (x + eps) <= b:
        x += eps
        
        now_f = func(x)
        call_times += 1
        
        pointsX.append(x)
        pointsY.append(now_f)

        if now_f > f_star:
            break
        else:
            x_star = x
            f_star = now_f

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

    pX, pY, grX, grY, gr_cnt = radix_search(f, a, b, eps)
    print(f"Radix Search Method Result: f_min = f({grX}) = {grY}, count of calculations = {gr_cnt}")

    plt.plot(X, Y, color="green")
    plt.plot(pX, pY, "bo")
    plt.axvline(grX, color="red", linestyle="--", linewidth=2)
    plt.axhline(grY, color="red", linestyle="--", linewidth=2)

    plt.savefig("task2/raadix_search.png", dpi=300)

