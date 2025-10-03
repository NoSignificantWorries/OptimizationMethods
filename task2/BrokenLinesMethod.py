import heapq

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


N_POINTS = 1000


def find_L_analitic(func, var, a, b, n_samples=10000):
    # Robust global estimate of sup |f'(x)| over [a,b] via dense sampling
    f_prime = sp.lambdify(var, sp.diff(func, var), modules=['numpy', 'math'])
    xs = np.linspace(a, b, n_samples)
    vals = np.abs(f_prime(xs))
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else 0.0


def find_L_vectorized(func, a, b, n_points=N_POINTS):
    x = np.linspace(a, b, n_points)
    f_x = func(x)
    
    x_i, x_j = np.meshgrid(x, x, indexing='ij')
    f_i, f_j = np.meshgrid(f_x, f_x, indexing='ij')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.abs(f_i - f_j) / np.abs(x_i - x_j)
    
    ratios = ratios[np.isfinite(ratios)]
    
    return np.max(ratios) if len(ratios) > 0 else 0


def find_L(expr, var, func, a, b, n_points=N_POINTS):

    # Compute a safe (over-)estimate for Lipschitz constant
    try:
        L = find_L_analitic(expr, var, a, b, n_samples=max(2000, 10 * n_points))
    except Exception:
        L = find_L_vectorized(func, a, b, n_points)

    return L


def broken_lines_method(func, a, b, eps, L):
    f_a, f_b = func(a), func(b)
    
    x_star, f_star = (a, f_a) if f_a < f_b else (b, f_b)
    
    heap = []
    
    def compute_vertex(x_i, x_j, f_i, f_j):
        z = 0.5 * (x_i + x_j) - (f_j - f_i) / (2 * L)
        z = np.clip(z, x_i, x_j)
        phi = 0.5 * (f_i + f_j) - 0.5 * L * (x_j - x_i)
        
        return z, phi
    
    z, phi = compute_vertex(a, b, f_a, f_b)
    heapq.heappush(heap, (phi, a, b, f_a, f_b, z))

    pointsX = [a, b]
    pointsY = [f_a, f_b]
    lower_bounds = [phi]
    
    call_times = 2
    
    while True:
        if not heap:
            break
        
        phi_min, x_i, x_j, f_i, f_j, z = heapq.heappop(heap)
        
        if (f_star - phi_min) < eps:
            break
        
        f_z = func(z)
        
        call_times += 1

        if f_z < f_star:
            x_star, f_star = z, f_z
        
        z_left, phi_left = compute_vertex(x_i, z, f_i, f_z)
        heapq.heappush(heap, (phi_left, x_i, z, f_i, f_z, z_left))
        
        z_right, phi_right = compute_vertex(z, x_j, f_z, f_j)
        heapq.heappush(heap, (phi_right, z, x_j, f_z, f_j, z_right))
        
        pointsX.append(z)
        pointsY.append(f_z)
        lower_bounds.append(min(phi_left, phi_right))

    return pointsX, pointsY, x_star, f_star, call_times


if __name__ == "__main__":
    x = sp.symbols('x')

    # str_expr = "exp(x) + x ** 2"
    str_expr = "cos(x)+(10*x-x^2)/50"
    a, b = 0, 10
    # eps = 10 ** (-3)
    eps = 10 ** (-4)
    samples = round((b - a) / eps)

    expr = sp.sympify(str_expr)
    f = sp.lambdify(x, expr, modules=['numpy', 'math'])
    
    L = find_L(expr, x, f, a, b)
    print(L)

    X = np.linspace(a, b, samples)
    Y = f(X)

    pX, pY, chX, chY, ch_cnt = broken_lines_method(f, a, b, eps, L)
    print(f"Brocken Lines Method Result: f_min = f({chX}) = {chY}, count of calculations = {ch_cnt}")

    plt.plot(X, Y, color="green")
    plt.plot(pX, pY, "bo")
    plt.axvline(chX, color="red", linestyle="--", linewidth=2)
    plt.axhline(chY, color="red", linestyle="--", linewidth=2)

    plt.savefig("task2/broken_lines_method.png", dpi=300)


