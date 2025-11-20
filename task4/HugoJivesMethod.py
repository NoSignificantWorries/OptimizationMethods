import numpy as np
import plotly.graph_objects as go


def AC(func, vec, delta, basis):
    j = 0
    vec_bar = vec.copy()
    f_bar = func(*vec_bar)
    cnt = 1

    n = len(basis)
    while j < n:

        f1 = func(*(vec_bar + delta[j] * basis[j]))
        cnt += 1
        if f_bar > f1:
            vec_bar += delta[j] * basis[j]
            f_bar = f1
            j += 1
            continue

        f2 = func(*(vec_bar - delta[j] * basis[j]))
        cnt += 1
        if f_bar > f2:
            vec_bar -= delta[j] * basis[j]
            f_bar = f2

        j += 1
    
    return vec_bar, func(*vec_bar), cnt


def hugo_jives_method_optimized(vec0, func, delta, gamma, basis, eps, max_iter=1000):
    points = []
    cnt = 0
    x = vec0.copy()
    
    for _ in range(max_iter):
        current_f = func(*x)
        points.append((x.copy(), current_f))
        
        x_new, f_new, cnt_ac = AC(func, x, delta, basis)
        cnt += cnt_ac
        
        if f_new < current_f - eps:
            x_pattern = x_new + (x_new - x)
            cnt += 1
            
            x_pattern_new, f_pattern_new, cnt_ac2 = AC(func, x_pattern, delta, basis)
            cnt += cnt_ac2
            
            if f_pattern_new < f_new:
                x = x_pattern_new
                points.append((x.copy(), f_pattern_new))
            else:
                x = x_new
                points.append((x.copy(), f_new))
        else:
            delta = delta / gamma
            if np.linalg.norm(delta) < eps:
                break
    
    final_f = func(*x)
    points.append((x.copy(), final_f))
    cnt += 1
    
    return points, x, final_f, cnt


if __name__ == "__main__":
    n = 2
    x1, x2 = -1.2, 1.2
    y1, y2 = -1.2, 1.2
    eps = 10 ** (-3)
    # eps = 10 ** (-4)
    delta = np.array([1.0, 1.0], dtype=np.float64)
    basis = np.eye(n, dtype=np.float64)
    gamma = 2

    samplesX = 150
    samplesY = 150

    def f(x, y):
        return x ** 2 + 3 * y ** 2 + np.exp(x) + np.exp(y)
    
    X = np.linspace(x1, x2, samplesX)
    Y = np.linspace(y1, y2, samplesY)
    xgrid, ygrid = np.meshgrid(X, Y)
    zgrid = f(xgrid, ygrid)

    X0 = np.array([1.0, 1.0])
    points, min_point, min_val, count = hugo_jives_method_optimized(X0, f, delta, gamma, basis, eps)
    
    print(f"Hugo-Jives Method Result: f_min = f({min_point}) = {min_val}, count of calculations = {count}")

    fig = go.Figure()
    
    x_points = []
    y_points = []
    z_points = []
    for i, (xy, z) in enumerate(points):
        x_points.append(xy[0])
        y_points.append(xy[1])
        z_points.append(z)
    

    fig.add_trace(go.Surface(
        x=xgrid,
        y=ygrid,
        z=zgrid,
        colorscale='Plasma',
        opacity=0.9,
        name='Function',
        showscale=True,
        colorbar=dict(title="f(x,y)")
    ))

    fig.add_trace(go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers+text+lines',
        marker=dict(
            size=5,
            color='#00ff00',
            symbol='circle',
            line=dict(
                color='black',
                width=2
            )
        ),
        line=dict(
            color='red',
            width=8,
            dash='dash'
        ),
        text=[f'{i}' for i in range(len(x_points))],
        textposition="top center",
        name='Traectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=[min_point[0]], y=[min_point[1]], z=[min_val],
        mode='markers+text',
        marker=dict(
            size=7,
            color='blue',
            line=dict(color='black', width=2)
        ),
        text=[f'Min<br>({min_point[0]:.4f}, {min_point[1]:.4f})<br>f = {min_val:.6f}'],
        textposition="top center",
        name='Minimum'
    ))

    fig.update_layout(
        title={
            'text': "Hugo-Jives function",
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(x,y)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1920,
        height=1080,
        font=dict(size=18)
    )

    fig.write_html("task4/surface_plot_interactive.html")
    # fig.write_image("task4/surface_plot.png", width=1200, height=800, scale=2)
    
    # fig.show()

