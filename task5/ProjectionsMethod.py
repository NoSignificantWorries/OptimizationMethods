import numpy as np
import plotly.graph_objects as go


def visualize_3d_trajectory_plotly(history):
    trajectory = [h[0] for h in history]
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    zs = [p[2] for p in trajectory]
    
    fig = go.Figure()

    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    

    fig.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale='Plasma',
        opacity=0.5,
        name='Sphere',
        showscale=True
    ))

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
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
        text=[f'{i}' for i in range(len(xs))],
        textposition="top center",
        name='Traectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
        mode='markers+text',
        marker=dict(
            size=7,
            color='blue',
            line=dict(color='black', width=2)
        ),
        text=[f'Min<br>({xs[-1]:.4f}, {ys[-1]:.4f})<br>f = {zs[-1]:.6f}'],
        textposition="top center",
        name='Minimum'
    ))

    fig.update_layout(
        title={
            'text': "Projections method",
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

    return fig

def projected_gradient_descent(f, grad_f, proj, x0, learning_rate=0.1, 
                              max_iter=1000, tol=1e-6, verbose=False):
    
    x = np.array(x0, dtype=float)
    history = []
    
    alpha = learning_rate
    for i in range(max_iter):
        f_val = f(x)
        history.append((x.copy(), f_val))
        
        gradient = grad_f(x)

        f_new = f(x - alpha * gradient)
        while f_new >= f_val:
            alpha = alpha / 2
            f_new = f(x - alpha * gradient)
        
        x_new = x - alpha * gradient
        
        x_new = proj(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            if verbose:
                print(f"Сходимость достигнута на итерации {i+1}")
            break
        
        x = x_new
        
        if verbose and (i+1) % 100 == 0:
            print(f"Итерация {i+1}: f(x) = {f_val:.6f}, x = {x}")
    
    f_val = f(x)
    history.append((x.copy(), f_val))
    
    if verbose and i == max_iter - 1:
        print(f"Достигнут максимум итераций ({max_iter})")
    
    return x, f_val, history


if __name__ == "__main__":
    # 4^x + 3^y + 2^z -> min
    # x^2 / 4 + y^2 / 9 + z^2 / 16 <= 1
    # x0 = (1, 1, 1)

    # x = 2x_
    # y = 3y_
    # z = 4z_

    # 16^x_ + 27^y_ + 16^z_
    # x_^2 + y_^2 + z_^2 <= 1
    # x0_ = (1/2, 1/3, 1/4)

    def f_origin(x):
        return 4 ** x[0] + 3 ** x[1] + 2 ** x[2]

    def f(x):
        return 16 ** x[0] + 27 ** x[1] + 16 ** x[2]
    
    def grad_f(x):
        return np.array([
            16 ** x[0] * np.log(16),
            27 ** x[1] * np.log(27),
            16 ** x[2] * np.log(16)
        ])
    
    def proj_unit_circle(x):
        norm = np.linalg.norm(x)
        if norm <= 1:
            return x
        return x / norm
    
    def in_unit_circle(x):
        return np.linalg.norm(x) <= 1
    
    x0 = np.array([1/2, 1/3, 1/4])
    
    x_opt_, f_opt_, history = projected_gradient_descent(
        f, grad_f, proj_unit_circle, x0, 
        learning_rate=0.1,
        max_iter=1000,
        tol=1e-3,
        verbose=True
    )

    x_opt = x_opt_ * np.array([2, 3, 4])
    f_opt = f_origin(x_opt)

    
    print("\nРезультаты оптимизации:")
    print(f"Оптимальная точка: x_ = [{x_opt_[0]:.6f}, {x_opt_[1]:.6f}, {x_opt_[2]:.6f}]")
    print(f"Значение функции: f_(x_) = {f_opt_:.6f}")
    print(f"Оптимальная точка: x = [{x_opt[0]:.6f}, {x_opt[1]:.6f}, {x_opt[2]:.6f}]")
    print(f"Значение функции: f(x) = {f_opt:.6f}")
    print(f"Норма точки: ||x|| = {np.linalg.norm(x_opt):.6f}")
    print(f"Количество итераций: {len(history)-1}")

    fig = visualize_3d_trajectory_plotly(history)
    
    fig.write_html("proj.html")
    
