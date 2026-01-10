def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x=float(x0)
    for _ in range(int(steps)):
        grad=2.0 * a * x + b
        x=x - lr*grad
    return float(x)
