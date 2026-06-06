import numpy as np


def objective(theta):
    x, y = theta
    return 0.08 * x ** 2 + 2.5 * y ** 2


def gradient(theta):
    x, y = theta
    return np.array([0.16 * x, 5.0 * y])


def gradient_descent(start, lr=0.1, steps=80):
    theta = np.array(start, dtype=float)
    losses = []

    for _ in range(steps):
        losses.append(objective(theta))
        theta = theta - lr * gradient(theta)

    return theta, np.array(losses)


def momentum_descent(start, lr=0.1, beta=0.9, steps=80):
    theta = np.array(start, dtype=float)
    velocity = np.zeros_like(theta)
    losses = []

    for _ in range(steps):
        losses.append(objective(theta))
        grad = gradient(theta)
        velocity = beta * velocity + grad
        theta = theta - lr * velocity

    return theta, np.array(losses)


def rmsprop_descent(start, lr=0.05, beta=0.9, eps=1e-8, steps=80):
    theta = np.array(start, dtype=float)
    square_avg = np.zeros_like(theta)
    losses = []

    for _ in range(steps):
        losses.append(objective(theta))
        grad = gradient(theta)
        square_avg = beta * square_avg + (1 - beta) * (grad ** 2)
        theta = theta - lr * grad / (np.sqrt(square_avg) + eps)

    return theta, np.array(losses)


def adam_descent(start, lr=0.08, beta1=0.9, beta2=0.999, eps=1e-8, steps=80):
    theta = np.array(start, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    losses = []

    for step in range(1, steps + 1):
        losses.append(objective(theta))
        grad = gradient(theta)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

    return theta, np.array(losses)


def clip_gradient_by_norm(grad, max_norm):
    norm = np.linalg.norm(grad)

    if norm <= max_norm:
        return grad

    return grad * (max_norm / norm)


def step_decay_lr(initial_lr, step, drop_every=25, drop_factor=0.5):
    num_drops = step // drop_every
    return initial_lr * (drop_factor ** num_drops)


def diagnose_losses(losses):
    if np.any(~np.isfinite(losses)):
        return "NaN or infinity detected. Check learning rate, logs, exponentials, and gradient clipping."

    if losses[-1] > losses[0]:
        return "Loss increased. Learning rate may be too large or gradients may be unstable."

    if losses[-1] > 0.9 * losses[0]:
        return "Loss barely improved. Learning rate may be too small or landscape may be flat."

    if np.std(losses[-10:]) > 0.2 * np.mean(losses[-10:]):
        return "Loss is noisy. Consider larger batch size, schedule, or moving average."

    return "Loss decreased. Optimization seems stable in this toy example."


start = [5.5, 2.5]

optimizers = {
    "gradient descent": gradient_descent(start, lr=0.18),
    "momentum": momentum_descent(start, lr=0.18, beta=0.82),
    "RMSProp": rmsprop_descent(start, lr=0.08),
    "Adam": adam_descent(start, lr=0.12),
}

for name, (theta, losses) in optimizers.items():
    print()
    print(name)
    print("final theta:", theta)
    print("initial loss:", losses[0])
    print("final loss:", losses[-1])
    print("diagnosis:", diagnose_losses(losses))

print()
print("Learning rate comparison")

for lr in [0.03, 0.18, 0.42]:
    theta, losses = gradient_descent(start, lr=lr, steps=40)
    print(f"lr={lr}, final loss={losses[-1]:.6f}, diagnosis={diagnose_losses(losses)}")

print()
print("Gradient clipping example")
large_grad = np.array([3.0, 12.0])
clipped_grad = clip_gradient_by_norm(large_grad, max_norm=5.0)

print("original gradient:", large_grad)
print("original norm:", np.linalg.norm(large_grad))
print("clipped gradient:", clipped_grad)
print("clipped norm:", np.linalg.norm(clipped_grad))

print()
print("Learning rate schedule example")
for step in [0, 10, 25, 40, 50, 75]:
    print(f"step={step}, lr={step_decay_lr(0.1, step):.5f}")
