import numpy as np
import matplotlib.pyplot as plt

# Параметры
def f(x):
    return 2*x - 0.5*x**3

x_vals = np.linspace(-2.5, 2.5, 400)
y_vals = f(x_vals)

# 1. График функции и y=x
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='\(f(x) = 2x - 0.5x^3\)')
plt.plot(x_vals, x_vals, 'r--', label='\(y = x\)')
plt.scatter([0, np.sqrt(2), -np.sqrt(2)], [0, np.sqrt(2), -np.sqrt(2)], color='black', zorder=5)
plt.title('Особые точки')
plt.grid(True)
plt.legend()

# 2. Паутинная диаграмма для начального условия x0 = 0.5
x0 = 0.5
trajectory = [x0]
for _ in range(10):
    x0 = f(x0)
    trajectory.append(x0)

plt.subplot(1, 2, 2)
plt.plot(x_vals, y_vals, 'b')
plt.plot(x_vals, x_vals, 'r--')
# Рисуем "паутину"
x_curr = 0.5
for _ in range(5):
    y_curr = f(x_curr)
    plt.plot([x_curr, x_curr], [x_curr, y_curr], 'g-')
    plt.plot([x_curr, y_curr], [y_curr, y_curr], 'g-')
    x_curr = y_curr
plt.title('Паутинная диаграмма (x0=0.5)')
plt.grid(True)

plt.tight_layout()
plt.show()