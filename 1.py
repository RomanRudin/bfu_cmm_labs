import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

g = 9.81


def total_time(h, a, m):
    q = np.sqrt(1 - 2 * a / m)
    return np.sqrt(2 * h / g) * (1 + q) / (1 - q)


masses = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
h = np.linspace(0.1, 3.0, 40)
a = np.linspace(0.005, 0.05, 40)
H, A = np.meshgrid(h, a)

all_T = [total_time(H, A, m)[A < m / 2] for m in masses]
norm = LogNorm(vmin=min(t.min() for t in all_T), vmax=max(t.max() for t in all_T))
cmap = cm.plasma

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

for m in masses:
    valid = A < m / 2
    T = np.where(valid, total_time(H, np.where(valid, A, a.min()), m), norm.vmin)
    colors = cmap(norm(T))
    colors[~valid, 3] = 0
    ax.plot_surface(H, A, np.full_like(H, m), facecolors=colors,
                    rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)

ax.set_xlabel("Начальная высота h, м", labelpad=10)
ax.set_ylabel("a = ΔE / v², кг", labelpad=10)
ax.set_zlabel("Масса шара m, кг", labelpad=10)
ax.set_zlim(0, 1.05)
# ax.set_title("T(h, a, m) = √(2h/g)·(1+q)/(1−q),   q = √(1 − 2a/m)")
ax.view_init(elev=22, azim=-135)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Время процесса T, с (лог. шкала)")

fig.tight_layout()
fig.savefig("1.png", dpi=150)
plt.show()