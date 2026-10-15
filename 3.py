import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import dblquad

R = 2.0
CENTERS = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)

d = 2 - np.sqrt(3)                        # ≈ 0.268
XMIN, XMAX, YMIN, YMAX = d, 2 - d, d, 2 - d
S_RECT = (XMAX - XMIN) * (YMAX - YMIN)

S_EXACT = 4 + 4*np.pi/3 - 4*np.sqrt(3)            # 1.260587
I_EXACT = 24 + 16*np.pi/3 - 22*np.sqrt(3)         # 2.650043


def inside(x, y):
    """Точка внутри всех четырёх кругов."""
    ok = np.ones_like(x, dtype=bool)
    for cx, cy in CENTERS:
        ok &= (x - cx)**2 + (y - cy)**2 <= R**2
    return ok


def f(x, y):
    return x + y**2


def monte_carlo(N, rng):
    xi, eta = rng.random(N), rng.random(N)
    x = XMIN + (XMAX - XMIN) * xi
    y = YMIN + (YMAX - YMIN) * eta
    ind = inside(x, y) 
    M = ind.sum()
    p = M / N
    S = S_RECT * p
    sigma_S = S_RECT * np.sqrt(p * (1 - p) / N)
    g = f(x, y) * ind
    I = S_RECT * g.mean()
    sigma_I = S_RECT * g.std(ddof=1) / np.sqrt(N)
    return x, y, ind, M, S, sigma_S, I, sigma_I


def y_bounds(x):
    cx = 0.0 if x >= 1 else 2.0
    h = np.sqrt(R**2 - (x - cx)**2)
    return 2 - h, h

S_quad = dblquad(lambda y, x: 1.0, XMIN, XMAX,
                 lambda x: y_bounds(x)[0], lambda x: y_bounds(x)[1])[0]
I_quad = dblquad(lambda y, x: f(x, y), XMIN, XMAX,
                 lambda x: y_bounds(x)[0], lambda x: y_bounds(x)[1])[0]
print(f"Точно:      S = {S_EXACT:.6f},  I = {I_EXACT:.6f}")
print(f"Квадратура: S = {S_quad:.6f},  I = {I_quad:.6f}")
print(f"Прямоугольник [{XMIN:.4f}; {XMAX:.4f}]², S_пр = {S_RECT:.6f}\n")

Ns = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000]
rng = np.random.default_rng(4)
rows = []
print(f"{'N':>9} {'M':>8} {'S':>9} {'σ_S':>8} {'|ΔS|':>8} {'I':>9} {'σ_I':>8} {'|ΔI|':>8} {'δI, %':>6}")
for N in Ns:
    x, y, ind, M, S, sS, I, sI = monte_carlo(N, rng)
    rows.append((N, M, S, sS, abs(S - S_EXACT), I, sI, abs(I - I_EXACT)))
    print(f"{N:>9} {M:>8} {S:9.5f} {sS:8.5f} {abs(S-S_EXACT):8.5f} "
          f"{I:9.5f} {sI:8.5f} {abs(I-I_EXACT):8.5f} {100*abs(I-I_EXACT)/I_EXACT:6.3f}")
    if N == 5_000:
        sample = (x, y, ind)
rows = np.array(rows)

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
x, y, ind = sample
ax[0].scatter(x[~ind], y[~ind], s=3, c="#d62728", alpha=0.5, label="вне фигуры")
ax[0].scatter(x[ind], y[ind], s=3, c="#1f77b4", alpha=0.6, label="внутри (M точек)")
t = np.linspace(0, 2*np.pi, 400)
for cx, cy in CENTERS:
    ax[0].plot(cx + R*np.cos(t), cy + R*np.sin(t), "k", lw=0.6, alpha=0.3)
    ax[0].plot(cx, cy, "k+", ms=10)
ax[0].add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN, fill=False,
                          ec="g", lw=2, ls="--", label="прямоугольник S_пр"))
verts = [(1, d), (2 - d, 1), (1, 2 - d), (d, 1)]
ax[0].plot(*zip(*verts), "ko", ms=6)
ax[0].set(xlim=(-0.4, 2.4), ylim=(-0.4, 2.4), aspect="equal", xlabel="x", ylabel="y",
          title="Фигура и N = 5000 случайных точек")
ax[0].legend(loc="upper right", fontsize=9)

gx, gy = np.meshgrid(np.linspace(XMIN, XMAX, 400), np.linspace(YMIN, YMAX, 400))
gz = np.where(inside(gx, gy), f(gx, gy), np.nan)
cs = ax[1].contourf(gx, gy, gz, 20, cmap="viridis")
fig.colorbar(cs, ax=ax[1], label="z = x + y²")
for cx, cy in CENTERS:
    ax[1].plot(cx + R*np.cos(t), cy + R*np.sin(t), "k", lw=0.6, alpha=0.3)
ax[1].set(xlim=(-0.4, 2.4), ylim=(-0.4, 2.4), aspect="equal", xlabel="x", ylabel="y",
          title="Функция z = x + y² на фигуре")
fig.tight_layout()
fig.savefig("fig1_figure_points.png", dpi=150)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for k, (col, scol, exact, name) in enumerate([(2, 3, S_EXACT, "S"), (5, 6, I_EXACT, "I")]):
    ax[k].errorbar(rows[:, 0], rows[:, col], yerr=2*rows[:, scol], fmt="o-",
                   capsize=4, label=f"оценка {name} ± 2σ")
    ax[k].axhline(exact, color="r", ls="--", label=f"точное {name} = {exact:.5f}")
    ax[k].set(xscale="log", xlabel="N", ylabel=name,
              title=f"Сходимость оценки {'площади' if name == 'S' else 'интеграла'}")
    ax[k].legend()
ax[2].loglog(rows[:, 0], rows[:, 4], "o-", label="|ΔS| факт.")
ax[2].loglog(rows[:, 0], rows[:, 3], "s--", label="σ_S оценка")
ax[2].loglog(rows[:, 0], rows[:, 7], "o-", label="|ΔI| факт.")
ax[2].loglog(rows[:, 0], rows[:, 6], "s--", label="σ_I оценка")
n = np.array([1e3, 1e6])
ax[2].loglog(n, 1.5/np.sqrt(n), "k:", lw=2, label="~ 1/√N")
ax[2].set(xlabel="N", ylabel="погрешность", title="Погрешность: O(1/√N)")
ax[2].legend(fontsize=9)
for a in ax:
    a.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_convergence.png", dpi=150)
print("\nграфики сохранены")