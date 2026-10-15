import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def system(state, t, a):
    x, y, z = state
    dxdt = -a*x**2 + y*z
    dydt = -a*y**2 + x*z
    dzdt = -a*z**2 + x*y
    return [dxdt, dydt, dzdt]

# Пример для a=1 и начальных условий вблизи точки (1,1,1)
a = 1
t = np.linspace(0, 10, 1000)
state0 = [1.1, 0.9, 1.0] # Небольшое возмущение
solution = odeint(system, state0, t, args=(a,))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution[:,0], solution[:,1], solution[:,2])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()