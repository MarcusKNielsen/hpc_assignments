import numpy as np
import matplotlib.pyplot as plt



Nthreads = np.arange(1,25)
SpeedUp = np.arange(1,25)

plt.figure()
plt.title("Amdahl's law")

plt.plot(Nthreads,Nthreads)

plt.plot(Nthreads,SpeedUp**(0.25),".-")
plt.plot(Nthreads,SpeedUp**(0.5) ,".-")
plt.plot(Nthreads,SpeedUp**(0.75),".-")

plt.xlabel("p: Number of threads")
plt.ylabel(r"Speed Up: $\dfrac{S(p)}{S(1)}$")
plt.grid(True)

plt.show()