import numpy as np
import math
from matplotlib import pyplot as plt

#overexcited
cosphi_over = np.linspace(0.95, 1, 100)
phi_over = [math.acos(cos) for cos in cosphi_over]
sinphi_over = [math.sin(angle) for angle in phi_over]

#underexcited
cosphi_under = np.linspace(-1, -0.95, 100)[1:]
phi_under = [math.acos(cos) for cos in cosphi_under]
sinphi_under = [math.sin(angle) *-1 for angle in phi_under]

sinphi = sinphi_over + sinphi_under

"""style_string = ''
i = 1
for value in sinphi:
    style_string = style_string + f'x{i}="{i}" y{i}="{value}" '
    i += 1"""

plt.plot(list(range(0, len(sinphi))), sinphi, )
plt.show()

a = 1