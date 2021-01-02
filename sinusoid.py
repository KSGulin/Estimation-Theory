import numpy as np
import warnings
from math import pi


def sinusoid(a, f, p, t0, t1, dt):
# Generates sinusoid with equally spaced time steps
# Inputs:
#   a - Amplitude. Must be positive
#   f- Frequency in radians
#   p - phase in radians
#   t0 - start time
#   t1 - end time
#   dt - time step. Will only be adhered to if (t1-t0) is a multiple of dt.	

	if (dt < 2*f):
		warnings.warn("Sampling frequency below Nyquist")

	t = np.linspace(t0, t1, int((t1-t0)/dt)+1)
	y = np.sin(f*t + p)

	return y

def h_transform(x, t):
	return x[0]*np.sin(x[1]*t + x[2])


def partials(x, t):
	H = np.zeros((len(t), 3))
	for i in range(len(t)):
		H[i][0] = np.sin(x[1]*t[i] + x[2])
		H[i][1] = x[0]*x[1]*np.cos(x[1]*t[i] + x[2])
		H[i][2] = x[0]*np.cos(x[1]*t[i] + x[2])

	return np.matrix(H)

def check_bounds(x):
	if x[1] > 50:
		x[1] = 10 - x[1]
	if x[2] > 2*pi:
		x[2] = 4*pi - x[2]
	if x[2] < 0:
		x[2] = 2*pi + x[2]

	return x
