import sinusoid
import nonlinearwls as wls
import numpy as np
from math import pi
import itertools


truth = sinusoid.sinusoid(1, 2*pi, pi/6, 0, 2, .01)
samples = truth[0::10]
samples = samples + np.random.normal(0, .15, 21)

init_scale = np.linspace(0, 10, 101)
init_p = np.linspace(-1, 7, 10)

grid = list(itertools.product(*[init_scale, init_p]))
x_true = np.asarray([1, 2*pi, pi/6])


est_inds = np.asarray([0,2])
t = np.linspace(0, 2, 21)
R = np.matrix(np.diag(.15*np.ones(len(t))))
x0 = np.zeros(3)
results = np.zeros((len(grid), 2))
for n, s in enumerate(grid):
	for i in range(len(x0)):
		if i not in est_inds:
			x0[i] = x_true[i]
		else:
			x0[i] = x_true[i]*s[0]
	P0 = np.matrix(np.diag([s[1], s[1]]))
	xf = wls.run_to_convergence(samples, t, x0, est_inds, R, P0, sinusoid.h_transform, sinusoid.partials, 10e-6, sinusoid.check_bounds)
	results[n] = xf[est_inds]

print(5)