import sinusoid
import nonlinearwls as wls
import numpy as np
from math import pi


truth = sinusoid.sinusoid(1, 2*pi, pi/6, 0, 2, .01)
samples = truth[0::10]
samples = samples + np.random.normal(0, .15, 21)

t = np.linspace(0, 2, 21)
x0 = np.asarray([.5,2*pi, 0])
P0 = np.matrix(np.diag([10, 10]))
R = np.matrix(np.diag(.15*np.ones(len(t))))
est_inds = np.asarray([0,2])

wls.run_to_convergence(samples, t, x0, est_inds, R, P0, sinusoid.h_transform, sinusoid.partials, 10e-6)

print(5)