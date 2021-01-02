import numpy as np

def update_step(y, x0, H, R, P0, hx, x0_bar):
	term1 = np.linalg.inv(np.transpose(H)*np.linalg.inv(R)*H + np.linalg.inv(P0))
	term2 = np.dot(np.transpose(H)*np.linalg.inv(R), (y-hx))
	term3 = np.dot(np.linalg.inv(P0), (x0_bar - x0))
	return np.squeeze(x0 + np.asarray(np.transpose(np.dot(term1, np.transpose((term2 + term3))))))

	### Add dhecking function

def run_to_convergence(y, t, x0, est_inds, R, P0, h_transform, partial_func, e, condition_func = None):
	x0_bar = x0[est_inds]
	while True:
		hx = h_transform(x0, t)
		H = partial_func(x0, t)
		xt = update_step(y, x0[est_inds], H[:,est_inds], R, P0, hx, x0_bar)

		x1 = np.zeros(len(x0))
		j = 0
		for i in range(len(x0)):
			if i not in est_inds:
				x1[i] = x0[i]
			else:
				x1[i] = xt[j]
				j += 1

		if (condition_func):
			x1 = condition_func(x1)

		if np.mean(np.abs(x1 - x0)) < e:
			return x1

		x0 = x1