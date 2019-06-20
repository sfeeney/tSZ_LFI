import numpy as np
import scipy.interpolate as si
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import pickle

def gmm(x, *p):
	n_comp = len(p) / 3
	amps = p[0::3]
	means = p[1::3]
	sigmas = p[2::3]
	if np.isscalar(x):
		print 'scalar'
		y = 0.0
	else:
		y = np.zeros(len(x))
	for i in range(n_comp):
		y += amps[i] * np.exp(-0.5 * ((x - means[i]) / sigmas[i]) ** 2)
	return y


# read data
wf = np.array([pickle.load(open('act/ell.pkl')), \
			   pickle.load(open('act/SzWienerFilter.pkl'))]).T
wf[:, 1] /= np.max(wf[:, 1])

# interpolants
ell = np.arange(25000)
wf_li = si.interp1d(wf[:, 0], wf[:, 1], bounds_error=False, fill_value=0.0)
wf_spl = si.UnivariateSpline(wf[:, 0], wf[:, 1], ext='zeros', s=5e-2)

# very simple single Gaussian version of the WF
wf_amp = np.max(wf[:, 1])
wf_mean = np.sum(wf[:, 0] * wf[:, 1]) / np.sum(wf[:, 1])
wf_var = np.sum((wf[:, 0] - wf_mean) ** 2 * wf[:, 1]) / np.sum(wf[:, 1])
wf_std = np.sqrt(wf_var)
wf_gauss = ss.norm.pdf(ell, wf_mean, np.sqrt(wf_var))
wf_gauss /= np.max(wf_gauss)

# optimize!
n_comps_min = 1
n_comps_max = 3
n_comps = np.arange(n_comps_min, n_comps_max + 1)
n_n_comps = len(n_comps)
n_rpt = 10
opt_p_opts = np.zeros((n_n_comps, n_comps_max * 3))
opt_rhos = np.zeros(n_n_comps)
cm = mpcm.get_cmap('plasma')
cols = [cm(c) for c in np.linspace(0.2, 0.8, n_n_comps)]
for i in range(n_n_comps):
	n_comp = n_comps[i]
	p_ini = [wf_amp / n_comp, wf_mean, wf_std] * n_comp
	bounds = [[0.0, 0.0, 1.0] * n_comp, \
			  [1.0, np.max(ell), np.max(ell)] * n_comp]
	p_opts = []
	rhos = np.zeros(n_rpt)
	for j in range(n_rpt):
		p_opt, p_cov = so.curve_fit(gmm, wf[:, 0], wf[:, 1], \
									p0=p_ini, maxfev=100000, \
									bounds=bounds)
		rho = np.sum((gmm(wf[:, 0], *p_opt) - wf[:, 1]) ** 2)
		p_opts.append(p_opt)
		rhos[j] = rho
		#mp.semilogy(n_comp, rho, 'b.')
	ind = np.argmin(rhos)
	opt_p_opts[i, 0: n_comp * 3] = p_opts[ind]
	opt_rhos[i] = rhos[ind]
	print n_comp, opt_rhos[i]

# convolve
conv_kern = ss.norm.pdf(ell, np.max(ell) / 2.0, 150.0)
wf_conv = np.convolve(wf_li(ell), conv_kern, mode='same')

# plot
fig, axes = mp.subplots(1, 2, figsize=(16, 8))
axes[0].plot(wf[:, 0], wf[:, 1], 'k', label='input')
axes[0].plot(ell, wf_spl(ell), 'r', label='spline')
axes[0].plot(ell, wf_conv, 'b', label='smooth')
for i in range(n_n_comps):
	axes[0].plot(ell, gmm(ell, *opt_p_opts[i, :]), color=cols[i], \
				 ls='--', label=r'{:d}-comp GMM'.format(i + 1))
axes[0].legend(loc='upper right')
axes[0].set_xlabel(r'$\ell$')
axes[0].set_ylabel(r'$f_\ell\,[\mu{\rm K}]$')
axes[0].set_xlim(0.0, 10000.0)
axes[0].set_ylim(1.01 * np.min(wf_spl(ell)), 1.01 * np.max(wf[:, 1]))
axes[1].plot(ell, wf_spl(ell) - wf_li(ell), 'r', label='spline')
axes[1].plot(ell, wf_conv - wf_li(ell), 'b', label='smooth')
for i in range(n_n_comps):
	axes[1].plot(ell, gmm(ell, *opt_p_opts[i, :]) - wf_li(ell), \
				 color=cols[i], ls='--')
axes[1].set_xlabel(r'$\ell$')
axes[1].set_ylabel(r'$\Delta f_\ell\,[\mu{\rm K}]$')
axes[1].set_xlim(0.0, 10000.0)
fig.savefig('act/wf_smoothing.pdf', bbox_inches='tight')

# save best option
np.savetxt('act/wf_smoothed.txt', np.transpose([ell, wf_conv]))
