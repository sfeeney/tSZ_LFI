import numpy as np
import matplotlib.pyplot as mp
import scipy.stats as ss
import scipy.interpolate as si
import map_functions as mfunc
#from scipy.interpolate import interp1d
from pixell import enmap
import pickle

def trap(f, dx):
	return dx * (f[0] / 2.0 + np.sum(f[1:-1]) + f[-1] / 2.0)

# settings
beam_fwhm_arcmin = 1.4
noise_rms_muk_arcmin = 18.0
map_pixel_size = 0.0001440
map_width_pix = 2182
map_height_pix = 364
n_reals = 5
n_patches = 6

# constants
TCMB = 2.726
hPlanck = 6.62607004e-34
kBoltzmann = 1.38064852e-23
frequency = 148e9 # Hertz
gnu = mfunc.g(hPlanck * frequency / kBoltzmann / TCMB)
y2tcmb = TCMB * gnu * 1.0e6

# noise calculations
noise_per_pix = noise_rms_muk_arcmin / \
				(map_pixel_size * 60.0 * 180.0 / np.pi)
l_max_exact = np.sqrt(4.0 * np.pi / map_pixel_size ** 2) - 1.0
l_max = int(l_max_exact)
noise_c_l = np.zeros(25000)
noise_c_l[0: l_max + 1] = (noise_per_pix * map_pixel_size) ** 2

# beam calculations
ell = np.arange(25000)
beam_sigma = beam_fwhm_arcmin * \
             np.pi / 180.0 / 60.0 / \
             np.sqrt(8.0 * np.log(2.0))
b_l = np.exp(-0.5 * (ell * beam_sigma) ** 2)

# define histogram bins used in realisations
binmin = 0.0
binmax = -120.0
binstep = 10.
negbins = np.arange(binmax,binmin+0.001,binstep)
negbincenters = np.arange(binmax+binstep/2.0,binmin-binstep/2.0+0.001,binstep)
posbins = -1.0 * negbins
posbincenters = -1.0 * negbincenters
bincenters = np.ravel(np.array([negbincenters,posbincenters[::-1]]))

# read in CMB C_ls
cmb_ell, cmb_c_l = np.loadtxt('act/cmb_c_l_fiducial.txt', unpack=True)
cmb_l_min = int(cmb_ell[0])
cmb_l_max = int(cmb_ell[-1])

# read in patch coordinates
with open('act/patch_coords.pkl', 'rb') as f:
    wcss = pickle.load(f)

# read in masks
masks = []
apo_masks = []
for jj in xrange(n_patches) :
    masks.append(enmap.read_map('act/mask00' + str(jj)))
    apo_masks.append(enmap.read_map('act/mask00' + str(jj) + 'edges'))

# read in Wiener filter
wf = np.array([pickle.load(open('act/ell.pkl')), \
			   pickle.load(open('act/SzWienerFilter.pkl'))]).T
wf[:, 1] /= np.max(wf[:, 1])
#wf = np.ones((len(ell), 2))
#wf[:, 0] = ell
wf_c_l = si.interp1d(wf[:, 0], wf[:, 1], bounds_error=False, \
					 fill_value=0.0)(ell)

# calculate total C_ls
tot_c_l = np.zeros(25000)
tot_c_l[cmb_l_min: cmb_l_max + 1] += cmb_c_l
tot_c_l *= b_l ** 2
tot_c_l[0: l_max + 1] += (noise_per_pix * map_pixel_size) ** 2
tot_c_l *= wf_c_l ** 2

# calculate expected pixel variance
tot_var = np.sum((2.0 * ell + 1.0) * tot_c_l) / 4.0 / np.pi

'''
# generate realizations
numerics = {}
cosmology = {}
cosmology['noise_rms_muk_arcmin'] = noise_rms_muk_arcmin
numerics['map_pixel_size'] = map_pixel_size
numerics['map_height_pix'] = map_height_pix
numerics['map_width_pix'] = map_width_pix
numerics['n_patch'] = 1
cosmology['beam_fwhm_arcmin'] = beam_fwhm_arcmin
cosmology['cmb_ell'] = cmb_ell
cosmology['cmb_c_l'] = cmb_c_l
pdfs = []
for i in range(n_reals):
	tot_map = mfunc.map_generate_random_noise(numerics, cosmology, wcss[0])
	pdf = mfunc.map_act_hist([tot_map], numerics, wf, masks, apo_masks, \
							 negbins, posbins)
	pdfs.append(pdf / trap(pdf, bincenters[1] - bincenters[0]))
pdfs = np.array(pdfs)
#maps = np.array(maps)
#print(np.mean(maps), np.std(maps), np.sqrt(tot_var))

# summary plots
ana_pdf = ss.norm.pdf(bincenters, 0.0, np.sqrt(tot_var))
mean_pdf = np.mean(pdfs, 0)
std_pdf = np.std(pdfs, 0)
mp.semilogy(bincenters, mean_pdf)
mp.fill_between(bincenters, mean_pdf - std_pdf, \
				mean_pdf + std_pdf, alpha=0.5)
mp.semilogy(bincenters, ana_pdf)
mp.show()
'''


# should already be enough to compare to noise only hists. generate?
# tSZ NEEDS CONVERTING TO TCMB
# mpirun to do all sims
# quantify expected bias / stddev in variance?
# check impact of masking/apodization and quantify. it's deterministic
# right, so same to every map every time.


# calculate statistics from histogram realisations
n_real = 1000
fname = 'outputs/tszpdflfi_test/combined_hist_{:d}.txt'
pdfs = []
for i in range(n_real):
    pdfs.append(np.genfromtxt(fname.format(i)))
pdfs = np.array(pdfs)
n_bins = pdfs.shape[1]
mean_pdf = np.mean(pdfs, 0)
std_pdf = np.std(pdfs, 0)

# read in analytic tSZ pdf
data = np.load('test/P.npz')
ana_pdf = data['P_uncl'][::-1]
#ana_bins = np.linspace(-data['signal_max'] * 1e6, \
#						data['signal_min'] * 1e6, ana_pdf.shape[0])
ana_bins = np.linspace(data['signal_max'], data['signal_min'], \
					   ana_pdf.shape[0]) * y2tcmb
ana_pdf_full = np.append(np.array(ana_pdf), np.zeros(len(ana_pdf) - 1))
ana_bins_full = np.append(ana_bins, -ana_bins[-2::-1])

# analytic pdf needs filtered CMB + noise adding to it...
'''print(tot_var)
noise_pdf = ss.norm.pdf(bincenters, 0.0, np.sqrt(tot_var))
print(trap(noise_pdf, bincenters[1] - bincenters[0]))
mp.plot(bincenters, noise_pdf)
mp.show()
exit()'''

# normalize both PDFs
ana_norm = trap(ana_pdf, ana_bins[1] - ana_bins[0])
ana_pdf /= ana_norm
ana_norm = trap(ana_pdf_full, ana_bins[1] - ana_bins[0])
ana_pdf_full /= ana_norm
mean_norm = trap(mean_pdf, bincenters[1] - bincenters[0])
mean_pdf /= mean_norm
std_pdf /= mean_norm

# convolve tSZ with Gaussian CMB + noise
# NB: edges of the convolution output ("edge" = last few noise-sigma)
#     have edge effects. they're nowhere near the data though
conv = np.convolve(ana_pdf_full, \
				   ss.norm.pdf(ana_bins_full, 0.0, np.sqrt(tot_var)), \
				   mode='same')
conv /= trap(conv, ana_bins[1] - ana_bins[0])

# plot results
mp.semilogy(bincenters, mean_pdf)
mp.fill_between(bincenters, mean_pdf - std_pdf, \
				mean_pdf + std_pdf, alpha=0.5)
mp.semilogy(ana_bins_full, conv, '--')
mp.ylim(1.0e-10, 1.0)
mp.show()




'''
need to
add beam to tsz
multiply tsz by correct factor to get t(?)
add cmb with beam (and pixel window?)
add noise
'''
