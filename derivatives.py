import numpy as np
import matplotlib.pyplot as mp
import onepoint_analytic as opa
import pickle
import scipy.interpolate as si
import scipy.stats as ss
import camb
from camb import model, initialpower, get_matter_power_interpolator
import map_functions as mfunc

# MPI job allocation function
def allocate_jobs(n_jobs, n_procs=1, rank=0):
    n_j_allocated = 0
    for i in range(n_procs):
        n_j_remain = n_jobs - n_j_allocated
        n_p_remain = n_procs - i
        n_j_to_allocate = n_j_remain / n_p_remain
        if rank == i:
            return range(n_j_allocated, \
                         n_j_allocated + n_j_to_allocate)
        n_j_allocated += n_j_to_allocate

# a_s to sigma_8 calculation
def a_s_to_sigma_8(cosmology_parameters):

    pars = camb.CAMBparams()
    pars.set_cosmology(
            H0 = cosmology_parameters['H0'],
            ombh2 = cosmology_parameters['Ob0'] * cosmology_parameters['h']**2,
            omch2 = cosmology_parameters['Oc0'] * cosmology_parameters['h']**2,
            omk = 0.,
            mnu = cosmology_parameters['Mnu'],
            standard_neutrino_neff = cosmology_parameters['Neff'],
            nnu = cosmology_parameters['Neff'],
            TCMB = cosmology_parameters['TCMB']
            )
    pars.InitPower.set_params(ns = cosmology_parameters['ns'], \
                              As = cosmology_parameters['As'], \
                              pivot_scalar = cosmology_parameters['pivot_scalar'])
    pars.NonLinear = model.NonLinear_none
    pars.set_matter_power(redshifts = [0.], kmax = 20.)
    results = camb.get_results(pars)
    return results.get_sigma8()[0]

# a_s to sigma_8 inversion via bisection
def sigma_8_to_a_s(sigma_8_tgt, cosmology_parameters, tol=0.0001, \
                   iter_max=100):
    
    upper_bound = 5e-9 # 1.269 * 2.71826876e-09
    lower_bound = 1e-9 # 0.76 * 2.71826876e-09
    a_s_guess = cosmology_parameters['As']
    delta = a_s_to_sigma_8(cosmology_parameters) - sigma_8_tgt
    for i in range(iter_max):
        if np.abs(delta) < tol:
            break
        else:
            if delta > 0.0:
                upper_bound = cosmology_parameters['As']
                a_s_guess = (lower_bound + cosmology_parameters['As']) / 2.0
            elif delta < 0.0:
                lower_bound = cosmology_parameters['As']
                a_s_guess = (upper_bound + cosmology_parameters['As']) / 2.0
            cosmology_parameters['As'] = a_s_guess
            delta = a_s_to_sigma_8(cosmology_parameters) - sigma_8_tgt
    if i == iter_max - 1:
        print('ERROR: bisection unconverged within max its!')
        exit()
    return a_s_guess

def trap(f, dx):
	return dx * (f[0] / 2.0 + np.sum(f[1:-1]) + f[-1] / 2.0)


# basic settings
use_mpi = True
generate_pdfs = False
path = './outputs/'
no_smooth = False
alt_wf = True
rve = False
hi_res = True
stub = ''
if hi_res:
	stub += '_hi_res'
if no_smooth:
	stub += '_no_smooth'
elif alt_wf:
	stub += '_alt_wf'
if rve:
	stub += '_raise_val_err'

# set up MPI environment
if use_mpi:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
else:
    n_procs = 1
    rank = 0

# fiducial parameters
# a_s_fid = 2.71826876e-09
sig_8_fid = 0.7999741174575746
om_m_fid = 0.25
p_0_fid = 1.0
sig_n_fid = 1.0
delta = 0.1 # 0.01
dm = 1.0 - delta
dp = 1.0 + delta

# constants
TCMB = 2.726
hPlanck = 6.62607004e-34
kBoltzmann = 1.38064852e-23
frequency = 148e9 # Hertz
gnu = mfunc.g(hPlanck * frequency / kBoltzmann / TCMB)
y2tcmb = TCMB * gnu * 1.0e6

# tSZ grids
if hi_res:
    n_grid_m = 50 * 5 # 3
    n_grid_z = 51 * 5 # 3
else:
    n_grid_m = 50
    n_grid_z = 51

# experimental settings
beam_fwhm_arcmin = 1.4
noise_rms_muk_arcmin = 18.0
map_pixel_size = 0.0001440

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

# read in CMB C_ls
cmb_ell, cmb_c_l = np.loadtxt('act/cmb_c_l_fiducial.txt', unpack=True)
cmb_l_min = int(cmb_ell[0])
cmb_l_max = int(cmb_ell[-1])

# read in Wiener filter
if alt_wf:
	wf = np.genfromtxt('act/wf_smoothed.txt')
else:
	wf = np.array([pickle.load(open('act/ell.pkl')), \
	               pickle.load(open('act/SzWienerFilter.pkl'))]).T
wf[:, 1] /= np.max(wf[:, 1])
wf_interp = si.interp1d(wf[:, 0], wf[:, 1], \
                        bounds_error=False, fill_value=0.0)
wf_c_l = wf_interp(ell)

# calculate total C_ls
tot_c_l = np.zeros(25000)
tot_c_l[cmb_l_min: cmb_l_max + 1] += cmb_c_l
tot_c_l *= b_l ** 2
tot_c_l[0: l_max + 1] += (noise_per_pix * map_pixel_size) ** 2
tot_c_l *= wf_c_l ** 2

# calculate expected pixel variance
tot_var = np.sum((2.0 * ell + 1.0) * tot_c_l) / 4.0 / np.pi

# parameter combinations required for numerical derivatives (for 
# sigma_8, omega_m and p_0)
run_pars = np.array([[sig_8_fid, om_m_fid, p_0_fid, sig_n_fid], \
                     [sig_8_fid * dm, om_m_fid, p_0_fid, sig_n_fid], \
                     [sig_8_fid * dp, om_m_fid, p_0_fid, sig_n_fid], \
                     [sig_8_fid, om_m_fid * dm, p_0_fid, sig_n_fid], \
                     [sig_8_fid, om_m_fid * dp, p_0_fid, sig_n_fid], \
                     [sig_8_fid, om_m_fid, p_0_fid * dm, sig_n_fid], \
                     [sig_8_fid, om_m_fid, p_0_fid * dp, sig_n_fid]])
n_jobs = run_pars.shape[0]
job_list = allocate_jobs(n_jobs, n_procs, rank)

# numerical parameters: same for all runs
num_dict = {

    # if this option is set to False, an error is thrown if you try
    # to compute something that already exists in path.
    'debugging': True,

    'verbose': True,

    # you probably want to keep this as tSZ, there is some code
    # for weak lensing convergence as well
    'signal_type': 'tSZ',

    # number of datapoints for various grids.
    # The values here are reasonably conservative.
    'Npoints_theta': 1000,
    ###'Npoints_theta': 200, # IS THIS OKAY?
    'Npoints_M': n_grid_m,
    'Npoints_z': n_grid_z,
    
    # grid boundaries
    'logM_min': 11.,
    'logM_max': 16.,
    'z_min': 0.005,
    'z_max': 6.,

    # maximum signal for which you want to evaluate the PDF (Compton-y for tSZ)
    # due to ringing, I'd recommend setting this to twice the maximum value
    # that you're actually interested in.
    'signal_max': 300e-6,

    # number of datapoints at which you want to know the PDF
    # the default value is overkill, but runtime is not a problem.
    # choose this as a power of 2, since a lot of FFTs are being done!
    'Npoints_signal': 2**17,

    # sidelength of the quadratic pixels in radians.
    # note that the pixelisation is only self-consistent at power-spectrum level.
    'pixel_sidelength': map_pixel_size,

    # smoothing and filtering
    'Wiener_filter': lambda ell: wf_interp(ell),
    'gaussian_kernel_FWHM': beam_fwhm_arcmin / 60.0 * np.pi / 180.0
}
if no_smooth:
	num_dict['pixel_sidelength'] = None
	num_dict['Wiener_filter'] = None
	num_dict['gaussian_kernel_FWHM'] = None
num = opa.numerics(num_dict)

# loop over jobs
if generate_pdfs:

	for i in job_list:

	    # set cosmological parameters
	    print(rank, i, run_pars[i, :])
	    cos_par = {
	        'H0': 70.,# km/s/Mpc
	        'Om0': run_pars[i, 1],# dimensionless (total) matter density
	        'w': -1., # flat LCDM
	        'ns': 0.96,# scalar spectral index
	        'As': 2.71826876e-9,# scalar amplitude at pivot scale
	        'pivot_scalar': 0.002, # pivot scale k_piv=0.002 Mpc^-1
	        'Ob0': 0.043,# dimensionless baryon density
	        'Mnu': 0., # no neutrinos
	        'Neff': 0.,# no neutrinos
	        'TCMB': 2.726,
	        'pressure_profile_P0': run_pars[i, 2] * 18.1
	    }
	    cos_par['h'] = cos_par['H0'] / 100.0
	    cos_par['OL0'] = 1.0 - cos_par['Om0'] #flat LCDM
	    cos_par['Oc0'] = cos_par['Om0'] - cos_par['Ob0'] # CDM density
	    cos_par['rhoM'] = cos_par['Om0'] * 2.7753e11
	    cos_par['As'] = sigma_8_to_a_s(run_pars[i, 0], cos_par)

	    # generate profiles
	    path = './outputs/deriv_run_{:d}'.format(i) + stub + '_'
	    cosmo = opa.cosmology(cos_par)
	    cosmo.create_HMF_and_bias(path, num)
	    pr = opa.profiles(cosmo, num)
	    pr.create_profiles(path)
	    if num.pixel_sidelength is not None or \
	       num.Wiener_filter is not None or \
	       num.gaussian_kernel_FWHM is not None:
		    pr.create_convolved_profiles(path)
	    pr.create_tildes(path)
	    p = opa.PDF(cosmo, num, pr)
	    p.create_alpha0(path)
	    p.create_P_tilde(path)

	    # this creates the final result.
	    # it is stored in the file path + 'P.npz'
	    # this file has fields:
	    #      P_uncl     : the PDF without clustering contribution
	    #      P_cl       : the PDF including clustering contribution
	    #      signal_min : minimum signal (0)
	    #      signal_max : maximum signal
	    # the PDF values are on an equally spaced grid between 
	    # signal_min and signal_max
	    p.create_P(path)

# aggregate results and form derivatives
if use_mpi:
	mpi.COMM_WORLD.barrier()
if rank == 0:

	# read all PDFs
	pdfs = []
	for i in range(n_jobs):

		# read raw files
		path = './outputs/deriv_run_{:d}'.format(i) + stub + '_'
		data = np.load(path + 'P.npz')
		pdf = data['P_uncl'][::-1]

		# define full set of bins, cut set of bins and fiducial 
		# noise smoothing
		if i == 0:
			bins = np.linspace(data['signal_max'], \
							   data['signal_min'], \
							   pdf.shape[0]) * y2tcmb
			i_cut_sys = np.abs(bins) < 700.0
			bins = bins[i_cut_sys]
			bins = np.append(bins, -bins[-2::-1])
			dtcmb = bins[1] - bins[0]
			noise_pdf = ss.norm.pdf(bins, 0.0, np.sqrt(tot_var))
			i_cut = np.abs(bins) < 200.0
			# @TODO: set cut limit to 690 to see if increasing 
			# n_theta makes more extreme bins better behaved
			bins_cut = bins[i_cut]

		# extend to positive delta-T_CMB and convolve all but 
		# fiducial tSZ PDF with fiducial noise PDF
		pdf = np.append(np.array(pdf[i_cut_sys]), \
						np.zeros(np.sum(i_cut_sys) - 1))
		if i == 0:
			pdfs.append(pdf)
		else:
			if no_smooth:
				pdfs.append(pdf[i_cut] / trap(pdf[i_cut], dtcmb))
			else:
				conv = np.convolve(pdf, noise_pdf, mode='same')
				pdfs.append(conv[i_cut] / trap(conv[i_cut], dtcmb))
			#mp.semilogy(bins_cut, pdfs[-1])
	#mp.show()

	# alter smoothing for sigma_noise derivatives
	if no_smooth:
		pdf_fid = pdfs[0][i_cut] / trap(pdfs[0][i_cut], dtcmb)
		pdf_minus = pdf_fid
		pdf_plus = pdf_fid
	else:
		pdf_fid = np.convolve(pdfs[0], noise_pdf, mode='same')
		pdf_fid = pdf_fid[i_cut] / trap(pdf_fid[i_cut], dtcmb)
		noise_pdf = ss.norm.pdf(bins, 0.0, dm * np.sqrt(tot_var))
		pdf_minus = np.convolve(pdfs[0], noise_pdf, mode='same')
		pdf_minus = pdf_minus[i_cut] / trap(pdf_minus[i_cut], dtcmb)
		noise_pdf = ss.norm.pdf(bins, 0.0, dp * np.sqrt(tot_var))
		pdf_plus = np.convolve(pdfs[0], noise_pdf, mode='same')
		pdf_plus = pdf_plus[i_cut] / trap(pdf_plus[i_cut], dtcmb)

	# derivatives: sigma_8, omega_m, p_0...
	derivs = np.zeros((len(bins_cut), 4))
	for i in range(3):
		ii = 2 * i
		derivs[:, i] = (pdfs[ii + 2] - pdfs[ii + 1]) / \
					   (run_pars[ii + 2, i] - run_pars[ii + 1, i])
	derivs[:, 3] = (pdf_plus - pdf_minus) / (dp - dm)

	# deltas too: just (pdf_plus - pdf_fid) / pdf_fid
	deltas = np.zeros((len(bins_cut), 4))
	for i in range(3):
		deltas[:, i] = pdfs[2 * (i + 1)] / pdf_fid - 1.0
	deltas[:, 3] = pdf_plus / pdf_fid - 1.0
	if delta == 0.1:
		deltas[:, 0] *= 0.1

	# plots
	cols = ['b', 'g', 'r', 'k']
	labs = [r'\sigma_8', r'\Omega_m', r'P_0', r'\sigma_{\rm noise}']
	fig, axes = mp.subplots(1, 2, figsize=(16, 5))
	for i in range(4):
		if i == 0 and delta == 0.1:
			axes[0].plot(bins_cut, deltas[:, i], cols[i], \
						 label=r'$+\Delta '+labs[i]+r'(\times 0.1)$')
		else:
			axes[0].plot(bins_cut, deltas[:, i], cols[i], \
						 label=r'$+\Delta '+labs[i]+'$')
		axes[1].semilogy(bins_cut, derivs[:, i], color=cols[i], \
						 label='$'+labs[i]+'$')
		axes[1].semilogy(bins_cut, np.abs(derivs[:, i]), color=cols[i], \
						 ls=':')
	axes[0].set_xlabel(r'$T\,[\mu{\rm K}]$')
	axes[0].set_ylabel(r'$p^+/p^{\rm fid} - 1$')
	axes[0].set_xlim(bins_cut[0], 0.0)
	if no_smooth:
		axes[0].legend(loc='lower left')
		axes[0].set_ylim(-0.4, 0.4)
	else:
		axes[0].legend(loc='upper left')
		axes[0].set_ylim(np.min(deltas[bins_cut <= 0.0]) * 1.01, \
						 np.max(deltas[bins_cut <= 0.0]) * 1.01)
	axes[1].set_xlabel(r'$T\,[\mu{\rm K}]$')
	axes[1].set_ylabel(r'$\partial p/\partial\theta$')
	axes[1].legend(loc='upper left')
	axes[1].set_xlim(bins_cut[0], 75.0)
	axes[1].set_ylim(1e-9, np.max(np.abs(derivs)) * 1.01)
	mp.savefig('./outputs/derivatives' + stub + '.pdf', \
			   bbox_inches='tight')
	mp.close()

	exit()

# p_0!
# need to ensure sigma_8 doesn't change when we change omega_m!
#  - should already be done. hmm. inversion not accurate enough?
#  - DOESN'T FIX: check 5% derivatives
#  - DONE: check fractional changes a la Hill
#     + Omega_M deltas look very different to Leander's
#     + NB: my deltas are T_CMB not Y. should be stretched, but same shape
#     + is sigma_8 not accurate enough? the tolerance is 0.0001 though, 
#       that's 0.01 percent!. can't be this. is he calculating same thing?
#     + I BELIEVE these are plotted with sigma_8 changing too
#  - DONE: check what happens if you vary Omega_M fixing A_s not sigma_8
#     + rerun derivatives with only two variations: pm om_m
#     + do not recalculate As after setting om_m
#     + save original derivatives
#     + or do all this by hand
#     + resulting deltas look much more like CH plots, but these are 
#       different to Leander's. hmm.
#     + STILL WIGGLY THOUGH!
# derivatives look very noisy: up the settings to what leander uses?
#  - NO: Npoints_theta?
#     + still noisy after increasing this to 1000
#  - NO: is it WF? is it beam?
#     + don't think so: noise is in no filter version
#     + does bring the noise to lower signals though. and introduce
#       weird increase at large y
#     + DONE: CLIP THIS REGION OFF. first 100000 indices
#     + including wf and beam does make things noisier, they're just 
#       also noisy to begin with. ask leander
#  - wf or beam?
#     + wf deffo exacerbates more than beam
#  - NO: what about pixel window function?
#     + still noisy without, but i believe i'm still calling convolve
#     + still noisy without convolve
#  - why would it be stochastic? it's not, right?
#  - DONE: reduce Npoints_signal? effectively smoothing? NOPE.

#  - am i using right pixel wf? using pre-calculated it seems
#     + NO. @TODO: fix this.

