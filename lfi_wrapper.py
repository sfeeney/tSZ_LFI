import numpy as np
import os
import pyDOE as pd
import camb
from camb import model, initialpower, get_matter_power_interpolator

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

# parameters specifying the numerical behaviour of the code #{{{
numerical_parameters = {

    # use MPI to parallelize
    'use_mpi': True,

    # set this to True if you want to generate dndOmega and the y-profiles
    # then some non-standard packages will be required
    # TODO
    'do_physics': False,
    
    # set this to true if you have already generated dndOmega and the y-profiles
    # then only maps can be generated using this data
    # TODO
    'do_maps': True,
    
    # set this to true if you have already generated maps
    # then only histograms can be generated using this data
    # TODO
    'do_hists': True,
    
    # do you want the code to pester you with printouts?
    'verbose': True,
    
    # do you want repeatable random numbers?
    'constrain': False,
    
    # angular sidelength of the final map in radians
    # TODO[364, 2182]
    'map_size': 18.0*np.pi/180./0.8,
    'map_width': 18.0*np.pi/180.,
    'map_height': 3.*np.pi/180.,
    'map_width_pix': 2182,
    'map_height_pix': 364,

    # the code takes only a fraction of the final map in order to get rid of edge effects.
    #   I didn't find an efficient way to implement period boundary conditions.
    #   0.8 should be small enough, larger values will probably work as well
    'map_fraction': 0.8,

    # pixel sidelength in radians
    # TODO
    #'map_pixel_size': 0.5/60.*np.pi/180,
    'map_pixel_size': 0.0001440,

    # parameters for the mass grid
    #   for the mass definition see the cosmology parameters below
    #   NOTE : all units are "h-units", i.e. [M] = M_sun/h etc.
    # TODO
    'map_logM_min': 11.,
    'map_logM_max': 16.,
    'map_Npoints_M': 50,
    
    # parameters for the redshift grid
    # TODO
    'map_z_min': 0.005,
    'map_z_max': 6.,
    'map_Npoints_z': 51,

    # measures the number of datapoints in each pixel over which the y-signal is averaged
    #   if map_grid_per_pixel = 1, a 3x3 grid is applied
    #   if map_grid_per_pixel = 2, a 5x5 grid is applied
    #   etc.
    'map_grid_per_pixel': 1,
    
    # whether to choose the number of clusters according to the Poisson distribution
    #   if set to False, the distribution given in arXiv:1812.05584 pg 6 IV A is used
    # should be set to True if all moments are needed (False produces only the correct average)
    'map_Poisson': True,

    # Fit function for the halo mass function
    # currently, only 'Tinker10' and 'Tinker08' are available
    'hmf_function': 'Tinker10',

    # Properties of the theta grid on which the y-profiles are evaluated
    #   theta_out = theta_max_scale * theta(r_out), where r_out is chosen according to the definition below
    'theta_max_scale': 2.5,
    #   the y-profile is evaluated on a finer grid close to the centre, because it is more rapidly
    #   varying there. theta_boundary specifies where the transition from the fine to the coarse grid
    #   is made, the precise value is not very important
    'theta_boundary': 1./4.,
    #   N_theta is a measure for the number of theta gridpoints, the actual number is about 3x larger
    'N_theta': 200,

    # Text file containing the noise power spectrum
    #   assumed to have 2 columns, the 1st one is ell, the second one Cell
    # TODO
    'noise_power_spectrum_file': 'SO_LAT_Nell_T_goal_fsky0p4_ILC_tSZ.txt',

    # some integration boundaries, should not need any changing
    'k_max': 100,
    'k_min': 1e-10,

    # power spectrum limit
    'l_max': 9000,

    # ACT/LFI-specific settings
    'n_patch': 6,
    'n_real': 2,

    # parameter grids
    'sigma_8_ext': np.array([0.7, 0.9]),
    'omega_m_ext': np.array([0.2, 0.37]),
    'p_0_ext': np.array([0.5, 1.5]),
    'sigma_n_ext': np.array([0.85, 1.15]),
    #'sigma_n_ext': np.array([2.3, 2.9]),

    # path to map storage
    #'path': '/mnt/ceph/users/sfeeney/tszpdflfi'
    'path': 'outputs'

    }
#}}}
path = numerical_parameters['path']

# set up MPI environment
if numerical_parameters['use_mpi']:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
else:
    n_procs = 1
    rank = 0

# set up random seeds
if numerical_parameters['constrain']:
    seed = 231014 + rank
else:
    if rank == 0:
        seeds = np.random.randint(221216, 231014, 1)[0]
        seeds = 221216 + np.random.choice(231014 - 221216, \
                                          size=n_procs, \
                                          replace=False)
    else:
        seeds = None
    if numerical_parameters['use_mpi']:
        seed = mpi.COMM_WORLD.scatter(seeds, root=0)
np.random.seed(seed)

# generate or read job list and corresponding parameter values
par_ranges = np.array([numerical_parameters['sigma_8_ext'], \
                       numerical_parameters['omega_m_ext'], \
                       numerical_parameters['p_0_ext'], \
                       numerical_parameters['sigma_n_ext']])
job_list = allocate_jobs(numerical_parameters['n_real'], n_procs, rank)
if numerical_parameters['do_physics']:

    # generate simulation locations on latin hypercube
    grid_locs = np.zeros((numerical_parameters['n_real'], 4))
    if rank == 0:
        grid_fracs = pd.lhs(4, numerical_parameters['n_real'], 'maximin')
        grid_locs = par_ranges[:, 0] + grid_fracs * \
                    (par_ranges[:, 1] - par_ranges[:, 0])
        np.savetxt(path + '/par_grid.txt', grid_locs)
        if numerical_parameters['constrain']:
            np.random.seed(seed)
    if numerical_parameters['use_mpi']:
        mpi.COMM_WORLD.Bcast(grid_locs, root=0)

else:

    # read from file
    grid_locs = np.genfromtxt(path + '/par_grid.txt')
    n_real_file = grid_locs.shape[0]
    if n_real_file != numerical_parameters['n_real']:
        if rank == 0:
            print('{:d}'.format(n_real_file) + \
                  ' realisations specified in input file;')
            print('{:d}'.format(numerical_parameters['n_real']) + \
                  ' requested in Python setup')
            print('producing {:d} realisations'.format(n_real_file))
        numerical_parameters['n_real'] = n_real_file

# context-specific imports
if numerical_parameters['do_physics'] :
    import physics_functions as pfunc
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from colossus.cosmology import cosmology as colossus_cosmology
if numerical_parameters['do_maps'] or numerical_parameters['do_hists'] :
    import map_functions as mfunc
    from scipy.interpolate import interp1d
    from pixell import enmap
    import pickle

# cosmology-independent i/o & setup
if numerical_parameters['do_maps'] or numerical_parameters['do_hists']:

    # read in CMB C_ls
    cmb_ell, cmb_c_l = np.loadtxt('act/cmb_c_l_fiducial.txt', unpack=True)

    # read in patch coordinates
    with open('act/patch_coords.pkl', 'rb') as f:
        wcss = pickle.load(f)

    # read in masks
    masks = []
    apo_masks = []
    for jj in xrange(numerical_parameters['n_patch']) :
        masks.append(enmap.read_map('act/mask00' + str(jj)))
        apo_masks.append(enmap.read_map('act/mask00' + str(jj) + 'edges'))

    # read in Wiener filter
    ell = pickle.load(open('act/ell.pkl'))
    Fell = pickle.load(open('act/SzWienerFilter.pkl'))
    wf = np.zeros((len(ell), 2))
    wf[:, 0] = ell
    wf[:, 1] = Fell / np.max(Fell)

    # define histogram bins
    binmin = 0.0
    binmax = -120.0
    binstep = 10.
    negbins = np.arange(binmax,binmin+0.001,binstep)
    negbincenters = np.arange(binmax+binstep/2.0,binmin-binstep/2.0+0.001,binstep)
    posbins = -1.0 * negbins
    posbincenters = -1.0 * negbincenters
    bincenters = np.ravel(np.array([negbincenters,posbincenters[::-1]]))


# loop over each process's jobs
for ii in job_list:

    grid_locs[:, 0] = 0.7999741174575746
    grid_locs[:, 1] = 0.25
    print(rank, grid_locs[ii, 0], grid_locs[ii, 1])
    exit()

    # cosmology parameters #{{{
    cosmology_parameters = {
        
        # Standard cosmology parameters
        # TODO
        'H0': 70.,# km/s/Mpc
        #'Om0': 0.25,# dimensionless (total) matter density
        'Om0': grid_locs[ii, 1],# dimensionless (total) matter density
        'w': -1., # flat LCDM
        'ns': 0.96,# scalar spectral index
        'As': 2.71826876e-9,# scalar amplitude at pivot scale k_piv=0.002 Mpc^-1
        'pivot_scalar': 0.002,
        'Ob0': 0.043,# dimensionless baryon density
        'Mnu': 0., # no neutrinos
        'Neff': 0.,# no neutrinos
        'TCMB': 2.726,# K

        # Physical constants
        'delta_c': 1.686,
        'c0': 2.99792458e5, #km/s
        'hPlanck': 6.62607004e-34,
        'kBoltzmann':1.38064852e-23,

        # Observation frequency, required to convert Compton-y to temperature decrement
        # TODO
        'frequency': 148e9, # Hertz

        # Mass definition of the mass grid
        #   '200m', '200c', 'vir'
        'mass_def_initial': '200m',

        # Mass definition of the y-profile fitting function
        #   '200c'
        'mass_def_batt': '200c',

        # Mass definition of the HMF fitting function
        #   '200m'
        'mass_def_Tinker': '200m',

        # Definition of the outer radius
        #   'vir', '200'
        'r_out_def': 'vir',

        # Fitting Models
        'concentration_model': 'duffy08',
        'halo_profile': 'nfw',

        # Pressure profile parameters, as defined in arXiv:1812.05584 pg 4
        'pressure_profile_P0': 18.1,
        'pressure_profile_xc0': 0.497,
        'pressure_profile_beta0': 4.35,
        'pressure_profile_alpha': 1.,
        'pressure_profile_gamma': -0.3,

        # Simple instrumental model
        'beam_fwhm_arcmin': 1.4,
        'noise_rms_muk_arcmin': 18.0,
        
        # CMB power spectrum
        'cmb_ell': cmb_ell,
        'cmb_c_l': cmb_c_l

        }
    #}}}
    cosmology_parameters['h'] = cosmology_parameters['H0'] / 100.0
    cosmology_parameters['OL0'] = 1.0 - cosmology_parameters['Om0'] #flat LCDM
    cosmology_parameters['Oc0'] = cosmology_parameters['Om0'] - \
                                  cosmology_parameters['Ob0'] # CDM density
    cosmology_parameters['rhoM'] = cosmology_parameters['Om0'] * 2.7753e11
    cosmology_parameters['As'] = sigma_8_to_a_s(grid_locs[ii, 0], \
                                                cosmology_parameters)

    # generate logarithmic mass grid
    numerical_parameters['map_logM_boundaries'] = np.linspace(
            numerical_parameters['map_logM_min'],
            numerical_parameters['map_logM_max'],
            num = numerical_parameters['map_Npoints_M'] + 1
            )
    
    # generate redshift grid
    numerical_parameters['map_z_boundaries'] = np.linspace(
            numerical_parameters['map_z_min'],
            numerical_parameters['map_z_max'],
            num = numerical_parameters['map_Npoints_z'] +1
            )

    # do physical tSZ calculations
    if numerical_parameters['do_physics'] :
        
        # add an astropy object
        cosmology_parameters['cosmo_object'] = FlatLambdaCDM(
                H0=cosmology_parameters['H0'] * u.km / u.s / u.Mpc,
                Tcmb0=cosmology_parameters['TCMB'] * u.K,
                Om0=cosmology_parameters['Om0'],
                Neff=cosmology_parameters['Neff'],
                m_nu=cosmology_parameters['Mnu'] * u.eV,
                Ob0=cosmology_parameters['Ob0'],
                name='my_cosmology'
                )

        # matter power spectrum parameters
        PK_params = {
                'zmin': 0. ,
                'zmax': numerical_parameters['map_z_max'] + 1.,
                'nz_step': 150,
                'kmax': numerical_parameters['k_max'] + 1.,
                'nonlinear': False,
                'hubble_units': True,
                'k_hunit': True,
                }

        # Create the Power Spectrum Interpolator with CAMB
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
        pars.InitPower.set_params(ns = cosmology_parameters['ns'], As = cosmology_parameters['As'], pivot_scalar = cosmology_parameters['pivot_scalar'])
        pars.NonLinear = model.NonLinear_none
        cosmology_parameters['PK'] = camb.get_matter_power_interpolator(
                pars,
                zmin = PK_params['zmin'],
                zmax = PK_params['zmax'],
                nz_step = PK_params['nz_step'],
                kmax = PK_params['kmax'],
                nonlinear = PK_params['nonlinear'],
                hubble_units = PK_params['hubble_units'],
                k_hunit = PK_params['k_hunit']
                ).P
        pars.set_matter_power(redshifts = [0.], kmax = 20.)
        results = camb.get_results(pars)
        cosmology_parameters['sigma8'] = results.get_sigma8()[0]

        '''
        # Generate CMB power spectrum
        pars.set_for_lmax(numerical_parameters['l_max'], \
                          lens_potential_accuracy=1)
        results = camb.get_results(pars)
        ell = np.linspace(0, numerical_parameters['l_max'], \
                          numerical_parameters['l_max'] + 1, dtype=int)
        c_l = results.get_cmb_power_spectra(pars, \
                                            lmax=numerical_parameters['l_max'], \
                                            spectra=['total'], \
                                            CMB_unit='muK', \
                                            raw_cl=True)['total'][:, 0]
        np.savetxt(path + '/cmb_c_l_{:d}.txt'.format(ii), \
                   np.column_stack((ell, c_l)))
        '''

        # Colossus cosmology
        # @TODO: can we remove?
        colossus_params = {
                'flat': True,
                'H0': cosmology_parameters['H0'],
                'Om0': cosmology_parameters['Om0'],
                'Ob0': cosmology_parameters['Ob0'],
                'sigma8': cosmology_parameters['sigma8'],
                'ns': cosmology_parameters['ns'],
                'de_model': 'w0',
                'w0': cosmology_parameters['w'],
                'Tcmb0': cosmology_parameters['TCMB'],
                'Neff': cosmology_parameters['Neff']
                }
        colossus_cosmology.setCosmology('cosmo_object', colossus_params)

        # generate the 2D number density of sources
        dndOmega = pfunc.map_generate_dndOmega(numerical_parameters, \
                                               cosmology_parameters)
        np.savez(path + '/dndOmega_{:d}.npz'.format(ii), \
                 dndOmega=dndOmega)
        
        # generate the y-profiles
        thetas, yprofiles = \
            pfunc.map_generate_yprofiles(numerical_parameters, \
                                         cosmology_parameters)
        np.savez(path + '/yprofiles_{:d}.npz'.format(ii), \
                 thetas=thetas, yprofiles=yprofiles)


    # produce maps and / or histograms
    if numerical_parameters['do_maps'] or numerical_parameters['do_hists']:

        # Compton-y --> Temperature conversion parameter
        cosmology_parameters['gnu'] = \
            mfunc.g(cosmology_parameters['hPlanck'] * \
                    cosmology_parameters['frequency'] / \
                    cosmology_parameters['kBoltzmann'] / \
                    cosmology_parameters['TCMB'])

        # read in tSZ calculations
        f = np.load(path + '/dndOmega_{:d}.npz'.format(ii))
        dndOmega = f['dndOmega']
        f = np.load(path + '/yprofiles_{:d}.npz'.format(ii))
        thetas = f['thetas']
        yprofiles = f['yprofiles']

        # loop over patches per realisation
        tot_maps = []
        for jj in xrange(numerical_parameters['n_patch']) :
            
            # report progress if desired
            if numerical_parameters['verbose'] :
                print 'I am in index = ' + str(ii)
            
            # generate a single tSZ map
            # Note: the runtime of this function is dominated by the 
            #       mfunc.throw_clusters function, which is just array
            #       manipulation.
            #       The usage of numba.jit is very helpful in keeping 
            #       runtime reasonable.
            stub = '_{:d}_{:d}'.format(ii, jj)
            if numerical_parameters['do_maps']:
                sz_map = mfunc.map_generate_final_map(numerical_parameters, \
                                                      cosmology_parameters, \
                                                      dndOmega, thetas, \
                                                      yprofiles, wcss[jj])
                enmap.write_fits(path + '/tsz_map' + stub + '.fits', sz_map)
            else:
                sz_map = enmap.read_map(path + '/tsz_map' + stub + '.fits')

            # generates a noise map of the same shape as the final map
            if numerical_parameters['do_maps']:
                noise_map = \
                    mfunc.map_generate_random_noise(numerical_parameters, \
                                                    cosmology_parameters, \
                                                    wcss[jj])
                enmap.write_fits(path + '/noise_map' + stub + '.fits', noise_map)
            else:
                noise_map = enmap.read_map(path + '/noise_map' + stub + '.fits')
            
            # combine components, rescaling by relevant parameters
            tot_maps.append(enmap.enmap(sz_map * grid_locs[ii, 2] + \
                                        noise_map * grid_locs[ii, 3], \
                                        wcs=wcss[jj]))

        # apply ACT masks and filters, and histogram!
        if numerical_parameters['do_hists']:
            pdf = mfunc.map_act_hist(tot_maps, numerical_parameters, \
                                     cosmology_parameters, wf, masks, \
                                     apo_masks, negbins, posbins)
            np.savetxt(path + '/combined_hist_{:d}.txt'.format(ii), pdf)

