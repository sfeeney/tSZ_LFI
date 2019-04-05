import numpy as np
import os

# parameters specifying the numerical behaviour of the code #{{{
numerical_parameters = {

        # set this to True if you want to generate dndOmega and the y-profiles
        # then some non-standard packages will be required
        'do_physics': True,
        
        
        # set this to true if you have already generated dndOmega and the y-profiles
        # then only maps can be generated using this data
        'do_maps': True,
        
        # do you want the code to pester you with printouts?
        'verbose': True,
        
        # angular sidelength of the final map in radians
        'map_size': 20.*np.pi/180,

        # the code takes only a fraction of the final map in order to get rid of edge effects.
        #   I didn't find an efficient way to implement period boundary conditions.
        #   0.8 should be small enough, larger values will probably work as well
        'map_fraction': 0.8,

        # pixel sidelength in radians
        'map_pixel_size': 0.5/60.*np.pi/180,

        # parameters for the mass grid
        #   for the mass definition see the cosmology parameters below
        #   NOTE : all units are "h-units", i.e. [M] = M_sun/h etc.
        'map_logM_min': 11.,
        'map_logM_max': 16.,
        'map_Npoints_M': 50,
        
        # parameters for the redshift grid
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
        'map_Poisson': False,

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
        'noise_power_spectrum_file': 'TODO',

        # some integration boundaries, should not need any changing
        'k_max': 100,
        'k_min': 1e-10

        }
#}}}

if numerical_parameters['do_physics'] :
    import physics_functions
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    import camb
    from camb import model, initialpower, get_matter_power_interpolator
    from colossus.cosmology import cosmology as colossus_cosmology
if numerical_parameters['do_maps'] :
    import map_functions
    from scipy.interpolate import interp1d

# cosmology parameters #{{{
cosmology_parameters = {
        
        # Standard cosmology parameters
        'H0': 70.,# km/s/Mpc
        'h': 0.7,# "little h" if needed
        'Om0': 0.25,# dimensionless (total) matter density
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
        'pressure_profile_gamma': -0.3

        }
#}}}

# Run this whenever parameters are updated
def computations() :#{{{
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

    cosmology_parameters['OL0'] =  1.-cosmology_parameters['Om0'] #flat LCDM
    cosmology_parameters['Oc0'] = cosmology_parameters['Om0'] - cosmology_parameters['Ob0'] # CDM density
    cosmology_parameters['rhoM'] = cosmology_parameters['Om0'] * 2.7753e11 # * cosmology_parameters['h']**2 # M_sun * Mpc^-3
    if numerical_parameters['do_maps'] :

        # Compton-y --> Temperature conversion parameter
        cosmology_parameters['gnu'] = map_functions.g(cosmology_parameters['hPlanck']*cosmology_parameters['frequency']/cosmology_parameters['kBoltzmann']/cosmology_parameters['TCMB'])

        # read in power noise power spectrum and turn into interpolator
        # I'm assuming these files to loook similar to those Colin linked to,
        # i.e. the first column contains ell, and some other column contains
        # noise power in Compton-y^2
        ell, Cell = np.loadtxt(numerical_parameters['noise_power_spectrum_file'], unpack = True, usecols = (0,1), comments = '#')
        Cell_interpolator = interp1d(ell, Cell, bounds_error = False, fill_value = 0.)
        cosmology_parameters['noise_power'] = Cell_interpolator


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
        # sigma8 is a useful check that everything went well
        print('sigma8 = ' + str(cosmology_parameters['sigma8']))

        # Colossus cosmology
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
#}}}


#######################
#### Usage example ####
#######################
path = 'test_maps' # the path where the maps will be stored
os.system('mkdir ' + path)

computations()

if numerical_parameters['do_physics'] :

    physics_functions.map_generate_dndOmega(numerical_parameters, cosmology_parameters, path)
    # generates the 2D number density of sources and stores in 'path/dndOmega.npz'

    physics_functions.map_generate_yprofiles(numerical_parameters, cosmology_parameters, path)
    # generates the y-profiles and stores in 'path/yprofiles.npz'

if numerical_parameters['do_maps'] :

    Nmaps = 1 # how many maps you want to generate
    for ii in xrange(Nmaps) :
        map_functions.map_generate_final_map(numerical_parameters, cosmology_parameters, path, ii)
        # generates a single map and stores as 'final_map_ii.npz' in 'path'
        # the .npz file contains a single square array, where each entry is a temperature in uK
        #
        # Note: the dndOmega and y-profile files need to exist in the same 'path' for this function to use them
        #
        # Note: the runtime of this function is dominated by the map_functions.throw_clusters function, which
        #       is just array manipulation.
        #       The usage of numba.jit is very helpful in keeping runtime reasonable.

        map_functions.map_generate_random_noise(numerical_parameters, cosmology_parameters, path, ii)
        # generates a noise map of the same shape as the final map, at 'path/noise_ii.npz'
        #   the noise power spectrum is read from the file 
        #       numerical_parameters['noise_power_spectrum_file']
        # Currently, the noise is not automatically added to the final map,
        # this would allow to look at the effects of different noise levels.
