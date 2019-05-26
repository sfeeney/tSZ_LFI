import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from numba import prange
from warnings import warn
import os.path
from scipy.integrate import quad
from time import time
from scipy.special import jv
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import mass_adv
from colossus.halo.profile_nfw import NFWProfile
from colossus.halo.concentration import concentration
from colossus.lss import bias as colossus_bias
import camb
from camb import model, initialpower, get_matter_power_interpolator
from scipy.integrate import simps
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
from scipy.special import jn_zeros
from scipy.special import sici
from scipy.interpolate import CubicSpline
from scipy.stats import moment
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import RectBivariateSpline
from pygsl import hankel as GSL_DHT

# TODO : del all unused variables as soon as possible
# TODO : write the save statements, as soon as the filename issue is solved
# TODO : add checks about sizes etc at various places!

def _set_param(input_dict, name, default) :
    return input_dict[name] if name in input_dict else default

class numerics(object) :#{{{
    # TODO : this should have fields with path specifiers
    # they should normally not be touched, but occasionally that might be useful
    # if two different convolutions / ... are wanted
    def __init__(self, numerics_dict={None}) :
        self.debugging = _set_param(numerics_dict, 'debugging', False)

        # grid parameters
        self.Npoints_M = _set_param(numerics_dict, 'Npoints_M', 50)
        self.Npoints_z = _set_param(numerics_dict, 'Npoints_z', 51)
        self.logM_min = _set_param(numerics_dict, 'logM_min', 12.)
        self.logM_max = _set_param(numerics_dict, 'logM_max', 16.)
        self.z_min = _set_param(numerics_dict, 'z_min', 0.005)
        self.Npoints_small_M = _set_param(numerics_dict, 'Npoints_small_M', 50)
        self.small_logM_min = _set_param(numerics_dict, 'small_logM_min', 1.)
        self.ell_min = _set_param(numerics_dict, 'ell_min', 2)
        self.ell_max = _set_param(numerics_dict, 'ell_max', 100000)
        self.N_ell = _set_param(numerics_dict, 'N_ell', 1000)

        self.signal_type = _set_param(numerics_dict, 'signal_type', None)
        if self.signal_type is None :
            raise RuntimeError('You need to specify a signal type in the numerics input dict.')
        if (self.signal_type is not 'tSZ') and (self.signal_type is not 'kappa') :
            raise RuntimeError('Signal type must be either tSZ or kappa.')

        if self.signal_type is 'kappa' :
            self.z_source = _set_param(numerics_dict, 'z_source', None)
            if self.z_source is None :
                raise RuntimeError('You have requested weak lensing without specifying a source redshift.')
            self.z_max = self.z_source - 1e-3 # get rid of spurious divergences
        elif self.signal_type is 'tSZ' :
            self.z_max = _set_param(numerics_dict, 'z_max', 6.)
        self.Npoints_signal = _set_param(numerics_dict, 'Npoints_signal', 2**17)
        self.signal_min = _set_param(numerics_dict, 'signal_min', 0.)
        if self.signal_type is 'kappa' :
            self.signal_max = _set_param(numerics_dict, 'signal_max', 5.)
        elif self.signal_type is 'tSZ' :
            self.signal_max = _set_param(numerics_dict, 'signal_max', 300e-6)

        self.Npoints_theta = _set_param(numerics_dict, 'Npoints_theta', 200)

        self.pixel_radius = _set_param(numerics_dict, 'pixel_radius', None)
        self.pixel_sidelength = _set_param(numerics_dict, 'pixel_sidelength', None)
        self.Wiener_filter = _set_param(numerics_dict, 'Wiener_filter', None)
        self.empirical_bl = _set_param(numerics_dict, 'empirical_bl', None)
        self.empirical_bl_ell = _set_param(numerics_dict, 'empirical_bl_ell', None)
        self.gaussian_kernel_FWHM = _set_param(numerics_dict, 'gaussian_kernel_FWHM', None)
        self.physical_smoothing_scale = _set_param(numerics_dict, 'physical_smoothing_scale', None)
        if self.physical_smoothing_scale is not None :
            self.smoothing_Npoints_scale = _set_param(numerics_dict, 'smoothing_Npoints_scale', 5.)
            # takes this number times Npoints theta
        
        self.sigma_chi_file = _set_param(numerics_dict, 'sigma_chi_file', None)

        self.__create_grids()
    def __create_grids(self) :
        # add the various grids
        self.lambda_grid = 2.*np.pi*np.fft.rfftfreq(
            self.Npoints_signal,
            d = 2.*(self.signal_max-self.signal_min)/float(self.Npoints_signal)
            )
        self.signal_grid = np.linspace(
            self.signal_min,
            self.signal_max,
            num = self.Npoints_signal
            )
        self.logM_grid = np.linspace(
            self.logM_min,
            self.logM_max,
            num = self.Npoints_M
            )
        self.z_grid = np.linspace(
            self.z_min,
            self.z_max,
            num = self.Npoints_z
            )
        self.small_logM_grid = np.linspace(
            self.small_logM_min,
            self.logM_min,
            num = self.Npoints_small_M
            )
        self.larr = 10.**np.linspace(
            np.log10(self.ell_min),
            np.log10(self.ell_max),
            num = self.N_ell
            )
        j_0_n = jn_zeros(0, self.Npoints_theta)
        self.scaled_real_theta_grid = j_0_n/j_0_n[-1] # normalized to unity
        self.scaled_reci_theta_grid = j_0_n # normalized to unity
        if (self.empirical_bl is not None) and (self.empirical_bl_ell is not None) :
            empirical_bl_interp = interp1d(
                self.empirical_bl_ell,
                self.empirical_bl,
                kind = 'quadratic',
                bounds_error = False, 
                fill_value = 1.
                )
            self.empirical_bl_fct = lambda ell : empirical_bl_interp(ell)
        else :
            self.empirical_bl_fct = None
#}}}

class cosmology(object) :#{{{
    # some fundamental constants
    c0 = 2.99792458e5 # km/s
    GN = 4.30091e-9 # Mpc/Msun*(km/s)**2
    delta_c = 1.686
    hPl = 6.62607004e-34 # SI
    kBoltzmann = 1.38064852e-23 # SI
    def __init__(self, cosmo_dict = {None}) :#{{{
        # basic cosmology parameters
        self.h = _set_param(cosmo_dict, 'h', 0.7)
        self.Om = _set_param(cosmo_dict, 'Om', 0.3)
        self.Ob = _set_param(cosmo_dict, 'Ob', 0.046)
        self.As = _set_param(cosmo_dict, 'As', 2.1e-9)
        self.pivot_scalar = _set_param(cosmo_dict, 'pivot_scalar', 0.05/self.h)
        self.w = _set_param(cosmo_dict, 'w', -1)
        self.ns = _set_param(cosmo_dict, 'ns', 0.97)
        self.Mnu = _set_param(cosmo_dict, 'Mnu', 0.)
        self.Neff = _set_param(cosmo_dict, 'Neff', 0.)
        self.TCMB = _set_param(cosmo_dict, 'TCMB', 2.726)

        # various definitions, should hopefully not need much change
        self.mass_def_initial = _set_param(cosmo_dict, 'mass_def_initial', '200m')
        self.mass_def_kappa_profile = _set_param(cosmo_dict, 'mass_def_profile', 'vir')
        self.mass_def_Tinker = _set_param(cosmo_dict, 'mass_def_Tinker', '200m')
        self.mass_def_Batt = _set_param(cosmo_dict, 'mass_def_Batt', '200c')
        self.r_out_def = _set_param(cosmo_dict, 'r_out_def', 'vir')
        self.r_out_scale = _set_param(cosmo_dict, 'r_out_scale', 2.5)
        self.concentration_model = _set_param(cosmo_dict, 'concentration_model', 'duffy08')
        self.small_mass_concentration_model = _set_param(cosmo_dict, 'small_mass_concentration_model', 'diemer19')
        self.halo_profile = _set_param(cosmo_dict, 'halo_profile', 'nfw')
        self.HMF_fuction = _set_param(cosmo_dict, 'HMF_function', 'Tinker10')

        # derived quantities
        self.OL = 1.-self.Om # dark energy
        self.Oc = self.Om - self.Ob # (cold) dark matter
        self.H0 = 100.*self.h
        self.rhoM = self.Om*2.7753e11

        # possibly, the user wants to use an external HMF
        self.external_HMF_file = _set_param(cosmo_dict, 'external_HMF_file', None)
         
        # astropy (adds self.astropy_cosmo object)
        #self.__initialize_astropy()
        
        # run CAMB (adds self.PK which is a power spectrum interpolator
        #self.__initialize_CAMB()
         
        # initialize Colossus cosmology
        #self._initialize_colossus()

        self.bias_arr = None
        self.hmf_arr = None
        self.d2_arr = None
        self.small_mass_hmf_arr = None
        self.small_mass_bias_arr = None
        self.my_angular_diameter_distance = None
        self.my_angular_diameter_distance_z1z2 = None
        self.my_H = None
        self.my_comoving_distance = None
        self.astropy_cosmo = None

        self.initialized_colossus = False

        self.my_D = None
        self.my_k_integral = None
        self.my_PK = None
        self.my_PK_psi = None
        self.sigma8 = None

    #}}}
    def __initialize_astropy(self) :#{{{
        # adds an astropy object
        print 'Initializing astropy.'
        self.astropy_cosmo = FlatLambdaCDM(
            H0 = self.H0*u.km/u.s/u.Mpc,
            Tcmb0 = self.TCMB*u.K,
            Om0 = self.Om,
            Neff = self.Neff,
            m_nu = self.Mnu*u.eV,
            Ob0 = self.Ob,
            name = 'my cosmology'
            )
    @property
    def angular_diameter_distance(self) :
        if self.astropy_cosmo is None :
            self.__initialize_astropy()
        if self.my_angular_diameter_distance is None :
            self.my_angular_diameter_distance = lambda z: (self.astropy_cosmo.angular_diameter_distance(z)).value*self.h
        return self.my_angular_diameter_distance
    @property
    def angular_diameter_distance_z1z2(self) :
        if self.astropy_cosmo is None :
            self.__initialize_astropy()
        if self.my_angular_diameter_distance_z1z2 is None :
            self.my_angular_diameter_distance_z1z2 = lambda z1,z2: (self.astropy_cosmo.angular_diameter_distance_z1z2(z1,z2)).value*self.h
        return self.my_angular_diameter_distance_z1z2
    @property
    def H(self) :
        if self.astropy_cosmo is None :
            self.__initialize_astropy()
        if self.my_H is None :
            self.my_H = lambda z: (self.astropy_cosmo.H(z)).value/self.h
        return self.my_H
    @property
    def comoving_distance(self) :
        if self.astropy_cosmo is None :
            self.__initialize_astropy()
        if self.my_comoving_distance is None :
            self.my_comoving_distance = lambda z: (self.astropy_cosmo.comoving_distance(z)).value*self.h
        return self.my_comoving_distance
    #}}}
    def __initialize_CAMB(self) :#{{{
        # adds linear matter power interpolator
        print 'Initializing CAMB.'
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0 = self.H0,
            ombh2 = self.Ob*self.h**2.,
            omch2 = self.Oc*self.h**2.,
            omk = 0.,
            mnu = self.Mnu,
            standard_neutrino_neff = self.Neff,
            nnu = self.Neff,
            TCMB = self.TCMB
            )
        pars.InitPower.set_params(
            ns = self.ns,
            As = self.As,
            pivot_scalar = self.pivot_scalar
            )
        pars.NonLinear = model.NonLinear_none
#        print(pars)
        self.my_PK = get_matter_power_interpolator(
            pars,
            zmin = 0.,
            zmax = 6.,
            nz_step = 150,
            kmax = 101.,
            nonlinear = False,
            hubble_units = True,
            k_hunit = True 
            ).P
        pars.set_matter_power(redshifts = [0.], kmax = 20.)
        results = camb.get_results(pars)
        self.sigma8 = results.get_sigma8()[0]
        print 'sigma8 = ' + str(self.sigma8)
    def PK(self, z, k) :
        if self.my_PK is None :
            self.__initialize_CAMB()
        return self.my_PK(z, k)
    def __initialize_nonlin_CAMB(self) :
        print 'Initializing nonlinear CAMB.'
        parsnl = camb.CAMBparams()
        parsnl.set_cosmology(
            H0 = self.H0,
            ombh2 = self.Ob*self.h**2.,
            omch2 = self.Oc*self.h**2.,
            omk = 0.,
            mnu = self.Mnu,
            standard_neutrino_neff = self.Neff,
            nnu = self.Neff,
            TCMB = self.TCMB
            )
        parsnl.InitPower.set_params(
            ns = self.ns,
            As = self.As,
            pivot_scalar = self.pivot_scalar
            )
#        parsnl.NonLinear = model.NonLinear_both
        parsnl.set_matter_power(redshifts = [0.], kmax = 101.)
        self.my_PK_psi = get_matter_power_interpolator(
            parsnl,
            zmin = 0.,
            zmax = 1100.,
            nz_step = 150,
            kmax = 101.,
            nonlinear = True,
            hubble_units = True,
            k_hunit = True,
            var1 = model.Transfer_Weyl,
            var2 = model.Transfer_Weyl
            ).P    
    def PK_psi(self, z, k) :
        if self.my_PK_psi is None :
            self.__initialize_nonlin_CAMB()
        return self.my_PK_psi(z, k)
    def D(self, z) :
        if self.my_PK is None :
            self.__initialize_CAMB()
        if self.my_D is None :
            self.my_D = lambda z: np.sqrt(self.PK(z,0.01)/self.PK(0.,0.01)) # growth function
        return self.my_D(z)
    def k_integral(self) :
        if self.my_PK is None :
            self.__initialize_CAMB()
        if self.my_k_integral is None :
            integrand = lambda k: (1./(2.*np.pi))*k*self.PK(0., k)
            self.my_k_integral, _ = quad(integrand, 1.e-10, 100., limit = 100)
        return self.my_k_integral
    #}}}
    def _initialize_colossus(self) :#{{{
        # adds colossus object
        print 'Initializing Colossus.'
        if self.sigma8 is None :
            self.__initialize_CAMB()
        colossus_cosmology.setCosmology(
            'my cosmology',
            {
            'flat': True,
            'H0': self.H0,
            'Om0': self.Om,
            'Ob0': self.Ob,
            'sigma8': self.sigma8,
            'ns': self.ns,
            'de_model': 'w0',
            'w0': self.w,
            'Tcmb0': self.TCMB,
            'Neff': self.Neff
            }
            )
        self.initialized_colossus = True
    #}}}
    def virial_radius(self, M, z, massdef) :#{{{
        if not self.initialized_colossus :
            self._initialize_colossus()
        _ , rvir, _ = mass_adv.changeMassDefinitionCModel(
            M,
            z,
            massdef,
            'vir',
            profile = self.halo_profile,
            c_model = self.concentration_model
            )
        return rvir * 1e-3 # Mpc/h
    #}}}
    def virial_radius_small_masses(self, M, z, massdef) :#{{{
        if not self.initialized_colossus :
            self._initialize_colossus()
        _ , rvir, _ = mass_adv.changeMassDefinitionCModel(
            M,
            z,
            massdef,
            'vir',
            profile = self.halo_profile,
            c_model = self.small_mass_concentration_model
            )
        return rvir * 1e-3 # Mpc/h
    #}}}
    def convert_mass(self, M, z, massdefin, massdefout) :#{{{
        if massdefin == massdefout :
            return M
        if not self.initialized_colossus :
            self._initialize_colossus()
        Mout, _, _ = mass_adv.changeMassDefinitionCModel(
            M,
            z,
            massdefin,
            massdefout,
            profile = self.halo_profile,
            c_model = self.concentration_model
            )
        return Mout
    #}}}
    @staticmethod
    @jit(nopython = True)
    def __hmf_Tinker2010(nu, z) :#{{{
        # Eq 8 of Tinker10, parameters from Table 4
        z1 = z
        if z1 > 3. : z1 = 3
        # HMF only calibrated below z = 3, use the value for z = 3 at higher redshifts
        beta = 0.589 * (1. + z1)**(0.20)
        phi = -0.729 * (1. + z1)**(-0.08)
        eta = -0.243 * (1. + z1)**(0.27)
        gamma = 0.864 * (1. + z1)**(-0.01)
        alpha = 0.368
        return alpha * ( 1. + (beta*nu)**(-2.*phi)  ) * nu**(2.*eta) * np.exp(-0.5*gamma*nu**2)
    #}}}
    @staticmethod
    @jit(nopython = True)
    def __hmf_Tinker2008(sigma) :#{{{
        B = 0.482
        d = 1.97
        e = 1.
        f = 0.51
        g = 1.228
        return B*((sigma/e)**(-d) + sigma**(-f)) * np.exp(-g/sigma**2.)
    #}}}
    @staticmethod
    @jit(nopython = True)
    def __hmf_ShethTormen(nu) :#{{{
        A = 0.3222
        a = 0.707
        p = 0.3
        return A*(2.*a/np.pi)**0.5*nu*(1.+(nu**2./a)**p)*np.exp(-a*nu**2./2.)
    #}}}
    @staticmethod
    @jit(nopython = True)
    def __bz_Tinker2010(nu) :#{{{
        # Eq 6 of Tinker10, with parameters from Table 2
        Delta = 200.
        y = np.log10(Delta)
        A = 1.0 + 0.24 * y * np.exp( - (4./y)**4 )
        a = 0.44 * y - 0.88
        B  =0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp( - (4./y)**4 )
        c = 2.4
        return 1. - A*nu**a/(nu**a + 1.686**a) + B * nu**b + C * nu**c
    #}}}
    @staticmethod
    @jit(nopython = True)
    def __window_function_FT(x) :#{{{
        return 3.*(np.sin(x) - x*np.cos(x))/x**3.
    #}}}
    @staticmethod
    def __chi(x) :#{{{
        return ( (x**2 - 3.)*np.sin(x) + 3.*x*np.cos(x) ) / x**3
    #}}}
    def __sigma(self, M, z) :#{{{
        RM = (3. * M / (4. * np.pi * self.rhoM ) )**(1./3.)
        integrand = lambda k : (1./(2.*np.pi**2))*( k**2 * self.PK(z, k) * (cosmology.__window_function_FT(k*RM))**2 )
        sigmasq,_ = quad(integrand, 1e-10, 100., limit = 100)
        return np.sqrt(sigmasq)
    #}}}
    def __chi_integral(self, M, z) :#{{{
        # computes the chi-integral [which is close to the derivative of sigma]
        RM = (3. * M / (4. * np.pi * self.rhoM ) )**(1./3.)
        integrand = lambda lk : (1.*np.log(10.)/np.pi**2)*( (10.**(lk))**3 * self.PK(z, (10.**lk)) * cosmology.__window_function_FT((10.**lk)*RM) * cosmology.__chi((10.**lk)*RM) )
        integral,_ = quad(integrand, -10., 2., limit = 100)
        return integral
    #}}}
    def dndM(self, M, z, s, chi_int, HMF_fuction) :#{{{
        if HMF_fuction is 'Tinker10' :
            f = cosmology.__hmf_Tinker2010(cosmology.delta_c/s, z)
            return -cosmology.delta_c*self.rhoM*f*chi_int/(2.*s**3.*M**2.)
        elif HMF_fuction is 'Tinker08' :
            g = cosmology.__hmf_Tinker2008(s)
            return -self.rhoM*g*chi_int/(2.*s**2.*M**2.)
        elif HMF_fuction is 'ShethTormen' :
            g = cosmology.__hmf_ShethTormen(cosmology.delta_c/s)
            return -self.rhoM*g*chi_int/(2.*s**2.*M**2.)
        else :
            raise RuntimeError('Unknown HMF function in cosmology.dndM')
    #}}}
    def create_HMF_and_bias(self, path, numerics, pardict = {None}) :#{{{
        # throws error if HMF_and_bias already exists in path
        do_d2 = _set_param(pardict, 'do_d2', True)
        Npoints_M = numerics.Npoints_M
        Npoints_z = numerics.Npoints_z
        logM_grid = numerics.logM_grid
        z_grid = numerics.z_grid
        self.hmf_arr = np.empty((Npoints_M, Npoints_z))
        self.d2_arr = np.empty((Npoints_M, Npoints_z))
        self.bias_arr = np.empty((Npoints_M, Npoints_z))
        self.small_mass_hmf_arr = np.empty((numerics.Npoints_small_M, numerics.Npoints_z))
        self.small_mass_bias_arr = np.empty((numerics.Npoints_small_M, numerics.Npoints_z))
        #### if sigma and chi_int don't exist, create them, otherwise extrapolate ####
        if numerics.sigma_chi_file is None :
            raise RuntimeError('Provide path for sigma / chi_int file.')
        if os.path.isfile(numerics.sigma_chi_file) :
            print 'Reading sigma and chi_int from file.'
            f = np.load(numerics.sigma_chi_file)
            interp_Mdef = f['Mdef']
            if interp_Mdef != self.mass_def_initial :
                raise RuntimeError('Please recompute the sigma and chi_int file with your new mass definition.')
            interp_logM = f['logM']
            if (min(logM_grid)<min(interp_logM)) or (max(logM_grid)>max(interp_logM)) :
                raise RuntimeError('Your mass grid extends too far. Recompute sigma and chi_int.')
            interp_z = f['z']
            if (min(z_grid)<min(interp_z)) or (max(z_grid)>max(interp_z)) :
                raise RuntimeError('Your redshift grid extends too far. Recompute sigma and chi_int.')
            interp_s = f['sigma'].item()
            interp_chi = f['chi_int'].item()
        else :
            print 'Computing the interpolators for sigma and chi_int.'
            interp_logM = np.linspace(1., 18., num = 80)
            interp_z = np.linspace(0.001, 10., num = 40)
            fine_sigma_arr = np.empty((len(interp_logM), len(interp_z)))
            fine_chi_int_arr = np.empty((len(interp_logM), len(interp_z)))
            for ii in xrange(len(interp_logM)) :
                start = time()
                for jj in xrange(len(interp_z)) :
                    z = interp_z[jj]
                    M = self.convert_mass(10.**interp_logM[ii], z, self.mass_def_initial, self.mass_def_Tinker)
                    fine_sigma_arr[ii,jj] = self.__sigma(M, interp_z[jj])
                    fine_chi_int_arr[ii,jj] = self.__chi_integral(M, interp_z[jj])
                end = time()
                print str((end-start)/60.*(len(interp_logM)-ii)) + ' minutes remaining in interpolating sigma and chi_int.'
            interp_s = RectBivariateSpline(
                interp_logM,
                interp_z,
                fine_sigma_arr
                )
            interp_chi = RectBivariateSpline(
                interp_logM,
                interp_z,
                fine_chi_int_arr
                )
            np.savez(
                numerics.sigma_chi_file,
                logM = interp_logM,
                z = interp_z,
                sigma = interp_s,
                chi_int = interp_chi,
                Mdef = self.mass_def_initial
                )
        sigma_arr = interp_s(logM_grid, z_grid)
        chi_int_arr = interp_chi(logM_grid, z_grid)
        small_mass_sigma_arr = interp_s(numerics.small_logM_grid, z_grid)
        small_mass_chi_int_arr = interp_chi(numerics.small_logM_grid, z_grid)
        ####
        if os.path.isfile(path + 'HMF_and_bias.npz') and not numerics.debugging :
            if self.external_HMF_file is None :
                raise RuntimeError('HMF_and_bias already exists.')
        if self.external_HMF_file is not None :
            warn('Assuming you want to overwrite the HMF from an external file. Will keep the bias.', UserWarning)
            if os.path.isfile(path + 'HMF_and_bias.npz') :
                warn('Found existing file. Reading bias from there.', UserWarning)
                f = np.load(path + 'HMF_and_bias.npz')
                self.bias_arr = f['bias']
            else :
                for ii in xrange(Npoints_M) :
                    for jj in xrange(Npoints_z) :
                        self.bias_arr[ii,jj] = cosmology.__bz_Tinker2010(cosmology.delta_c/sigma_arr[ii,jj])
            f = np.load(self.external_HMF_file)
            external_hmf = f['hmf']
            external_logM_vir = f['logMvir']
            external_z_arr = f['z']
            mass_interpolated_hmf = np.empty((Npoints_M, external_hmf.shape[1]))
            for ii in xrange(len(external_z_arr)) :
                logM_arr = np.log10(self.convert_mass(10.**external_logM_vir, external_z_arr[ii], 'vir', self.mass_def_initial))
                # Jacobian = dM200m/dMvir
                Jacobian = (self.convert_mass(10.**external_logM_vir+1e2, external_z_arr[ii], 'vir', self.mass_def_initial)-self.convert_mass(10.**external_logM_vir-1e2, external_z_arr[ii], 'vir', self.mass_def_initial))/(2*1e2)
                hmf_my_units = external_hmf[:,ii]/Jacobian
                hmf_my_units_interp = interp1d(logM_arr, hmf_my_units, kind = 'linear', bounds_error = False, fill_value = 0.)
                hmf_my_units_fct = lambda logM: hmf_my_units_interp(logM)
                mass_interpolated_hmf[:,ii] = hmf_my_units_fct(logM_grid)
            for ii in xrange(Npoints_M) :
                redshift_interpolated_hmf = interp1d(external_z_arr, mass_interpolated_hmf[ii,:], kind = 'linear', bounds_error = False, fill_value = 0.)
                redshift_interpolated_hmf_fct = lambda z: redshift_interpolated_hmf(z)
                self.hmf_arr[ii,:] = redshift_interpolated_hmf_fct(z_grid)
        # computes HMF and bias for the given cosmology
        else :
            def d2_integrand(logM, z) :
                mass = self.convert_mass(10.**logM,z, self.mass_def_initial, self.mass_def_Tinker)
                sigma = np.nan_to_num(interp_s(logM, z))
                chi_int = np.nan_to_num(interp_chi(logM, z))
                vol_fact = self.comoving_distance(z)**2./(self.H(z)/self.c0)
                return np.log(10.) * 10.**logM * vol_fact * self.dndM(mass, z, sigma, chi_int, self.HMF_fuction)
            for ii in xrange(Npoints_M) :
                print ii
                if ii == 0 :
                    logM_lo = logM_grid[0] - 0.5*(logM_grid[1]-logM_grid[0])
                else :
                    logM_lo = 0.5*(logM_grid[ii]+logM_grid[ii-1])
                if ii == Npoints_M-1 :
                    logM_hi = logM_grid[-1] + 0.5*(logM_grid[-1]-logM_grid[-2])
                else :
                    logM_hi = 0.5*(logM_grid[ii]+logM_grid[ii+1])
                for jj in xrange(Npoints_z) :
                    z = z_grid[jj]
                    M = self.convert_mass(10.**logM_grid[ii], z, self.mass_def_initial, self.mass_def_Tinker)
                    self.hmf_arr[ii,jj] = self.dndM(M, z, sigma_arr[ii,jj], chi_int_arr[ii,jj], self.HMF_fuction)
                    if jj == 0 :
                        z_lo = max([0.5*z_grid[0], z_grid[0] - 0.5*(z_grid[1]-z_grid[0])])
                    else :
                        z_lo = 0.5*(z_grid[jj]+z_grid[jj-1])
                    if jj == Npoints_z-1 :
                        z_hi = z_grid[-1] + 0.5*(z_grid[-1]-z_grid[-2])
                    else :
                        z_hi = 0.5*(z_grid[jj]+z_grid[jj+1])
                    if do_d2 :
                        self.d2_arr[ii,jj],_ = dblquad(
                            d2_integrand,
                            z_lo, z_hi,
                            logM_lo, logM_hi
                            )
                    self.bias_arr[ii,jj] = cosmology.__bz_Tinker2010(cosmology.delta_c/sigma_arr[ii,jj])
        for ii in xrange(numerics.Npoints_small_M) :
            for jj in xrange(numerics.Npoints_z) :
                z = z_grid[jj]
                M = self.convert_mass(10.**numerics.small_logM_grid[ii], z, self.mass_def_initial, self.mass_def_Tinker)
                self.small_mass_hmf_arr[ii,jj] = self.dndM(M, z, small_mass_sigma_arr[ii,jj], small_mass_chi_int_arr[ii,jj], 'ShethTormen')
                self.small_mass_bias_arr[ii,jj] = cosmology.__bz_Tinker2010(cosmology.delta_c/small_mass_sigma_arr[ii,jj])
#                if not self.initialized_colossus :
#                    self._initialize_colossus()
#                self.small_mass_bias_arr[ii,jj] = colossus_bias.haloBias(M, model = 'tinker10', z = z, mdef = self.mass_def_Tinker)
        np.savez(
            path + 'HMF_and_bias.npz',
            hmf = self.hmf_arr,
            d2 = self.d2_arr,
            bias = self.bias_arr,
            small_mass_hmf = self.small_mass_hmf_arr,
            small_mass_bias = self.small_mass_bias_arr
            )
    #}}}
#}}}

class profiles(object) :#{{{
    def __init__(self, cosmo, num, param_dict = {None}) :#{{{
        # checks whether path contains profiles, convolved_profiles, and tildes files

        self.cosmo = cosmo
        self.num = num
        # TODO : read in profile params
        # TODO : check whether tSZ/WL/...
        self.Sigma_crit = lambda z: 1.6625e18*cosmo.angular_diameter_distance(num.z_source)/cosmo.angular_diameter_distance(z)/cosmo.angular_diameter_distance_z1z2(z, num.z_source)
        self.theta_out_arr = None
        self.signal_arr = None
        self.convolved_signal_arr = None
        self.tilde_arr = None
        self.y_ell = None
    #}}}
    @staticmethod
    @jit(nopython = True)
    def _kappa_profile(theta, rho0, rhoM, rs, r_out, d_A, theta_out, critical_surface_density) :#{{{
        # this is the function that returns the value of a kappa profile
        # as a function of angle and other parameters
        R_proj = np.tan(theta)*d_A #define projected R in Mpc/h
        if (theta >= theta_out) :
            return 0. #kappa=0 outside halo boundary
        else :
            lin=0. # start integral at d_A*theta and multiply by 2 below
            lout=np.sqrt(r_out**2. - (np.tan(theta)*d_A)**2.) #integrate to cluster boundary as defined above
            # lout has units Mpc/h
            Jacobian = 1.
            if (R_proj > rs) : #case where d_A*tan(theta) > r_s
                return Jacobian*(rho0*2.*(rs**3./(2.*(r_out+rs)*((R_proj-rs)*(R_proj+rs))**1.5)*(np.pi*rs*(r_out+rs)+2.*np.sqrt((r_out-R_proj)*(r_out+R_proj)*(R_proj-rs)*(R_proj+rs))-2.*rs*(r_out+rs)*np.arctan2(R_proj**2.+r_out*rs,-1.*np.sqrt((r_out-R_proj)*(r_out+R_proj)*(R_proj-rs)*(R_proj+rs)))))-2.*(lout-lin)*rhoM) / critical_surface_density
            elif (R_proj == rs) : #case where d_A*tan(theta) = r_s
                return Jacobian*(rho0*2.*(np.sqrt(r_out-rs)*rs*(r_out+2.*rs)/(3.*(r_out+rs)**1.5))-2.0*(lout-lin)*rhoM) / critical_surface_density
            elif (R_proj < rs) : #case where d_A*tan(theta) < r_s
                return Jacobian*(rho0*2.*(rs**3./((r_out+rs)*(rs**2.-R_proj**2.)**1.5)*(-1.*np.sqrt((r_out-R_proj)*(r_out+R_proj)*(rs**2.-R_proj**2.))+rs*(r_out+rs)*np.log(R_proj*(r_out+rs)/(R_proj**2.+r_out*rs-np.sqrt((r_out-R_proj)*(r_out+R_proj)*(rs**2.-R_proj**2.))))))-2.*(lout-lin)*rhoM) / critical_surface_density
            else :
                raise RuntimeError('Problem in __kappa_profile.')
    #}}}
    @staticmethod
    @jit(nopython = True, parallel = True)
    def __kappa_interpolate_and_integrate(out_arr, rsample, rho_of_r, dAtant, rout) :#{{{
        for ii in prange(len(dAtant)-1) :
            lout = np.sqrt(rout**2.-dAtant[ii]**2.)
            larr = np.linspace(0., lout, 10000)
            rarr = np.sqrt(larr**2.+dAtant[ii]**2.)
            integrand = np.interp(
                rarr,
                rsample,
                rho_of_r
                )
            out_arr[ii] = np.trapz(integrand, x = larr)
    #}}}
    @staticmethod
    @jit(nopython = True, parallel = True)
    def __y_profile(theta_grid, M200c, d_A, r200c, r_out, y_norm, theta_out, h, z) :#{{{
        out_grid = np.empty(len(theta_grid))
        P0=18.1*(M200c/(1.e14*h))**(0.154)*(1.+z)**(-0.758)
        xc=0.497*(M200c/(1.e14*h))**(-0.00865)*(1.+z)**(0.731)
        beta=4.35*(M200c/(1.e14*h))**(0.0393)*(1.+z)**(0.415)
        alpha=1.
        gamma=-0.3
        Jacobian = 1./h # the integration is now over Mpc/h-l
        #integrand_arr = [lambda l: Jacobian*P0*(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**gamma*(1.+(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**alpha)**(-beta) for theta in theta_grid]
        lin = 0.
        lout = np.sqrt(r_out**2. - (np.tan(theta_grid)*d_A)**2.)
        for ii in prange(len(theta_grid)) :
            theta = theta_grid[ii]
            if (theta >= theta_out) :
            # added >= instead of > to avoid nans at integration boundary
                out_grid[ii] = 0. #y=0 outside cluster boundary
            else :
                # implement Battaglia fitting function for pressure profile and do the line-of-sight integral
                # Sep 27 --- removed h's here -- is this correct?
                integrand = lambda l : Jacobian*P0*(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**gamma*(1.+(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**alpha)**(-beta)
                #lin=0. # start integral at d_A*theta and multiply by 2 below
                #lout=np.sqrt(r_out**2. - (np.tan(theta)*d_A)**2.) #integrate to cluster boundary as defined above
                integration_grid = np.linspace(lin, lout[ii], 1000)
                integral = np.trapz(integrand(integration_grid), x = integration_grid)
#                ############
#                x = np.linspace(lin, lout, num = 1000.)
#                y = integrand(x)
#                plt.plot(x,y)
#                plt.show()
#                ############
                out_grid[ii] =  y_norm * integral * 2.
        return out_grid
    #}}}
    def __generate_profile(self, M, z, DHT_dict={None}) :#{{{
        # generates a single profile
        # M is assumed to be in the initial mass definition
        H = self.cosmo.H(z)
        d_A = self.cosmo.angular_diameter_distance(z)
        rvir = self.cosmo.virial_radius(M, z, self.cosmo.mass_def_initial)
        rhoc = 2.775e7*H**2.
        if self.cosmo.r_out_def is 'vir' :
            r_out = self.cosmo.r_out_scale * rvir
        else :
            raise RuntimeError('Your r_out_def is not implemented.')
        theta_out = np.arctan(r_out/d_A)

        theta_grid = self.num.scaled_real_theta_grid * theta_out

        if self.num.signal_type is 'kappa' :
            Mvir = self.cosmo.convert_mass(M, z, self.cosmo.mass_def_initial, self.cosmo.mass_def_kappa_profile)
            cvir = 5.72*(Mvir/1e14)**(-0.081)/(1.+z)**(0.71)
#            cvir *= 0.8 + 0.2*np.tanh((np.log10(Mvir)-14.)/2.)
            rs = rvir/cvir
            rho0 = Mvir/(4.*np.pi*rs**3.*(np.log(1.+cvir)-cvir/(1.+cvir)))
            rhoM = self.cosmo.astropy_cosmo.Om0*(1.+z)**3.*2.775e7/self.cosmo.h**2.
            critical_surface_density = self.Sigma_crit(z)
            if self.num.physical_smoothing_scale is None :
                signal_fct = lambda t: profiles._kappa_profile(
                    t,
                    rho0, rhoM, rs, r_out, d_A, theta_out, critical_surface_density
                    )
                signal_prof = map(signal_fct, theta_grid)
                #print (self.num.physical_smoothing_scale/d_A)*(180./np.pi)*60.
                #plt.plot(theta_grid*(180./np.pi)*60., signal_prof)
            else :
                # PHYSICAL SMOOTHING
                signal_prof = np.zeros(self.num.Npoints_theta)
                rs_scaled = rs/r_out
                smoothing_scaled = self.num.physical_smoothing_scale/r_out
                SI, CI = sici(DHT_dict['DHT_ksample']*rs_scaled)
                rho_NFW_of_k = rs_scaled**3.*(
                    -np.cos(DHT_dict['DHT_ksample']*rs_scaled)*CI
                    +np.sin(DHT_dict['DHT_ksample']*rs_scaled)*(0.5*np.pi-SI)
                    )
                Gaussian_of_k = np.exp(-0.5*(DHT_dict['DHT_ksample']*smoothing_scaled)**2.)
                integrand = np.sqrt(DHT_dict['DHT_ksample'])*rho_NFW_of_k*Gaussian_of_k
                _, rho_of_r = DHT_dict['DHTobj'].apply(integrand)
                rho_of_r *= 1./np.sqrt(DHT_dict['DHT_rsample'])
#                plt.loglog(DHT_dict['DHT_rsample'], rho_of_r*DHT_dict['DHT_ksample'][-1]**2.*r_out**3.*rho0)
#                plt.loglog(DHT_dict['DHT_rsample'], rho0/(DHT_dict['DHT_rsample']*r_out/rs)/(1.+DHT_dict['DHT_rsample']*r_out/rs)**2.)
#                plt.show()
                profiles.__kappa_interpolate_and_integrate(
                    signal_prof,
                    DHT_dict['DHT_rsample']*r_out,
                    rho_of_r,
                    d_A*np.tan(theta_grid),
                    r_out
                    )
                signal_prof *= 4.*DHT_dict['DHT_ksample'][-1]**2.*r_out**3.*rho0/critical_surface_density
                #plt.plot(theta_grid*(180./np.pi)*60., signal_prof)
                #plt.show()
                #plt.plot(theta_grid*theta_out*(180./np.pi)*60., signal_prof/signal_prof_new)
                #plt.show()
        elif self.num.signal_type is 'tSZ' :
            M200c = self.cosmo.convert_mass(M, z, self.cosmo.mass_def_initial, self.cosmo.mass_def_Batt)
            r200c = (3.*M200c/4./np.pi/200./rhoc)**0.333333333333
            P_norm = 2.61051e-18*(self.cosmo.Ob/self.cosmo.Om)*H**2.*M200c/r200c
            y_norm = 4.013e-6*P_norm*self.cosmo.h**2.
            #signal_fct = lambda t: profiles.__y_profile(
            #    t,
            #    M200c, d_A, r200c, r_out, y_norm, theta_out, self.cosmo.h, z
            #    )
            signal_prof = profiles.__y_profile(
                theta_grid,
                M200c, d_A, r200c, r_out, y_norm, theta_out, self.cosmo.h, z
                )
        else :
            raise RuntimeError('Unsupported signal type in __generate_profile.')
        return theta_out, signal_prof
    #}}}
    def create_profiles(self, path) :#{{{
        # throws error if profiles already exist in path
        if (os.path.isfile(path + 'profiles.npz') and not self.num.debugging) or ((self.signal_arr is not None) and (self.theta_out_arr is not None)):
            raise RuntimeError('Profiles already exist.')
        # if physical smoothing is required, create a DHT object
        if self.num.physical_smoothing_scale is not None :
            print 'Smoothing kappa profiles with physical smoothing.'
            DHTobj = GSL_DHT.DiscreteHankelTransform(int(self.num.smoothing_Npoints_scale*self.num.Npoints_theta))
            DHTobj.init(0.5, self.num.smoothing_Npoints_scale) # want the nu = 1/2 Hankel transform
            DHT_rsample = np.array(map(DHTobj.x_sample, np.arange(int(self.num.smoothing_Npoints_scale*self.num.Npoints_theta), dtype = int)))
            DHT_ksample = np.array(map(DHTobj.k_sample, np.arange(int(self.num.smoothing_Npoints_scale*self.num.Npoints_theta), dtype = int)))
            DHT_dict = {
                'DHTobj': DHTobj,
                'DHT_rsample': DHT_rsample,
                'DHT_ksample': DHT_ksample
                }
        else :
            DHT_dict = {None}
        # computes profiles and stores them in path
        self.signal_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z, self.num.Npoints_theta))
        self.theta_out_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z))
        start1 = time()
        for ii in xrange(self.num.Npoints_M) :
            start = time()
            for jj in xrange(self.num.Npoints_z) :
                self.theta_out_arr[ii,jj], self.signal_arr[ii,jj] = self.__generate_profile(
                    10.**self.num.logM_grid[ii],
                    self.num.z_grid[jj],
                    DHT_dict
                    )
            end = time()
            if ii%4 == 0 :
                print str((end-start)/60.*(self.num.Npoints_M-ii)) + ' minutes remaining in create profiles.'
        end1 = time()
        print end1-start1
        np.savez(
            path + 'profiles.npz',
            theta_out = self.theta_out_arr,
            signal = self.signal_arr
            )
    #}}}
    @staticmethod
    def __circular_pixel_window_function(x) :#{{{
        return 2.*jv(1,x)/x
    #}}}
    @staticmethod
    def _gaussian_pixel_window_function(x) :#{{{
        # for unit FWHM
        return np.exp(-x**2./16./np.log(2.))
    #}}}
    def create_convolved_profiles(self, path) :#{{{
        # throws error if convolved profiles already exist in path
        if (os.path.isfile(path + 'convolved_profiles.npz') and not self.num.debugging) or (self.convolved_signal_arr is not None) :
            raise RuntimeError('Convolved profiles already exist.')
        if (self.signal_arr is not None and self.theta_out_arr is not None) :
            pass
        else :
            if os.path.isfile(path + 'profiles.npz') :
                f = np.load(path + 'profiles.npz')
                self.theta_out_arr = f['theta_out']
                self.signal_arr = f['signal']
            else :
                raise RuntimeError('Profiles have not already been computed.')
        # checks if self has a field that contains the (unconvolved) profiles,
        # otherwise reads them in from path
        # TODO : read in signal_arr, theta_out_arr
        # TODO : write windowing functions (which depend on theta_out)
        # creates a DHT object with the appropriate parameters,
        # then carries out the convolution using function convolve() below
        # computes the (analytic) Hankel transform of the window function at the
        #   reciprocal space gridpoints
        # if it is passed, load the Wiener filter file
        if (self.num.Wiener_filter is not None) :
            # TODO : load file
            print 'Using Wiener filter ' + self.num.Wiener_filter
            self.convolve_with_Wiener = True
            pass
        else :
            self.convolve_with_Wiener = False
        if (self.num.pixel_radius is not None) :
            print 'Convolving with circular pixel.'
            self.convolve_with_circular_pixel = True
        else :
            self.convolve_with_circular_pixel = False
        if (self.num.pixel_sidelength is not None) :
            print 'Convolving with quadratic pixel.'
            self.convolve_with_quadratic_pixel = True
        else :
            self.convolve_with_quadratic_pixel = False
        if self.convolve_with_circular_pixel and self.convolve_with_quadratic_pixel :
            raise RuntimeError('You want to convolve with circular AND quadratic pixels.')
        if self.num.gaussian_kernel_FWHM is not None :
            print 'Applying additional effective Gaussian smoothing.'
        # check wether W_ell's are available for the quadratic pixel, otherwise compute them
        if self.num.empirical_bl_fct is not None :
            print 'Applying empirical smoothing.'
        if self.convolve_with_quadratic_pixel :
            if os.path.isfile('./constants/quadratic_pixel_W_ell.npz') :
                print 'Reading W_ell for quadratic pixel from file.'
                f = np.load('./constants/quadratic_pixel_W_ell.npz')
                pixel_ell = f['ell']
                pixel_W_ell = f['W_ell']
            else :
                print 'Computing W_ell for quadratic pixel.'
                pixel_ell = np.logspace(-2., 3., base = 10., num = int(1e5))
                pixel_W_ell = np.empty(len(pixel_ell))
                for ii in xrange(len(pixel_ell)) :
                    B_ell_phi = lambda phi: np.sinc(0.5*pixel_ell[ii]*np.cos(phi))**1.*np.sinc(0.5*pixel_ell[ii]*np.sin(phi))**1.
                    pixel_W_ell[ii],_ = quad(B_ell_phi, 0., np.pi/4.)
                pixel_W_ell *= 4./np.pi 
                np.savez(
                    path + 'quadratic_pixel_W_ell.npz',
                    ell = pixel_ell,
                    W_ell = pixel_W_ell
                    )
            quadratic_pixel_window_function_interp = interp1d(pixel_ell, pixel_W_ell, kind = 'quadratic', bounds_error = False, fill_value = (1., 0.))
            quadratic_pixel_window_function = lambda ell: quadratic_pixel_window_function_interp(ell)
        DHTobj = GSL_DHT.DiscreteHankelTransform(self.num.Npoints_theta)
        DHTobj.init(0, 1.)
        self.convolved_signal_arr = np.empty(self.signal_arr.shape) # assumes that theta is the last (3rd) direction
        for ii in xrange(self.signal_arr.shape[0]) : # logM-loop
            start = time()
            for jj in xrange(self.signal_arr.shape[1]) : # z-loop
                _, reci_signal = DHTobj.apply(self.signal_arr[ii,jj,:])
                Window = np.ones(self.num.Npoints_theta)
                if self.convolve_with_circular_pixel :
                    Window *= profiles.__circular_pixel_window_function(self.num.scaled_reci_theta_grid*self.num.pixel_radius/self.theta_out_arr[ii,jj])
                elif self.convolve_with_quadratic_pixel :
                    Window *= quadratic_pixel_window_function(self.num.scaled_reci_theta_grid*0.5*self.num.pixel_sidelength/self.theta_out_arr[ii,jj])
                else :
                    warn('You called create_convolved_profiles without actually convolving', UserWarning)
                if self.num.empirical_bl_fct is not None :
                    Window_empirical = self.num.empirical_bl_fct(self.num.scaled_reci_theta_grid/self.theta_out_arr[ii,jj])
                    Window *= Window_empirical
                if self.num.gaussian_kernel_FWHM is not None :
                    Window *= profiles._gaussian_pixel_window_function(self.num.scaled_reci_theta_grid*self.num.gaussian_kernel_FWHM/self.theta_out_arr[ii,jj])
#                plt.semilogx(self.num.scaled_reci_theta_grid, Window, label = 'Window')
#                plt.semilogx(self.num.scaled_reci_theta_grid, reci_signal/max(reci_signal), label = 'signal')
#                plt.legend(loc = 'upper right')
#                plt.show()
                reci_signal = reci_signal * Window
                _,self.convolved_signal_arr[ii,jj,:] = DHTobj.apply(reci_signal)
                self.convolved_signal_arr[ii,jj,:] *= (self.num.scaled_reci_theta_grid[-1]**2.)
#                plt.plot(self.num.scaled_real_theta_grid*self.theta_out_arr[ii,jj]*(180./np.pi)*60., self.signal_arr[ii,jj,:])
#                plt.plot(self.num.scaled_real_theta_grid*self.theta_out_arr[ii,jj]*(180./np.pi)*60., self.convolved_signal_arr[ii,jj,:])
#                plt.show()
                # FIX MONOTONICITY
                d = np.diff(self.convolved_signal_arr[ii,jj,:])
                if np.any(np.isnan(self.convolved_signal_arr[ii,jj,:])) :
                    print 'NAN'
                    print ii,jj
                    plt.plot(self.convolved_signal_arr[ii,jj,:])
                    plt.plot(np.nan_to_num(self.convolved_signal_arr[ii,jj,:]))
                    plt.show()
                if np.any(d>=0) :
                    print 'diff'
                    plt.plot(d)
                    plt.show()

            end = time()
            if ii%4 == 0 :
                print str((end-start)/60.*(self.signal_arr.shape[0]-ii)) + ' minutes remaining in create_convolved_profiles.'
        np.savez(
            path + 'convolved_profiles.npz',
            theta_out = self.theta_out_arr,
            convolved_signal = self.convolved_signal_arr
            )
    #}}}
    def create_tildes(self, path) :#{{{
        # throws error if tildes already exist in path
        if (os.path.isfile(path + 'tildes.npz') and not self.num.debugging) or (self.tilde_arr is not None) :
            raise RuntimeError('Tildes already exist.')
        if (self.convolved_signal_arr is not None) :
            print 'Doing tilde on convolved signal.'
            signal = self.convolved_signal_arr
            theta_out = self.theta_out_arr
        else :
            if os.path.isfile(path + 'convolved_profiles.npz') :
                print 'Doing tilde on convolved signal from file.'
                f = np.load(path + 'convolved_profiles.npz')
                signal = f['convolved_signal']
                theta_out = f['theta_out']
            elif ((self.signal_arr is not None) and (self.theta_out_arr is not None)) :
                print 'Doing tilde on unconvolved signal.'
                signal = self.signal_arr
                theta_out = self.theta_out_arr
            elif os.path.isfile(path + 'profiles.npz') :
                print 'Doing tilde on unconvolved signal from file.'
                f = np.load(path + 'profiles.npz')
                signal = f['signal']
                theta_out = f['theta_out']
            else :
                raise RuntimeError('No convolved or unconvolved profiles exist.')
        # if convolved_profiles do not exist, reads profiles but prints warning
        # else reads convolved_profiles
        # TODO : read profiles and theta_out values, store in theta_out, signal
        # computes the Y/kappa/...-tildes
        self.tilde_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z, len(self.num.lambda_grid)), dtype = np.complex64)
        for ii in xrange(self.num.Npoints_M) :
            start = time()
            for jj in xrange(self.num.Npoints_z) :
                try :
                    spl_interpolator = CubicSpline(signal[ii,jj,:][::-1], self.num.scaled_real_theta_grid[::-1], extrapolate = False) # theta ( signal )
                    der_interpolator = spl_interpolator.derivative() # dtheta/dsignal
                    theta_of_signal = np.nan_to_num(spl_interpolator(self.num.signal_grid))
                    dtheta_dsignal = np.fabs(np.nan_to_num(der_interpolator(self.num.signal_grid)))
    #                plt.plot(theta_of_signal, self.num.signal_grid, label = 'interp')
    #                plt.plot(self.num.scaled_real_theta_grid, signal[ii,jj,:], label = 'original')
    #                plt.xlim(0., 1.)
    #                plt.ylim(0., max(signal[ii,jj,:]))
    #                plt.legend(loc = 'upper right')
    #                plt.show()
    #                plt.plot(theta_of_signal, dtheta_dsignal)
    #                plt.show()
                    self.tilde_arr[ii,jj,:] = np.fft.rfft(theta_of_signal*dtheta_dsignal)
                    self.tilde_arr[ii,jj,:] *= 2.*np.pi*theta_out[ii,jj]**2.*(self.num.signal_grid[1]-self.num.signal_grid[0])
                    self.tilde_arr[ii,jj,:] -= self.tilde_arr[ii,jj,0] # subtract the zero mode
                except ValueError :
                    print 'FT failed in (' + str(ii) + ',' + str(jj) + ')'
                    self.tilde_arr[ii,jj,:] = theta_out[ii,jj]**2. * profiles.__tilde_lambda_loop(
                        self.num.scaled_real_theta_grid,
                        signal[ii,jj,:],
                        2.*self.num.lambda_grid
                        )
                #plt.plot(self.num.lambda_grid, np.real(self.tilde_arr[ii,jj,:]), label = 'new')
                #plt.plot(self.num.lambda_grid, np.real(comparison), label = 'old')
                #plt.legend()
                #plt.show()
            end = time()
            if ii%4 == 0 :
                print str((end-start)/60.*(self.num.Npoints_M-ii)) + ' minutes remaining in create_tildes.'
#        np.savez(
#            path + 'tildes.npz',
#            tildes = self.tilde_arr
#            )
    #}}}
    @staticmethod
    @jit(nopython=True, parallel = False)
    def __tilde_lambda_loop(theta_grid, profile, lambda_grid) :#{{{
        out_arr = np.empty(len(lambda_grid), dtype = np.complex64)
        for ii in prange(len(lambda_grid)) :
            integrand = 2.*np.pi*theta_grid*(np.exp(np.complex(0,-1)*lambda_grid[ii]*profile) - 1.)
            out_arr[ii] = np.trapz(integrand, x = theta_grid)
        return out_arr
    #}}}
    def create_y_ell(self) :#{{{
        if not self.cosmo.initialized_colossus :
            self.cosmo._initialize_colossus()
        rhos = np.empty((self.num.Npoints_small_M, self.num.Npoints_z))
        rs = np.empty((self.num.Npoints_small_M, self.num.Npoints_z))
        concentrations = np.empty((self.num.Npoints_small_M, self.num.Npoints_z))
        rvirs = np.empty((self.num.Npoints_small_M, self.num.Npoints_z))
        for jj in xrange(self.num.Npoints_z) :
            rvirs[:,jj] = self.cosmo.virial_radius_small_masses(
                10.**self.num.small_logM_grid,
                self.num.z_grid[jj],
                self.cosmo.mass_def_initial
                )
            concentrations[:,jj] = concentration(
                10.**self.num.small_logM_grid,
                self.cosmo.mass_def_initial,
                self.num.z_grid[jj],
                model = self.cosmo.small_mass_concentration_model,
                conversion_profile = self.cosmo.halo_profile
                )
            rhos[:,jj], rs[:,jj] = NFWProfile.fundamentalParameters(
                10.**self.num.small_logM_grid,
                concentrations[:,jj],
                self.num.z_grid[jj],
                self.cosmo.mass_def_initial
                )
        rhos *= 1e9 # Msun h^2/Mpc^3
        rs *= 1e-3 # Mpc/h
        xmax = 1.0*rvirs/rs
        #xmax[np.where(xmax>8.)] = 8.
        ls = self.cosmo.angular_diameter_distance(self.num.z_grid)[None,:]/rs
        L = (self.num.larr+0.5)[None,None,:]/ls[:,:,None]
        #a = 1./(1.+self.num.z_grid)
        #aL = a[None,:,None]*L
        #SI, CI = sici(aL)
        SI_L, CI_L = sici(L)
        SI_Lxmax, CI_Lxmax = sici(L*(1.+xmax[:,:,None]))
        Scrit = self.Sigma_crit(self.num.z_grid)
        #self.y_ell = 4.*np.pi*rs[:,:,None]*rhos[:,:,None]*a[None,:,None]**3./ls[:,:,None]**2./Scrit[None,:,None]*( -np.cos(aL)*CI + 0.5 * np.sin(aL) * (np.pi-2.*SI) )
#        self.y_ell = 4.*np.pi*rs[:,:,None]*rhos[:,:,None]/ls[:,:,None]**2./Scrit[None,:,None]*( -np.cos(L)*CI + 0.5 * np.sin(L) * (np.pi - 2.*SI) )
        self.y_ell = 4.*np.pi*rs[:,:,None]*rhos[:,:,None]/ls[:,:,None]**2./Scrit[None,:,None]*(
            -1./L/(1.+xmax[:,:,None])*(
                L*(1.+xmax[:,:,None])*np.cos(L)*(CI_L-CI_Lxmax) + 
                np.sin(L*xmax[:,:,None]) +
                L * np.sin(L) * (SI_L - SI_Lxmax) +
                L * xmax[:,:,None] * np.sin(L) * (SI_L - SI_Lxmax)
                )
            )
    #}}}
#}}}

class PDF(object) :#{{{
    # primarily inherits grids from cosmology and numerics
    def __init__(self, cosmo, num, prof, pardict={None}) :#{{{
        self.cosmo = cosmo
        self.num = num
        self.prof = prof
        self.alpha0_arr_cl = None
        self.alpha0_arr_uncl = None
        self.Ptilde_uncl = None
        self.Ptilde_cl = None
        self.P_uncl = None
        self.P_cl = None

        self.nu_obs = _set_param(pardict, 'nu_obs', None)
        if self.num.signal_type is 'tSZ' :
            if self.nu_obs is not None :
                x = cosmology.hPl*self.nu_obs/cosmology.kBoltzmann/self.cosmo.TCMB
                gnu = x/np.tanh(0.5*x) - 4.
                self.T_of_y = lambda y: gnu*y*self.cosmo.TCMB*1e6 # uK
            else :
                warn('You have not specified an observation frequency. Output in terms of Compton-y.', UserWarning)
        # reads in which names are assigned to required files in path,
        #   if num is None sets to default (useful for later binning)
        # checks whether required files exist in path
    #}}}
    def create_alpha0(self, path) :#{{{
        if (os.path.isfile(path + 'alpha0.npz') and not self.num.debugging) or ((self.alpha0_arr_cl is not None) and (self.alpha0_arr_uncl is not None)):
            raise RuntimeError('alpha0 already exists.')
        if ((self.cosmo.bias_arr is not None) and (self.cosmo.hmf_arr is not None)) :
            pass
        elif os.path.isfile(path + 'HMF_and_bias.npz') :
            print 'Reading HMF_and_bias from file.'
            f = np.load(path + 'HMF_and_bias.npz')
            self.cosmo.bias_arr = f['bias']
            self.cosmo.hmf_arr = f['hmf']
        else :
            raise RuntimeError('HMF and bias not found.')
        if self.prof.tilde_arr is not None :
            pass
        elif os.path.isfile(path + 'tildes.npz') :
            print 'Reading tildes from file.'
            f = np.load(path + 'tildes.npz')
            self.prof.tilde_arr = f['tildes']
        else :
            raise RuntimeError('Tildes not found.')
        self.alpha0_arr_uncl = np.empty((self.num.Npoints_z, len(self.num.lambda_grid)), dtype = complex)
        self.alpha0_arr_cl = np.empty((self.num.Npoints_z, len(self.num.lambda_grid)), dtype = complex)
        for ii in xrange(self.num.Npoints_z) :
            start = time()
            b = self.cosmo.bias_arr[:,ii]
            n = self.cosmo.hmf_arr[:,ii]
            tilde_slice = self.prof.tilde_arr[:,ii,:] # [ mass , lambda ]
            if self.cosmo.mass_def_initial != self.cosmo.mass_def_Tinker :
                dM200m = self.cosmo.convert_mass(10.**self.num.logM_grid+1e2, self.num.z_grid[ii], self.cosmo.mass_def_initial, self.cosmo.mass_def_Tinker) - self.cosmo.convert_mass(10.**self.num.logM_grid-1e2, self.num.z_grid[ii], self.cosmo.mass_def_initial, self.cosmo.mass_def_Tinker)
                Jacobian = dM200m/(2.*1e2)
            else :
                Jacobian = np.ones(self.num.Npoints_M)
            integrand_cl = Jacobian[:,None]*b[:,None]*n[:,None]*tilde_slice*10.**self.num.logM_grid[:,None]*np.log(10.)
            integrand_uncl = Jacobian[:,None]*n[:,None]*tilde_slice*10.**self.num.logM_grid[:,None]*np.log(10.)
            self.alpha0_arr_cl[ii,:] = simps(integrand_cl, x = self.num.logM_grid, axis = 0)
            self.alpha0_arr_uncl[ii,:] = simps(integrand_uncl, x = self.num.logM_grid, axis = 0)
            end = time()
            if ii%4 == 0 :
                print str((end-start)/60.*(self.num.Npoints_z-ii)) + ' minutes remaining in create_alpha0.'
        np.savez(
            path + 'alpha0.npz',
            alpha0_cl = self.alpha0_arr_cl,
            alpha0_uncl = self.alpha0_arr_uncl
            )
    #}}}
    def create_P_tilde(self, path) :#{{{
        if (os.path.isfile(path + 'Ptilde.npz') and not self.num.debugging) or ((self.Ptilde_uncl is not None) and (self.Ptilde_cl is not None)) :
            raise RuntimeError('Ptilde already exists.')
        if ((self.alpha0_arr_cl is not None) and (self.alpha0_arr_uncl is not None)) :
            pass
        elif os.path.isfile(path + 'alpha0.npz') :
            print 'Reading alpha0 from file.'
            f = np.load(path + 'alpha0.npz')
            self.alpha0_arr_cl = f['alpha0_cl']
            self.alpha0_arr_uncl = f['alpha0_uncl']
        else :
            raise RuntimeError('alpha0 not found.')
        prefactor_uncl = self.cosmo.comoving_distance(self.num.z_grid)**2./(self.cosmo.H(self.num.z_grid)/cosmology.c0)
        integrand_uncl = prefactor_uncl[:,None] * self.alpha0_arr_uncl
        prefactor_cl = self.cosmo.comoving_distance(self.num.z_grid)**4./(self.cosmo.H(self.num.z_grid)/cosmology.c0)*self.cosmo.D(self.num.z_grid)**2. * 0.5*self.cosmo.k_integral()
        integrand_cl = prefactor_cl[:,None]*self.alpha0_arr_cl**2.
        self.Ptilde_uncl = np.exp(simps(integrand_uncl, x = self.num.z_grid, axis = 0))
        self.Ptilde_cl = self.Ptilde_uncl * np.exp(simps(integrand_cl, x = self.num.z_grid, axis = 0))
        # throws error if Ptilde_unclustered already exists in path
        # computes Ptilde
        np.savez(
            path + 'Ptilde.npz',
            Ptilde_uncl = self.Ptilde_uncl,
            Ptilde_cl = self.Ptilde_cl,
            signal_min = self.num.signal_min,
            signal_max = self.num.signal_max
            )
    #}}}
    def create_P(self, path, pardict = {None}) :#{{{
        do_gaussian_piece = _set_param(pardict, 'gaussian_piece', True)
        if (os.path.isfile(path + 'P.npz') and not self.num.debugging) :
            raise RuntimeError('P already exists.')
        if ((self.Ptilde_uncl is not None) and (self.Ptilde_cl is not None)) :
            pass
        elif os.path.isfile(path + 'Ptilde.npz') :
            print 'Reading Ptilde from file.'
            f = np.load(path + 'Ptilde.npz')
            self.Ptilde_uncl = f['Ptilde_uncl']
            self.Ptilde_cl = f['Ptilde_cl']
            self.num.signal_max = f['signal_max']
            self.num.signal_min = f['signal_min']
        else :
            raise RuntimeError('Ptilde not found.')
#        plt.plot(np.real(self.Ptilde_uncl))
#        plt.plot(np.imag(self.Ptilde_uncl))
#        plt.show()
        if (self.num.signal_type is 'kappa') and do_gaussian_piece :
            if (self.cosmo.small_mass_hmf_arr is None) or (self.cosmo.small_mass_bias_arr is None) :
                f = np.load(path + 'HMF_and_bias.npz')
                self.cosmo.small_mass_hmf_arr = f['small_mass_hmf']
                self.cosmo.small_mass_bias_arr = f['small_mass_bias']
            signal_values_FT = np.linspace(self.num.signal_min, self.num.signal_max, num = self.num.Npoints_signal)
            self.prof.create_y_ell()
            self.create_kappa_variance()
            Gaussian_piece = np.exp(-signal_values_FT**2./2./self.small_mass_kappa_variance)
            FT_Gaussian_piece = np.fft.rfft(Gaussian_piece)
            self.Ptilde_cl *= FT_Gaussian_piece
            self.Ptilde_uncl *= FT_Gaussian_piece
        self.P_uncl = np.fft.irfft(self.Ptilde_uncl)
        self.P_cl = np.fft.irfft(self.Ptilde_cl)
        # throws error if P already exists in path
        # checks if self has P and clustering correction, else reads them from path
        # performs the Fourier transform
        np.savez(
            path + 'P.npz',
            P_uncl = self.P_uncl,
            P_cl = self.P_cl,
            signal_min = self.num.signal_min,
            signal_max = self.num.signal_max
            )
    #}}}
    def create_P_Poisson(self, path, index, pardict = {None}) :#{{{
        self.map_sidelength = _set_param(pardict, 'map_sidelength', (np.pi/180.)*9.)
        self.map_area = self.map_sidelength**2.
        if self.cosmo.d2_arr is None :
            print 'Reading d2 from file.'
            f = np.load(path + 'HMF_and_bias.npz')
            self.cosmo.d2_arr = f['d2']
        if self.prof.tilde_arr is None :
            raise RuntimeError('Please create tildes first.')
        cluster_numbers = np.random.poisson(self.map_area*self.cosmo.d2_arr)
        self.P_tilde_Poisson = np.exp(
            1./self.map_area * np.sum(cluster_numbers[:,:,None] * self.prof.tilde_arr, axis = (0,1))
            )
        self.P_Poisson = np.fft.irfft(self.P_tilde_Poisson)
        np.savez(
            path + '/Pfine/Pfine_' + str(index) + '.npz',
            P = self.P_Poisson,
            signal_min = self.num.signal_min,
            signal_max = self.num.signal_max
            )
    #}}}
    def create_kappa_variance(self) :#{{{
        onehalo_mass_integrand = self.cosmo.small_mass_hmf_arr[:,:,None]*self.prof.y_ell**2.
        twohalo_mass_integrand = self.cosmo.small_mass_hmf_arr[:,:,None]*(self.cosmo.small_mass_bias_arr[:,:,None])*self.prof.y_ell
        onehalo_mass_integral = simps(onehalo_mass_integrand*np.log(10.)*10.**self.num.small_logM_grid[:,None,None], x = self.num.small_logM_grid, axis = 0)
        twohalo_mass_integral = simps(twohalo_mass_integrand*np.log(10.)*10.**self.num.small_logM_grid[:,None,None], x = self.num.small_logM_grid, axis = 0)
        volume_factor = self.cosmo.comoving_distance(self.num.z_grid)**2./(self.cosmo.H(self.num.z_grid)/cosmology.c0)
        Plin = np.empty((self.num.Npoints_z, len(self.num.larr)))
        for ii in xrange(self.num.Npoints_z) :
            Plin[ii,:] = self.cosmo.PK(self.num.z_grid[ii], (self.num.larr+0.5)/self.cosmo.comoving_distance(self.num.z_grid[ii]))
            Plin[ii,:] *= ((self.num.larr+0.5)/self.cosmo.comoving_distance(self.num.z_grid[ii]))<101.
        self.onehalo_term = simps(volume_factor[:,None]*onehalo_mass_integral, x = self.num.z_grid, axis = 0)
        self.twohalo_term = simps(volume_factor[:,None]*twohalo_mass_integral**2.*Plin, x = self.num.z_grid, axis = 0)
        # Need to convolve with pixel window function,
        # otherwise we get an unphysical divergence
        if os.path.isfile('./constants/quadratic_pixel_W2_ell.npz') :
            print 'Reading W2_ell for quadratic pixel from file.'
            f = np.load('./constants/quadratic_pixel_W2_ell.npz')
            pixel_ell = f['ell']
            pixel_W2_ell = f['W2_ell']
        else :
            print 'Computing W2_ell for quadratic pixel.'
            pixel_ell = np.logspace(-2., 3., base = 10., num = int(1e5))
            pixel_W2_ell = np.empty(len(pixel_ell))
            for ii in xrange(len(pixel_ell)) :
                B_ell_phi_sq = lambda phi: np.sinc(0.5*pixel_ell[ii]*np.cos(phi))**2.*np.sinc(0.5*pixel_ell[ii]*np.sin(phi))**2.
                pixel_W2_ell[ii],_ = quad(B_ell_phi_sq, 0., np.pi/4.)
            pixel_W2_ell *= 4./np.pi 
            np.savez(
                './constants/quadratic_pixel_W2_ell.npz',
                ell = pixel_ell,
                W2_ell = pixel_W2_ell
                )
        quadratic_pixel_window_function_interp = interp1d(pixel_ell, pixel_W2_ell, kind = 'quadratic', bounds_error = False, fill_value = (1., 0.))
        quadratic_pixel_window_function = lambda ell: quadratic_pixel_window_function_interp(ell)
        Window = np.ones(self.num.N_ell)
        if self.num.pixel_sidelength is not None :
            Window *= quadratic_pixel_window_function(self.num.larr*0.5*self.num.pixel_sidelength)
        if self.num.gaussian_kernel_FWHM is not None :
            Window *= profiles._gaussian_pixel_window_function(self.num.larr*self.num.gaussian_kernel_FWHM)**2.
        self.onehalo_term *= Window
        self.twohalo_term *= Window
        self.Ckappa_ell = self.onehalo_term + self.twohalo_term
        self.small_mass_kappa_variance = simps((2.*self.num.larr+1.)/4./np.pi * self.Ckappa_ell, x = self.num.larr)
        print self.small_mass_kappa_variance
    #}}}
    def compare_to_sims(self, path, sim_path, ax, fudge_factor) :#{{{
        if self.P_cl is not None :
            pass
        elif os.path.isfile(path + 'P.npz') :
            print 'Reading P from file.'
            f = np.load(path + 'P.npz')
            self.P_cl = f['P_cl']
            self.num.signal_min = f['signal_min']
            self.num.signal_max = f['signal_max']
        else :
            raise RuntimeError('P not found.')
        if self.num.signal_type is 'kappa' :
            Nmaps = 1e4
            MassiveNus_first_part = 'MassiveNus_Om0.3_As2.1e-09_Mnu0.0_zs'
            zs_str = str(round(self.num.z_source,1))
            if self.num.gaussian_kernel_FWHM is not None :
                sim_binedges_file = sim_path + MassiveNus_first_part + zs_str + '_binedges_filtFWHM5.0arcmin.txt'
                sim_p_file = sim_path + MassiveNus_first_part + zs_str + '_PDF_stddev_filtFWHM5.0arcmin.txt'
            else :
                sim_binedges_file = sim_path + MassiveNus_first_part + zs_str + '_binedges.txt'
                sim_p_file = sim_path + MassiveNus_first_part + zs_str + '_PDF_stddev.txt'
            bin_edges = np.loadtxt(sim_binedges_file)
            sim_p, sim_std = np.loadtxt(sim_p_file, unpack = True)
            sim_std /= np.sqrt(Nmaps)
        elif self.num.signal_type is 'tSZ' :
            bin_edges = np.linspace(-100., 0., num = 101)
            sim_p, sim_std = np.loadtxt('./Colin_1uK_binning.csv', unpack = True, delimiter = ' ')
            sim_p = np.flip(sim_p)
            sim_std = np.flip(sim_std)
        bin_centres = 0.5*(bin_edges[1:] + bin_edges[:-1])
        binned_PDF = np.empty(len(bin_centres))
        if self.num.signal_type is 'tSZ' :
            signal_min = self.T_of_y(self.num.signal_min)
            signal_max = self.T_of_y(self.num.signal_max)
        else :
            signal_min = self.num.signal_min
            signal_max = self.num.signal_max
        signal_values_FT = fudge_factor*np.linspace(signal_min, signal_max, num = len(self.P_cl))
        # shift to zero kappa
        if self.num.signal_type is 'kappa' :
            mean_kappa_FT = np.sum(signal_values_FT*self.P_cl)/np.sum(self.P_cl)
            signal_values_FT -= mean_kappa_FT
#        var_kappa = np.sum(signal_values_FT**2.*self.P_cl)/np.sum(self.P_cl)
#        print np.sqrt(var_kappa)
        PDF_interp = interp1d(signal_values_FT, self.P_cl, kind = 'quadratic', bounds_error = False, fill_value = 0.)
#        plt.semilogy(signal_values_FT, self.P_cl)
#        plt.show()
        PDF_interp_fct = lambda signal: PDF_interp(signal)
        for ii in xrange(len(bin_centres)) :
            binned_PDF[ii],_ = quad(PDF_interp_fct, bin_edges[ii], bin_edges[ii+1])
        binned_PDF /= np.sum(binned_PDF)
        bin_centres_FT = bin_centres.copy()
        line_sim = ax.errorbar(bin_centres, sim_p, yerr = sim_std, label = 'simulation', color = 'blue', lw = 2)
        line_FT_fudge = ax.semilogy(bin_centres_FT, binned_PDF, label = 'FT fudge', color = 'red', lw = 2)
        if True :
            signal_values_FT = np.linspace(signal_min, signal_max, num = len(self.P_cl))
            # shift to zero kappa
            if self.num.signal_type is 'kappa' :
                mean_kappa_FT = np.sum(signal_values_FT*self.P_cl)/np.sum(self.P_cl)
                signal_values_FT -= mean_kappa_FT
    #        var_kappa = np.sum(signal_values_FT**2.*self.P_cl)/np.sum(self.P_cl)
    #        print np.sqrt(var_kappa)
            PDF_interp = interp1d(signal_values_FT, self.P_cl, kind = 'quadratic', bounds_error = False, fill_value = 0.)
    #        plt.semilogy(signal_values_FT, self.P_cl)
    #        plt.show()
            PDF_interp_fct = lambda signal: PDF_interp(signal)
            for ii in xrange(len(bin_centres)) :
                binned_PDF[ii],_ = quad(PDF_interp_fct, bin_edges[ii], bin_edges[ii+1])
            binned_PDF /= np.sum(binned_PDF)
            line_FT = ax.semilogy(bin_centres_FT, binned_PDF, label = 'FT', color = 'green', lw = 2)
        return line_sim, line_FT, line_FT_fudge
    #}}}
#}}}

class signal_map(object) :#{{{
    def __init__(self, cosmo, num, prof, pardict = {None}) :#{{{
        self.cosmo = cosmo
        self.num = num
        self.prof = prof
        self.map_sidelength = _set_param(pardict, 'map_sidelength', (np.pi/180.)*10.)
        if self.num.pixel_sidelength is None :
            raise RuntimeError('Pixel sidelength is not specified for map.')
        else :
            self.map_linear_size = int(self.map_sidelength/self.num.pixel_sidelength)
        self.map_area = (self.map_linear_size*self.num.pixel_sidelength)**2.
        self.map_fraction = _set_param(pardict, 'map_fraction', 0.9)
        self.map_grid_per_pixel = _set_param(pardict, 'map_grid_per_pixel', 3)
        self.final_map = None
        self.hist = None
        self.exact_cluster_number = _set_param(pardict, 'exact_cluster_number', False)
    #}}}
    @staticmethod
    @jit
    def throw_clusters(cluster_number, final_map, this_map, central_pixels_x, central_pixels_y):#{{{
        central_pixel_x_this_map = (this_map.shape[0]-1)/2
        central_pixel_y_this_map = (this_map.shape[1]-1)/2
        for kk in xrange(cluster_number):                                                                     
            extent_x_right = final_map.shape[0] - int(central_pixels_x[kk]) - 1
            extent_x_left  = int(central_pixels_x[kk])
            extent_y_up    = final_map.shape[1] - int(central_pixels_y[kk]) - 1
            extent_y_down  = int(central_pixels_y[kk])
            # number of pixels free in all directions from the central pixel
            # actually, the this_map can be smaller:
            extent_x_right = (np.array([extent_x_right, central_pixel_x_this_map])).min()
            extent_x_left  = (np.array([extent_x_left, central_pixel_x_this_map])).min()
            extent_y_up    = (np.array([extent_y_up, central_pixel_y_this_map])).min()
            extent_y_down  = (np.array([extent_y_down, central_pixel_y_this_map])).min()
            final_map[
                    int(central_pixels_x[kk])-extent_x_left:int(central_pixels_x[kk])+extent_x_right+1,
                    int(central_pixels_y[kk])-extent_y_down:int(central_pixels_y[kk])+extent_y_up+1
                    ] += this_map[
                            central_pixel_x_this_map-extent_x_left:central_pixel_x_this_map+extent_x_right+1,
                            central_pixel_y_this_map-extent_y_down:central_pixel_y_this_map+extent_y_up+1
                            ]
        return final_map
    #}}}
    def create_final_map(self, path) :#{{{
        if self.cosmo.d2_arr is None :
            print 'Reading d2 from file.'
            f = np.load(path + 'HMF_and_bias.npz')
            self.cosmo.d2_arr = f['d2']
        if (self.num.signal_type is 'tSZ') and ((self.prof.signal_arr is None) or (self.prof.theta_out_arr is None)) :
            print 'Reading profiles from file.'
            f = np.load(path + 'profiles.npz')
            self.prof.theta_out_arr = f['theta_out']
            self.prof.signal_arr = f['signal']
        self.final_map = np.zeros((self.map_linear_size, self.map_linear_size))
        for ii in xrange(self.num.Npoints_M) :
            for jj in xrange(self.num.Npoints_z) :
                if not self.exact_cluster_number :
                    cluster_number = np.random.poisson(self.cosmo.d2_arr[ii,jj] * self.map_area)
                else :
                    cluster_number = int(np.floor(self.cosmo.d2_arr[ii,jj] * self.map_area))
                if (cluster_number > 0) or (self.exact_cluster_number and (self.cosmo.d2_arr[ii,jj] * self.map_area > 0.)) :
                    central_pixels_x = np.random.random_integers(0, self.map_linear_size-1, size = cluster_number)
                    central_pixels_y = np.random.random_integers(0, self.map_linear_size-1, size = cluster_number)
                    random_offset_x = np.random.rand()-0.5
                    random_offset_y = np.random.rand()-0.5
                    if self.num.signal_type is 'tSZ' :
                    # interpolate the signal profile (for tSZ direct computation is expensive)
                        theta_out = self.prof.theta_out_arr[ii,jj]
                        signal_interpolator = interp1d(
                            self.num.scaled_real_theta_grid*theta_out,
                            self.prof.signal_arr[ii,jj,:],
                            kind = 'quadratic',
                            bounds_error = False,
                            fill_value = (max(self.prof.signal_arr[ii,jj,:]), 0.)
                            )
                        signal_of_theta = lambda t : signal_interpolator(t)
                    elif self.num.signal_type is 'kappa' :
                        H = self.cosmo.H(self.num.z_grid[jj])
                        d_A = self.cosmo.angular_diameter_distance(self.num.z_grid[jj])
                        rvir = self.cosmo.virial_radius(
                            10.**self.num.logM_grid[ii],
                            self.num.z_grid[jj],
                            self.cosmo.mass_def_initial
                            )
                        rhoc = 2.775e7*H**2.
                        if self.cosmo.r_out_def is 'vir' :
                            r_out = self.cosmo.r_out_scale * rvir
                        else :
                            raise RuntimeError('Your r_out_def is not implemented.')
                        theta_out = np.arctan(r_out/d_A)
                        Mvir = self.cosmo.convert_mass(
                            10.**self.num.logM_grid[ii],
                            self.num.z_grid[jj],
                            self.cosmo.mass_def_initial,
                            self.cosmo.mass_def_kappa_profile
                            )
                        cvir = 5.72 * (Mvir/1e14)**(-0.081)/(1.+self.num.z_grid[jj])**(0.71)
                        rs = rvir / cvir
                        rho0 = Mvir/(4.*np.pi*rs**3.*(np.log(1.+cvir)-cvir/(1.+cvir)))
                        rhoM = self.cosmo.astropy_cosmo.Om0*(1.+self.num.z_grid[jj])**3.*2.775e7/self.cosmo.h**2.
                        critical_surface_density = self.prof.Sigma_crit(self.num.z_grid[jj])
                        signal_of_theta = lambda t : profiles._kappa_profile(
                            t,
                            rho0, rhoM, rs, r_out, d_A, theta_out, critical_surface_density
                            )
                    this_map = np.zeros((
                        2*int(theta_out/self.num.pixel_sidelength) + 5,
                        2*int(theta_out/self.num.pixel_sidelength) + 5
                        ))
                    pixel_indices_x = np.linspace(-(this_map.shape[0]-1)/2.,(this_map.shape[0]-1)/2.,num=this_map.shape[0])
                    pixel_indices_y = np.linspace(-(this_map.shape[1]-1)/2.,(this_map.shape[1]-1)/2.,num=this_map.shape[1])
                    if cluster_number > 0 :
                        nn = 0
                        for kk in xrange(-self.map_grid_per_pixel, self.map_grid_per_pixel+1) :
                            for ll in xrange(-self.map_grid_per_pixel, self.map_grid_per_pixel+1) :
                                angles = self.num.pixel_sidelength * np.sqrt(np.add.outer(
                                    (pixel_indices_x+random_offset_x+float(kk)/float(self.map_grid_per_pixel+0.5))**2.,
                                    (pixel_indices_y+random_offset_y+float(ll)/float(self.map_grid_per_pixel+0.5))**2.,
                                    ))
                                this_map[angles<theta_out] += map(signal_of_theta,angles[angles<theta_out])
                                nn += 1
                        this_map /= float(nn)
                        self.final_map = signal_map.throw_clusters(cluster_number, self.final_map, this_map, central_pixels_x, central_pixels_y)
                        if self.exact_cluster_number :
                            last_central_pixel_x = np.random.random_integers(0, self.map_linear_size-1, size = 1)
                            last_central_pixel_y = np.random.random_integers(0, self.map_linear_size-1, size = 1)
                            phi = ((np.arctan2.outer(pixel_indices_x+random_offset_x,pixel_indices_y+random_offset_y)+np.pi)/2./np.pi + np.random.rand())%1. # in [0,1]
                            difference = self.cosmo.d2_arr[ii,jj] * self.map_area - float(cluster_number)
                            this_map[phi>difference] = 0.
    #                        print cluster_number
    #                        plt.matshow(this_map)
    #                        plt.show()
                            self.final_map = signal_map.throw_clusters(1, self.final_map, this_map, last_central_pixel_x, last_central_pixel_y)
                    else :
                        last_central_pixel_x = np.random.random_integers(0, self.map_linear_size-1, size = 1)
                        last_central_pixel_y = np.random.random_integers(0, self.map_linear_size-1, size = 1)
                        phi = ((np.arctan2.outer(pixel_indices_x+random_offset_x,pixel_indices_y+random_offset_y)+np.pi)/2./np.pi + np.random.rand())%1. # in [0,1]
                        difference = self.cosmo.d2_arr[ii,jj] * self.map_area - float(cluster_number)
                        nn = 0
                        for kk in xrange(-self.map_grid_per_pixel, self.map_grid_per_pixel+1) :
                            for ll in xrange(-self.map_grid_per_pixel, self.map_grid_per_pixel+1) :
                                angles = self.num.pixel_sidelength * np.sqrt(np.add.outer(
                                    (pixel_indices_x+random_offset_x+float(kk)/float(self.map_grid_per_pixel+0.5))**2.,
                                    (pixel_indices_y+random_offset_y+float(ll)/float(self.map_grid_per_pixel+0.5))**2.,
                                    ))
                                this_map[(phi<difference)*(angles<theta_out)] += map(signal_of_theta,angles[(phi<difference)*(angles<theta_out)])
                                nn += 1
                        this_map /= float(nn)
                        self.final_map = signal_map.throw_clusters(1, self.final_map, this_map, last_central_pixel_x, last_central_pixel_y)
        spare_pixels_horizontal = int((1.-self.map_fraction)/2.*self.final_map.shape[0])
        spare_pixels_vertical = int((1.-self.map_fraction)/2.*self.final_map.shape[1])
        self.final_map = self.final_map[spare_pixels_horizontal:-spare_pixels_horizontal-1,spare_pixels_vertical:-spare_pixels_vertical-1]
    #}}}
    def create_histogram(self, bin_edges, path = None, index = None) :#{{{
        if self.final_map is None :
            raise RuntimeError('You have not computed a map.')
        hist,_ = np.histogram(self.final_map.flatten(), bin_edges)
        self.hist = hist/float(sum(hist))
        if path is not None :
            if index == 0 :
                np.savez(
                    path + 'bin_edges.npz',
                    bin_edges = bin_edges
                    )
            if not self.exact_cluster_number :
                np.savez(
                    path + '/map_histograms/hist_' + str(index) + '.npz',
                    hist = self.hist
                    )
            else :
                np.savez(
                    path + '/exact_map_histograms/hist_' + str(index) + '.npz',
                    hist = self.hist
                    )
    #}}}
#}}}

class Poisson_histograms(object) :#{{{
    def __init__(self, path) :#{{{
        self.map_histograms = None
        self.exact_map_histograms = None
        self.P_histograms = None
        self.path = path
        self.bin_centres = None
        if not os.path.isfile(self.path + 'bin_edges.npz') :
            raise RuntimeError('Path does not contain a bin_edges file.')
        else :
            print 'Loaded bin_edges.'
            f = np.load(self.path + 'bin_edges.npz')
            self.bin_edges = f['bin_edges']
            self.bin_centres = 0.5*(self.bin_edges[1:]+self.bin_edges[:-1])
        self.map_avg = None
        self.exact_map_avg = None
        self.P_avg = None
        self.P_exact_histogram = None
    #}}}
    def read_map_histograms(self, maxindex = 1000000) :#{{{
        self.map_histograms = []
        index = 0
        while True :
            if index > maxindex :
                print 'Loaded ' + str(index) + ' map histograms.'
                break
            try :
                f = np.load(self.path + '/map_histograms/hist_' + str(index) + '.npz')
                self.map_histograms.append(f['hist'])
            except IOError :
                print 'Loaded ' + str(index) + ' map histograms.'
                break
            index += 1
        self.map_histograms = np.array(self.map_histograms)
    #}}}
    def read_exact_map_histograms(self, maxindex = 1000000) :#{{{
        self.exact_map_histograms = []
        index = 0
        while True :
            if index > maxindex :
                print 'Loaded ' + str(index) + ' map histograms.'
                break
            try :
                f = np.load(self.path + '/exact_map_histograms/hist_' + str(index) + '.npz')
                self.exact_map_histograms.append(f['hist'])
            except IOError :
                print 'Loaded ' + str(index) + ' exact map histograms.'
                break
            index += 1
        self.exact_map_histograms = np.array(self.exact_map_histograms)
    #}}}
    def read_P_histograms(self, maxindex = 1000000) :#{{{
        self.P_histograms = []
        index = 0
        while True :
            if index > maxindex :
                print 'Loaded ' + str(index) + ' P histograms.'
                break
            try :
                f = np.load(self.path + 'P_histograms/P_hist_' + str(index) + '.npz')
                self.P_histograms.append(f['P_hist'])
            except IOError :
                try :
                    f = np.load(self.path + 'Pfine/Pfine_' + str(index) + '.npz')
                    interp_P = interp1d(
                        np.linspace(f['signal_min'], f['signal_max'], num = len(f['P'])),
                        f['P'],
                        kind = 'quadratic',
                        bounds_error = False,
                        fill_value = (f['P'][0], 0.)
                        )
                    P_of_signal = lambda signal : interp_P(signal)
                    hist_here = np.empty(len(self.bin_centres))
                    for ii in xrange(len(self.bin_centres)) :
                        hist_here[ii],_ = quad(P_of_signal, self.bin_edges[ii], self.bin_edges[ii+1])
                    hist_here = hist_here/float(sum(hist_here))
                    np.savez(
                        self.path + 'P_histograms/P_hist_'+ str(index) + '.npz',
                        P_hist = hist_here
                        )
                    self.P_histograms.append(hist_here)
                except IOError :
                    print 'Loaded ' + str(index) + ' P histograms.'
                    break
            index += 1
        if os.path.isfile(self.path + 'P.npz') :
            f = np.load(self.path + 'P.npz')
            interp_P = interp1d(
                np.linspace(f['signal_min'], f['signal_max'], num = len(f['P_uncl'])),
                f['P_uncl'],
                kind = 'quadratic',
                bounds_error = False,
                fill_value = (f['P_uncl'][0], 0.)
                )
            P_of_signal = lambda signal : interp_P(signal)
            hist_here = np.empty(len(self.bin_centres))
            for ii in xrange(len(self.bin_centres)) :
                hist_here[ii],_ = quad(P_of_signal, self.bin_edges[ii], self.bin_edges[ii+1])
            hist_here = hist_here/float(sum(hist_here))
            self.P_exact_histogram = hist_here
        self.P_histograms = np.array(self.P_histograms)
    #}}}
    def get_averages(self) :#{{{
        if self.bin_centres is None :
            raise RuntimeError('No bin_centres found.')
        if (self.map_histograms is None) and (self.P_histograms is None) :
            raise RuntimeError('No histograms loaded.')
        if self.map_histograms is not None :
            self.map_avg = np.average(self.map_histograms, axis = 0)
            plt.semilogy(self.bin_centres, self.map_avg, label = 'maps')
        if self.exact_map_histograms is not None :
            self.exact_map_avg = np.average(self.exact_map_histograms, axis = 0)
            plt.semilogy(self.bin_centres, self.exact_map_avg, label = 'exact maps')
        if self.P_histograms is not None :
            self.P_avg = np.average(self.P_histograms, axis = 0)
            plt.semilogy(self.bin_centres, self.P_avg, label = 'Poisson analytic')
        if self.P_exact_histogram is not None :
            plt.semilogy(self.bin_centres, self.P_exact_histogram, label = 'exact FT (uncl)')
        plt.legend(loc = 'upper right')
        plt.xlabel('signal')
        plt.ylabel('PDF')
        plt.show()
    #}}}
    def get_2nd_moment(self) :
        if self.bin_centres is None :
            raise RuntimeError('No bin_centres found.')
        if (self.map_histograms is None) and (self.P_histograms is None) :
            raise RuntimeError('No histograms loaded.')
        if self.map_histograms is not None :
            if self.map_avg is None :
                self.map_avg = np.average(self.map_histograms, axis = 0)
            self.map_cov = np.cov(self.map_histograms/self.map_avg[None,:], rowvar = False)
            self.map_var = np.diag(self.map_cov)
        #    plt.semilogy(self.bin_centres, self.map_var, label = 'maps')
        if self.exact_map_histograms is not None :
            if self.exact_map_avg is None :
                self.exact_map_avg = np.average(self.exact_map_histograms, axis = 0)
            self.exact_map_cov = np.cov(self.exact_map_histograms/self.exact_map_avg[None,:], rowvar = False)
            self.exact_map_var = np.diag(self.exact_map_cov)
        #    plt.semilogy(self.bin_centres, self.exact_map_var, label = 'exact maps')
        if self.P_histograms is not None :
            if self.P_avg is None :
                self.P_avg = np.average(self.P_histograms, axis = 0)
            self.P_cov = np.cov(self.P_histograms/self.P_avg[None,:], rowvar = False)
            self.P_var = np.diag(self.P_cov)
        #    plt.semilogy(self.bin_centres, self.P_var, label = 'Poisson analytic')
        #plt.legend(loc = 'upper left')
        #plt.xlabel('signal')
        #plt.ylabel('variance')
        #plt.show()
        #plt.matshow(self.map_cov/np.sqrt(np.multiply.outer(self.map_var,self.map_var)))
        #plt.title('maps')
        #plt.show()
        #plt.matshow(self.exact_map_cov/np.sqrt(np.multiply.outer(self.exact_map_var,self.exact_map_var)))
        #plt.title('exact maps')
        #plt.show()
        #plt.plot(self.bin_centres, self.map_var/self.P_var**2., label = 'ratio')
        #plt.legend(loc = 'upper left')
        #plt.show()
    def get_nth_moment(self, n) :
        if self.bin_centres is None :
            raise RuntimeError('No bin_centres found.')
        if (self.map_histograms is None) and (self.P_histograms is None) :
            raise RuntimeError('No histograms loaded.')
        if self.map_histograms is not None :
            if self.map_avg is None :
                self.map_avg = np.average(self.map_histograms, axis = 0)
            nthmoment_maps = moment(self.map_histograms/self.map_avg[None,:], moment = n, axis = 0)
            plt.plot(self.bin_centres, nthmoment_maps, label = 'maps')
        if self.exact_map_histograms is not None :
            if self.exact_map_avg is None :
                self.exact_map_avg = np.average(self.exact_map_histograms, axis = 0)
            nthmoment_exact_maps = moment(self.exact_map_histograms/self.exact_map_avg[None,:], moment = n, axis = 0)
            plt.plot(self.bin_centres, nthmoment_exact_maps, label = 'exact maps')
        if self.P_histograms is not None :
            if self.P_avg is None :
                self.P_avg = np.average(self.P_histograms, axis = 0)
            nthmoment_P = moment(self.P_histograms/self.P_avg[None,:], moment = n, axis = 0)
            plt.plot(self.bin_centres, nthmoment_P, label = 'Poisson analytic')
        plt.legend(loc = 'upper left')
        plt.xlabel('signal')
        plt.ylabel(str(n) + ' moment')
        plt.show()
        plt.plot(self.bin_centres, nthmoment_maps/nthmoment_P, label = 'ratio')
        plt.legend(loc = 'upper left')
        plt.show()
#}}}
                

# workflow
# Full run (state of the art)
if __name__ == '__main__' :#{{{
    name = './testsApr30/'
    cosmo = cosmology(
        {
#        'external_HMF_file': './constants/MassiveNus_HMF_from_Colin.npz'
        }
        )
    #H_ell, HC_ell = np.loadtxt('./MassiveNus_C_ells/Halofit_C_ell_zs1.0.csv', unpack = True, delimiter = ',')
    #M_ell, MC_ell = np.loadtxt('./MassiveNus_C_ells/MassiveNus_C_ell_zs1.0.csv', unpack = True, delimiter = ',')
    #MC_ell_interp = interp1d(
    #    M_ell,
    #    MC_ell,
    #    kind = 'quadratic',
    #    bounds_error = False,
    #    fill_value = 0.
    #    )
    #MC_ell_at_H_ell = MC_ell_interp(H_ell)
    #bl = np.sqrt(MC_ell_at_H_ell/HC_ell)
    #empirical_ell = H_ell[np.where(bl>0)]
    #empirical_bl = bl[np.where(bl>0)]
    num = numerics(
        {
#        'empirical_bl': empirical_bl,
#        'empirical_bl_ell': empirical_ell,
        'small_logM_min': 4.,
        'logM_min': 11.0,
        'z_source': 1.0,
        'debugging': True,
        'sigma_chi_file': './constants/MassiveNus_sigma_chi_file.npz',
        'signal_type': 'kappa',
        'Npoints_theta': 1000,
        'signal_max': 5.,
        'pixel_sidelength': (np.pi/180.)*0.41/60.,
#        'gaussian_kernel_FWHM': (np.pi/180.)*5./60.,
        'physical_smoothing_scale': 0.001 # Mpc/h
        }
        )
    #cosmo.create_HMF_and_bias(name, num, {'do_d2': False})
    pr = profiles(cosmo, num)
    pr.create_profiles(name)
    pr.create_convolved_profiles(name)
    pr.create_tildes(name)
    p = PDF(cosmo, num, pr)
    p.create_alpha0(name)
    p.create_P_tilde(name)
    p.create_P(name)
    fig,ax = plt.subplots()
    line_sim, line_FT, line_FT_fudge = p.compare_to_sims(name, './MassiveNus_sims/', ax, 1.)
#    ax.set_xlim(-0.05, 0.35)
#    ax.set_ylim(1e-7, 2e-1)
#    plt.savefig('./testsApr30/modified_concentration_zs2.0_f0.83.pdf', bbox_inches = 'tight')
    plt.show()
#}}}
