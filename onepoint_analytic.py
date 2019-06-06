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
import pickle

def _set_param(input_dict, name, default) :
    return input_dict[name] if name in input_dict else default

class numerics(object) :#{{{
    def __init__(self, numerics_dict={None}) :
        self.debugging = _set_param(numerics_dict, 'debugging', False)
        self.verbose = _set_param(numerics_dict, 'verbose', True)

        # grid parameters
        self.Npoints_M = _set_param(numerics_dict, 'Npoints_M', 50)
        self.Npoints_z = _set_param(numerics_dict, 'Npoints_z', 51)
        self.logM_min = _set_param(numerics_dict, 'logM_min', 12.)
        self.logM_max = _set_param(numerics_dict, 'logM_max', 16.)
        self.z_min = _set_param(numerics_dict, 'z_min', 0.005)
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

        self.Npoints_theta = _set_param(numerics_dict, 'Npoints_theta', 1000)

        self.pixel_radius = _set_param(numerics_dict, 'pixel_radius', None)
        self.pixel_sidelength = _set_param(numerics_dict, 'pixel_sidelength', None)
        self.Wiener_filter = _set_param(numerics_dict, 'Wiener_filter', None)
        self.gaussian_kernel_FWHM = _set_param(numerics_dict, 'gaussian_kernel_FWHM', None)

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
        self.larr = 10.**np.linspace(
            np.log10(self.ell_min),
            np.log10(self.ell_max),
            num = self.N_ell
            )
        j_0_n = jn_zeros(0, self.Npoints_theta)
        self.scaled_real_theta_grid = j_0_n/j_0_n[-1] # normalized to unity
        self.scaled_reci_theta_grid = j_0_n # normalized to unity
#}}}

class cosmology(object) :#{{{
    # some fundamental constants
    c0 = 2.99792458e5 # km/s
    GN = 4.30091e-9 # Mpc/Msun*(km/s)**2
    delta_c = 1.686
    hPl = 6.62607004e-34 # SI
    kBoltzmann = 1.38064852e-23 # SI
    def __init__(self, cosmo_dict = {None}) :#{{{
        
        print(cosmo_dict)
        # basic cosmology parameters
        self.h = _set_param(cosmo_dict, 'h', 0.7)
        self.Om = _set_param(cosmo_dict, 'Om0', 0.3)
        self.Ob = _set_param(cosmo_dict, 'Ob0', 0.046)
        self.As = _set_param(cosmo_dict, 'As', 2.1e-9)
        self.pivot_scalar = _set_param(cosmo_dict, 'pivot_scalar', 0.05/self.h)
        self.w = _set_param(cosmo_dict, 'w', -1)
        self.ns = _set_param(cosmo_dict, 'ns', 0.97)
        self.Mnu = _set_param(cosmo_dict, 'Mnu', 0.)
        self.Neff = _set_param(cosmo_dict, 'Neff', 0.)
        self.TCMB = _set_param(cosmo_dict, 'TCMB', 2.726)
        self.P0 = _set_param(cosmo_dict, 'pressure_profile_P0', 18.1)

        # various definitions, should hopefully not need much change
        self.mass_def_initial = _set_param(cosmo_dict, 'mass_def_initial', '200m')
        self.mass_def_kappa_profile = _set_param(cosmo_dict, 'mass_def_profile', 'vir')
        self.mass_def_Tinker = _set_param(cosmo_dict, 'mass_def_Tinker', '200m')
        self.mass_def_Batt = _set_param(cosmo_dict, 'mass_def_Batt', '200c')
        self.r_out_def = _set_param(cosmo_dict, 'r_out_def', 'vir')
        self.r_out_scale = _set_param(cosmo_dict, 'r_out_scale', 2.5)
        self.concentration_model = _set_param(cosmo_dict, 'concentration_model', 'duffy08')
        self.halo_profile = _set_param(cosmo_dict, 'halo_profile', 'nfw')
        self.HMF_fuction = _set_param(cosmo_dict, 'HMF_function', 'Tinker10')

        # derived quantities
        self.OL = 1.-self.Om # dark energy
        self.Oc = self.Om - self.Ob # (cold) dark matter
        self.H0 = 100.*self.h
        self.rhoM = self.Om*2.7753e11

        self.bias_arr = None
        self.hmf_arr = None
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
        if os.path.isfile(path + 'HMF_and_bias.npz') and not numerics.debugging :
            raise RuntimeError('HMF_and_bias have already been computed.')
        Npoints_M = numerics.Npoints_M
        Npoints_z = numerics.Npoints_z
        logM_grid = numerics.logM_grid
        z_grid = numerics.z_grid
        self.hmf_arr = np.empty((Npoints_M, Npoints_z))
        self.bias_arr = np.empty((Npoints_M, Npoints_z))
        for ii in xrange(Npoints_M) :
            start = time()
            for jj in xrange(Npoints_z) :
                z = z_grid[jj]
                M = self.convert_mass(10.**logM_grid[ii], z, self.mass_def_initial, self.mass_def_Tinker)
                sigma = self.__sigma(M, z)
                chi_int = self.__chi_integral(M, z)
                self.hmf_arr[ii,jj] = self.dndM(M, z, sigma, chi_int, self.HMF_fuction)
                self.bias_arr[ii,jj] = cosmology.__bz_Tinker2010(cosmology.delta_c/sigma)
            end = time()
            if (ii%4 == 0) and numerics.verbose :
                print str((end-start)/60.*(Npoints_M-ii)) + ' minutes remaining in create_HMF_and_bias.'
        np.savez(
            path + 'HMF_and_bias.npz',
            hmf = self.hmf_arr,
            bias = self.bias_arr,
            )
    #}}}
#}}}

class profiles(object) :#{{{
    def __init__(self, cosmo, num, param_dict = {None}) :#{{{
        # checks whether path contains profiles, convolved_profiles, and tildes files

        self.cosmo = cosmo
        self.num = num
        self.Sigma_crit = lambda z: 1.6625e18*cosmo.angular_diameter_distance(num.z_source)/cosmo.angular_diameter_distance(z)/cosmo.angular_diameter_distance_z1z2(z, num.z_source)
        self.theta_out_arr = None
        self.signal_arr = None
        self.convolved_signal_arr = None
        self.tilde_arr = None
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
    def __y_profile(theta_grid, M200c, d_A, r200c, r_out, y_norm, theta_out, h, z, P0) :#{{{
        out_grid = np.empty(len(theta_grid))
        #P0=18.1*(M200c/(1.e14*h))**(0.154)*(1.+z)**(-0.758)
        P0=P0*(M200c/(1.e14*h))**(0.154)*(1.+z)**(-0.758)
        xc=0.497*(M200c/(1.e14*h))**(-0.00865)*(1.+z)**(0.731)
        beta=4.35*(M200c/(1.e14*h))**(0.0393)*(1.+z)**(0.415)
        alpha=1.
        gamma=-0.3
        Jacobian = 1./h # the integration is now over Mpc/h-l
        lin = 0.
        lout = np.sqrt(r_out**2. - (np.tan(theta_grid)*d_A)**2.)
        for ii in prange(len(theta_grid)) :
            theta = theta_grid[ii]
            if (theta >= theta_out) :
                out_grid[ii] = 0. #y=0 outside cluster boundary
            else :
                integrand = lambda l : Jacobian*P0*(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**gamma*(1.+(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200c/xc)**alpha)**(-beta)
                integration_grid = np.linspace(lin, lout[ii], 1000)
                integral = np.trapz(integrand(integration_grid), x = integration_grid)
                out_grid[ii] =  y_norm * integral * 2.
        return out_grid
    #}}}
    def __generate_profile(self, M, z) :#{{{
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
            rs = rvir/cvir
            rho0 = Mvir/(4.*np.pi*rs**3.*(np.log(1.+cvir)-cvir/(1.+cvir)))
            rhoM = self.cosmo.astropy_cosmo.Om0*(1.+z)**3.*2.775e7/self.cosmo.h**2.
            critical_surface_density = self.Sigma_crit(z)
            signal_fct = lambda t: profiles._kappa_profile(
                t,
                rho0, rhoM, rs, r_out, d_A, theta_out, critical_surface_density
                )
            signal_prof = map(signal_fct, theta_grid)
        elif self.num.signal_type is 'tSZ' :
            M200c = self.cosmo.convert_mass(M, z, self.cosmo.mass_def_initial, self.cosmo.mass_def_Batt)
            r200c = (3.*M200c/4./np.pi/200./rhoc)**0.333333333333
            P_norm = 2.61051e-18*(self.cosmo.Ob/self.cosmo.Om)*H**2.*M200c/r200c
            y_norm = 4.013e-6*P_norm*self.cosmo.h**2.
            signal_prof = profiles.__y_profile(
                theta_grid,
                M200c, d_A, r200c, r_out, y_norm, theta_out, self.cosmo.h, z, self.cosmo.P0
                )
        else :
            raise RuntimeError('Unsupported signal type in __generate_profile.')
        return theta_out, signal_prof
    #}}}
    def create_profiles(self, path) :#{{{
        if (os.path.isfile(path + 'profiles.npz') and not self.num.debugging) or ((self.signal_arr is not None) and (self.theta_out_arr is not None)):
            raise RuntimeError('Profiles already exist.')
        self.signal_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z, self.num.Npoints_theta))
        self.theta_out_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z))
        start1 = time()
        for ii in xrange(self.num.Npoints_M) :
            start = time()
            for jj in xrange(self.num.Npoints_z) :
                self.theta_out_arr[ii,jj], self.signal_arr[ii,jj] = self.__generate_profile(
                    10.**self.num.logM_grid[ii],
                    self.num.z_grid[jj],
                    )
            end = time()
            if (ii%4 == 0) and self.num.verbose :
                print str((end-start)/60.*(self.num.Npoints_M-ii)) + ' minutes remaining in create profiles.'
        end1 = time()
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
        if (self.num.Wiener_filter is not None) :
            print 'Using Wiener filter '# + self.num.Wiener_filter
            self.convolve_with_Wiener = True
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
                if self.num.gaussian_kernel_FWHM is not None :
                    Window *= profiles._gaussian_pixel_window_function(self.num.scaled_reci_theta_grid*self.num.gaussian_kernel_FWHM/self.theta_out_arr[ii,jj])
                if self.convolve_with_Wiener :
                    Window *= self.num.Wiener_filter(self.num.scaled_reci_theta_grid/self.theta_out_arr[ii,jj])
                reci_signal = reci_signal * Window
                _,self.convolved_signal_arr[ii,jj,:] = DHTobj.apply(reci_signal)
                self.convolved_signal_arr[ii,jj,:] *= (self.num.scaled_reci_theta_grid[-1]**2.)
                d = np.diff(self.convolved_signal_arr[ii,jj,:])
            end = time()
            if (ii%4 == 0) and self.num.verbose :
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
        self.tilde_arr = np.empty((self.num.Npoints_M, self.num.Npoints_z, len(self.num.lambda_grid)), dtype = np.complex64)
        for ii in xrange(self.num.Npoints_M) :
            start = time()
            for jj in xrange(self.num.Npoints_z) :
                try :
                    spl_interpolator = CubicSpline(signal[ii,jj,:][::-1], self.num.scaled_real_theta_grid[::-1], extrapolate = False) # theta ( signal )
                    der_interpolator = spl_interpolator.derivative() # dtheta/dsignal
                    theta_of_signal = np.nan_to_num(spl_interpolator(self.num.signal_grid))
                    dtheta_dsignal = np.fabs(np.nan_to_num(der_interpolator(self.num.signal_grid)))
                    self.tilde_arr[ii,jj,:] = np.fft.rfft(theta_of_signal*dtheta_dsignal)
                    self.tilde_arr[ii,jj,:] *= 2.*np.pi*theta_out[ii,jj]**2.*(self.num.signal_grid[1]-self.num.signal_grid[0])
                    self.tilde_arr[ii,jj,:] -= self.tilde_arr[ii,jj,0] # subtract the zero mode
                except ValueError :
                    warn('FT failed in (' + str(ii) + ',' + str(jj) + ').\nThis is caused by numerical issues in the convolution,\nwhich make the profile not monotonically decreasing.\nFalling back to slower method.', UserWarning)
                    self.tilde_arr[ii,jj,:] = theta_out[ii,jj]**2. * profiles.__tilde_lambda_loop(
                        self.num.scaled_real_theta_grid,
                        signal[ii,jj,:],
                        2.*self.num.lambda_grid
                        )
            end = time()
            if (ii%4 == 0) and self.num.verbose :
                print str((end-start)/60.*(self.num.Npoints_M-ii)) + ' minutes remaining in create_tildes.'
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
#}}}

class PDF(object) :#{{{
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
            if (ii%4 == 0) and self.num.verbose :
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
        self.P_uncl = np.fft.irfft(self.Ptilde_uncl)
        self.P_cl = np.fft.irfft(self.Ptilde_cl)
        np.savez(
            path + 'P.npz',
            P_uncl = self.P_uncl,
            P_cl = self.P_cl,
            signal_min = self.num.signal_min,
            signal_max = self.num.signal_max
            )
    #}}}
#}}}

# WORKFLOW EXAMPLE
if __name__ == '__main__' :#{{{
    path = './test/'
    cosmo = cosmology(
        # set these to the values in your fiducial cosmology
        {
        'h': 0.7,
        'Om0': 0.25,
        'Ob0': 0.043,
        'As': 2.71826876e-09,
        'pivot_scalar': 0.002,
        'w': -1.0,
        'ns': 0.96,
        'Mnu': 0.0,
        'Neff': 0.0,
        'TCMB': 2.726
        }
        )

    # read in Wiener filter
    wf = np.array([pickle.load(open('act/ell.pkl')), \
                   pickle.load(open('act/SzWienerFilter.pkl'))]).T
    wf[:, 1] /= np.max(wf[:, 1])
    wf_interp = interp1d(wf[:, 0], wf[:, 1], \
                         bounds_error=False, fill_value=0.0)

    num = numerics(
        {

        # if this option is set to False, an error is thrown if you try
        # to compute something that already exists in path.
        'debugging': True,

        'verbose': True,

        # you probably want to keep this as tSZ, there is some code
        # for weak lensing convergence as well
        'signal_type': 'tSZ',

        # number of datapoints for various grids.
        # The values here are reasonably conservative.
        ###'Npoints_theta': 1000,
        'Npoints_theta': 200, # IS THIS OKAY?
        'Npoints_M': 50,
        'Npoints_z': 51,
        
        # grid boundaries
        ###'logM_min': 12.,
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
        ###'pixel_sidelength': (np.pi/180.)*0.41/60.,
        'pixel_sidelength': 0.0001440,

        # smoothing and filtering
        'Wiener_filter': lambda ell: wf_interp(ell),
        'gaussian_kernel_FWHM': 1.4 / 60.0 * np.pi / 180.0
        }
        )

    cosmo.create_HMF_and_bias(path, num)
    pr = profiles(cosmo, num)
    pr.create_profiles(path)
    pr.create_convolved_profiles(path)
    pr.create_tildes(path)
    p = PDF(cosmo, num, pr)
    p.create_alpha0(path)
    p.create_P_tilde(path)

    # this creates the final result.
    # it is stored in the file path + 'P.npz'
    # this file has fields:
    #      P_uncl     : the PDF without clustering contribution
    #      P_cl       : the PDF including clustering contribution
    #      signal_min : minimum signal (0)
    #      signal_max : maximum signal
    # the PDF values are on an equally spaced grid between signal_min and signal_max
    p.create_P(path)
#}}}
