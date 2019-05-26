import numpy as np
from scipy.integrate import quad
from scipy.integrate import trapz
from colossus.halo import mass_adv
from colossus.halo import mass_so
import time
from numba import jit

# Auxiliary functions (window function related) #{{{
def W(x):
    # the window function's Fourier transform
    return 3.* (np.sin(x) - x*np.cos(x)) / x**3.

def chi(x):
    # the chi function [derivative of the window function]
    return ( (x**2 - 3.)*np.sin(x) + 3.*x*np.cos(x) ) / x**3
#}}}
def D(z, cosmology):
    # computes the growth function
    k = 0.01 # arbitrary choice
    return np.sqrt(cosmology['PK'](z, k) / cosmology['PK'](0., k))

#################################
## Mass definition conversions ##
#################################
def virial_radius(M, z, numerics, cosmology):#{{{
    #if cosmology['mass_def_initial'] != 'vir':
    #    print 'Initial Mass definition not virial'
    M_vir, r_vir, c_vir = mass_adv.changeMassDefinitionCModel(
            M,
            z,
            cosmology['mass_def_initial'],
            'vir',
            profile = cosmology['halo_profile'],
            c_model = cosmology['concentration_model']
            )
    return r_vir*1e-3
#}}}
def mass_to_M200c(M, z, cosmology):#{{{
    # converts M200m mass into M200c mass
    M200c, R200c, c200c = mass_adv.changeMassDefinitionCModel(
            M,
            z,
            cosmology['mass_def_initial'],
            cosmology['mass_def_batt'],
            profile = cosmology['halo_profile'],
            c_model = cosmology['concentration_model']
            )
    return M200c
#}}}
def mass_to_M200m(M, z, cosmology):#{{{
    # converts Mvir mass into M200m mass
    if cosmology['mass_def_initial'] == '200m':
        return M
    else:
        M200m, R200m, c200m = mass_adv.changeMassDefinitionCModel( 
                M,
                z,
                cosmology['mass_def_initial'],
                cosmology['mass_def_Tinker'],
                profile = cosmology['halo_profile'],
                c_model = cosmology['concentration_model']
                )
        return M200m
#}}}


#################################
###### y-profile functions ######
#################################
def yprof_batt(z, yprof_inputs, theta, numerics, cosmology):#{{{
    M200c = yprof_inputs['M200c']
    d_A = yprof_inputs['d_A']
    r200 = yprof_inputs['r200']
    r_out = yprof_inputs['r_out']
    y_norm = yprof_inputs['y_norm']
    theta_out = yprof_inputs['theta_out']
    ##
    if (theta >= theta_out):
    # added >= instead of > to avoid nans at integration boundary
        return 0. #y=0 outside cluster boundary
    else:
        # implement Battaglia fitting function for pressure profile and do the line-of-sight integral
        # Sep 27 --- removed h's here -- is this correct?
        P0=cosmology['pressure_profile_P0']*(M200c/(1.e14*(cosmology['h'])))**(0.154)*(1.+z)**(-0.758)
        xc=cosmology['pressure_profile_xc0']*(M200c/(1.e14*(cosmology['h'])))**(-0.00865)*(1.+z)**(0.731)
        beta=cosmology['pressure_profile_beta0']*(M200c/(1.e14*(cosmology['h'])))**(0.0393)*(1.+z)**(0.415)
        alpha=cosmology['pressure_profile_alpha']
        gamma=cosmology['pressure_profile_gamma']
        Jacobian = 1./cosmology['h'] # the integration is now over Mpc/h-l
        integrand = lambda l : Jacobian*P0*(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200/xc)**gamma*(1.+(np.sqrt(l**2.+d_A**2.*(np.tan(theta))**2.)/r200/xc)**alpha)**(-beta)
        lin=0. # start integral at d_A*theta and multiply by 2 below
        lout=np.sqrt(r_out**2. - (np.tan(theta)*d_A)**2.) #integrate to cluster boundary as defined above
        integral, err = quad(integrand, lin, lout, limit = 100)
        return y_norm * integral * 2.
#}}}
def compute_yprof(M, z, yprof_inputs, numerics, cosmology):#{{{
    M200c = yprof_inputs['M200c']
    d_A = yprof_inputs['d_A']
    r200 = yprof_inputs['r200']
    r_out = yprof_inputs['r_out']
    y_norm = yprof_inputs['y_norm']
    theta_out = yprof_inputs['theta_out']
    ##
    theta_min = 0.
    theta_grid_1 = np.linspace(theta_min , theta_out*numerics['theta_boundary'] , num = 2*numerics['N_theta'], endpoint = False)
    theta_grid_2 = np.linspace(theta_out*numerics['theta_boundary'], theta_out, num = numerics['N_theta'], endpoint = True)
    theta_grid = np.concatenate((theta_grid_1, theta_grid_2))
    yprof = np.zeros(len(theta_grid))
    for ii in xrange(len(theta_grid)):
        yprof[ii] = yprof_batt(z, yprof_inputs, theta_grid[ii], numerics, cosmology)
    return theta_grid, yprof
#}}}


#################################
### Mass function calculation ###
#################################
# Tinker Mass Functions#{{{
def hmf_Tinker2010(nu, z):
    # Eq 8 of Tinker10, parameters from Table 4
    z1 = min([z, 3.])
    # HMF only calibrated below z = 3, use the value for z = 3 at higher redshifts
    beta = 0.589 * (1. + z1)**(0.20)
    phi = -0.729 * (1. + z1)**(-0.08)
    eta = -0.243 * (1. + z1)**(0.27)
    gamma = 0.864 * (1. + z1)**(-0.01)
    alpha = 0.368
    return alpha * ( 1. + (beta*nu)**(-2.*phi)  ) * nu**(2.*eta) * np.exp(-0.5*gamma*nu**2)

def hmf_Tinker2008(sigma):
    B = 0.482
    d = 1.97
    e = 1.
    f = 0.51
    g = 1.228
    return B*((sigma/e)**(-d) + sigma**(-f)) * np.exp(-g/sigma**2.)
#}}}
def bz_Tinker2010(nu, cosmology):#{{{
    # Eq 6 of Tinker10, with parameters from Table 2
    Delta = 200. # currently our value -- do we need to experiment with this?
    y = np.log10(Delta)
    A = 1.0 + 0.24 * y * np.exp( - (4./y)**4 )
    a = 0.44 * y - 0.88
    B  =0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp( - (4./y)**4 )
    c = 2.4
    return 1. - A*nu**a/(nu**a + cosmology['delta_c']**a) + B * nu**b + C * nu**c
#}}}
def sigma(M, z, numerics, cosmology):#{{{
    RM = (3. * M / (4. * np.pi * cosmology['rhoM'] ) )**(1./3.)
    integrand = lambda k : (1./(2.*np.pi**2))*( k**2 * cosmology['PK'](z, k) * (W(k*RM))**2 )
    sigmasq, err = quad(integrand, numerics['k_min'], numerics['k_max'], limit = 100, epsrel = 1e-3)
    return np.sqrt(sigmasq)
#}}}
def chi_integral(M, z, numerics, cosmology):#{{{
    # computes the chi-integral [which is close to the derivative of sigma] at z = 0
    RM = (3. * M / (4. * np.pi * cosmology['rhoM'] ) )**(1./3.)
    integrand = lambda lk : (1.*np.log(10.)/np.pi**2)*( (10.**(lk))**3 * cosmology['PK'](z, (10.**lk)) * W((10.**lk)*RM) * chi((10.**lk)*RM) )
    integral, err = quad(integrand, np.log10(numerics['k_min']), np.log10(numerics['k_max']), limit = 100, epsrel = 1e-3)
    return integral
#}}}
def dndM(M, z, numerics, cosmology):#{{{
    s = sigma(M, z, numerics, cosmology)
    chi_int = chi_integral(M, z, numerics, cosmology)
    if numerics['hmf_function'] == 'Tinker10':
        f = hmf_Tinker2010((cosmology['delta_c']/s), z)
        return - (cosmology['delta_c'] * cosmology['rhoM'] * f * chi_int) / (2. * s**3 * M**2)
    if numerics['hmf_function'] == 'Tinker08':
        g = hmf_Tinker2008(s)
        return - (cosmology['rhoM'] * g * chi_int) / (2. * s**2 * M**2)
#}}}
# dndOmega functions #{{{
def d2(M, z, numerics, cosmology):
    d3 = dndM(M, z, numerics, cosmology)
    return (((cosmology['cosmo_object'].comoving_distance(z)).value)*cosmology['h'])**2. * ((cosmology['cosmo_object'].H(z)).value/(cosmology['c0']*cosmology['h']))**(-1.) * d3
vectorized_d2 = np.vectorize(d2, cache = True)
def Mint_d2(Mlow, Mhigh, z, numerics, cosmology):
    logM_grid = np.linspace(np.log10(Mlow), np.log10(Mhigh), num = 5)
    integrand_grid = 10.**logM_grid*np.log(10.)*vectorized_d2(10.**logM_grid, z, numerics, cosmology)
    return trapz(integrand_grid, x = logM_grid)
vectorized_Mint_d2 = np.vectorize(Mint_d2, cache = True)
def zint_d2(Mlow, Mhigh, zlow, zhigh, numerics, cosmology):
    z_grid = np.linspace(zlow, zhigh, num = 5)
    integrand_grid = vectorized_Mint_d2(Mlow, Mhigh, z_grid, numerics, cosmology)
    return trapz(integrand_grid, x = z_grid)
vectorized_zint_d2 = np.vectorize(zint_d2, cache = True)
#}}}

#################################
####### Physics functions #######
#################################
# ( only these functions should be called from outside )
def map_generate_dndOmega(numerics, cosmology, path):#{{{
    dndOmega = np.zeros((numerics['map_Npoints_M'], numerics['map_Npoints_z']))
    for ii in xrange(numerics['map_Npoints_M']):
        start = time.time()
        dndOmega[ii, :] = vectorized_zint_d2(
                                10.**numerics['map_logM_boundaries'][ii],
                                10.**numerics['map_logM_boundaries'][ii+1],
                                numerics['map_z_boundaries'][:-1],
                                numerics['map_z_boundaries'][1:],
                                numerics,
                                cosmology
                                )
        end = time.time()
        if numerics['verbose'] :
            print str((numerics['map_Npoints_M']-ii)*(end-start)/60.) + ' minutes remaining in map_generate_dndOmega'
    # write to file
    np.savez(path + '/dndOmega.npz', dndOmega = dndOmega)
#}}}
def map_generate_bias(numerics, cosmology, path) :#{{{
    if numerics['verbose'] :
        print 'Generating bias'
    bias_arr = np.zeros((numerics['map_Npoints_M'], numerics['map_Npoints_z']))
    for ii in xrange(numerics['map_Npoints_M']) :
        for jj in xrange(numerics['map_Npoints_z']) :
            M = 0.5*(10.**numerics['map_logM_boundaries'][ii]+10.**numerics['map_logM_boundaries'][ii+1])
            z = 0.5*(numerics['map_z_boundaries'][jj] + numerics['map_z_boundaries'][jj+1])
            nu = cosmology['delta_c'] / sigma(M, z, numerics, cosmology)
            bias_arr[ii,jj] = bz_Tinker2010(nu, cosmology)
    np.savez(
        path + '/bias.npz',
        bias = bias_arr
        )
#}}}
def map_generate_yprofiles(numerics, cosmology, path):#{{{
    yprofiles = [[[] for ii in xrange(numerics['map_Npoints_z'])] for jj in xrange(numerics['map_Npoints_M'])]
    thetas = [[[] for ii in xrange(numerics['map_Npoints_z'])] for jj in xrange(numerics['map_Npoints_M'])]
    # first index corresponds to M, second index to z
    for ii in xrange(numerics['map_Npoints_M']):
        start = time.time()
        for jj in xrange(numerics['map_Npoints_z']):
            # compute average M, z in each bin
            M = 0.5*(10**numerics['map_logM_boundaries'][ii] + 10**numerics['map_logM_boundaries'][ii+1])
            z = 0.5*(numerics['map_z_boundaries'][jj] + numerics['map_z_boundaries'][jj+1])
            H = ((cosmology['cosmo_object'].H(z)).value)/cosmology['h'] # h*km/s/Mpc
            d_A = ((cosmology['cosmo_object'].angular_diameter_distance(z)).value)*cosmology['h'] # Mpc / h
            r_vir = virial_radius(M, z, numerics, cosmology)
            M200c = mass_to_M200c(M, z, cosmology)
            rhoc=2.775e7*H**2. #critical density in units of M_sun*h^2/Mpc^3
            r200=(3.*M200c/4./np.pi/200./rhoc)**0.3333333 #Mpc/h
            P_norm = 2.61051e-18*(cosmology['cosmo_object'].Ob0/cosmology['cosmo_object'].Om0)*H**2.*M200c/r200 #pressure profile normalization in units of h^2*eV/cm^3
            y_norm = P_norm*4.013e-6 * cosmology['h']**2 #multiply by sigmaT/m_e/c^2*Mpcincm in units that agree with eV/cm^3 above; y_norm is dimensionless Compton-y
            if cosmology['r_out_def'] == 'vir':
                r_out = numerics['theta_max_scale'] * r_vir
            elif cosmology['r_out_def'] == '200':
                r_out = numerics['theta_max_scale'] * r200
            else:
                print 'Problem in r_out_def'
                return
            theta_out = np.arctan(r_out/d_A)
            yprof_inputs = {
                    'M200c': M200c,
                    'd_A': d_A,
                    'r200': r200,
                    'r_out': r_out,
                    'y_norm': y_norm,
                    'theta_out': theta_out
                    }
            t, y = compute_yprof(M, z, yprof_inputs, numerics, cosmology)
            t = t[np.where(y!=0)]
            y = y[np.where(y!=0)]
            thetas[ii][jj] = t
            yprofiles[ii][jj] = y
        end = time.time()
        if numerics['verbose'] :
            print str((numerics['map_Npoints_M']-ii)*(end-start)/60.) + ' minutes remaining in map_generate_yprofiles'
    np.savez(path + '/yprofiles.npz', thetas = thetas, yprofiles = yprofiles)
#}}}

#################################
## Generate linear density Cls ## 
#################################
def map_generate_linear_density_Cells(numerics, cosmology, path) :#{{{
    """
    generates the Cells
    """
    if numerics['verbose'] :
        print 'Generating linear density C_ell'
    lmin = int(numerics['ell_min_scale'] * np.pi / numerics['map_size'])
    lmax = int(numerics['ell_max_scale'] * np.pi / numerics['map_pixel_size'])
    ell_arr = np.arange(lmin, lmax)
    Cell_arr = np.zeros((len(ell_arr), numerics['map_Npoints_z']))
    for ii in xrange(len(ell_arr)) :
        for jj in xrange(numerics['map_Npoints_z']) :
            integrand = lambda z: ((cosmology['cosmo_object'].H(z)).value/(cosmology['c0']*cosmology['h']))*cosmology['PK'](z, ell_arr[ii]/((cosmology['cosmo_object'].comoving_distance(z)).value*cosmology['h']))/((cosmology['cosmo_object'].comoving_distance(z)).value*cosmology['h'])**2.
            #Cell_arr[ii,jj],err = quad(integrand, numerics['map_z_boundaries'][jj], numerics['map_z_boundaries'][jj+1])
            Cell_arr[ii,jj] = integrand(0.5*(numerics['map_z_boundaries'][jj]+numerics['map_z_boundaries'][jj+1])) # integrate with delta function kernel, hopefully precise enough
    np.savez(
        path + '/linear_density_Cells.npz',
        ell = ell_arr,
        Cell = Cell_arr
        )
#}}}
