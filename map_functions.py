import numpy as np
from scipy.integrate import quad
from scipy.integrate import trapz
from colossus.halo import mass_adv
from colossus.halo import mass_so
import time
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from numba import jit

# Auxiliary functions (window function, spectral function) #{{{
def W(x):
    # the window function's Fourier transform
    return 3.* (np.sin(x) - x*np.cos(x)) / x**3.

def chi(x):
    # the chi function [derivative of the window function]
    return ( (x**2 - 3.)*np.sin(x) + 3.*x*np.cos(x) ) / x**3

def g(x):# Spectral Function
    return x*(np.tanh(0.5*x))**(-1.) - 4.

def T(cosmology, y):
    return cosmology['TCMB'] * cosmology['gnu'] * y * 1e6 # in uK
#}}}

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
def sigma(M, z, numerics, cosmology):#{{{
    RM = (3. * M / (4. * np.pi * cosmology['rhoM'] ) )**(1./3.)
    integrand = lambda k : (1./(2.*np.pi**2))*( k**2 * cosmology['PK'](z, k) * (W(k*RM))**2 )
    sigmasq, err = quad(integrand, numerics['k_min'], numerics['k_max'], limit = 100)
    return np.sqrt(sigmasq)
#}}}
def chi_integral(M, z, numerics, cosmology):#{{{
    # computes the chi-integral [which is close to the derivative of sigma] at z = 0
    RM = (3. * M / (4. * np.pi * cosmology['rhoM'] ) )**(1./3.)
    integrand = lambda lk : (1.*np.log(10.)/np.pi**2)*( (10.**(lk))**3 * cosmology['PK'](z, (10.**lk)) * W((10.**lk)*RM) * chi((10.**lk)*RM) )
    integral, err = quad(integrand, np.log10(numerics['k_min']), np.log10(numerics['k_max']), limit = 100)
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
def Mint_d2(Mlow, Mhigh, z, numerics, cosmology):
    logM_grid = np.linspace(np.log10(Mlow), np.log10(Mhigh), num = 5)
    integrand_grid = np.zeros(len(logM_grid))
    for ii in xrange(len(logM_grid)):
        M = 10.**logM_grid[ii]
        integrand_grid[ii] = M * np.log(10) * d2(M, z, numerics, cosmology)
    return trapz(integrand_grid, x = logM_grid)
def zint_d2(Mlow, Mhigh, zlow, zhigh, numerics, cosmology):
    z_grid = np.linspace(zlow, zhigh, num = 5)
    integrand_grid = np.zeros(len(z_grid))
    for ii in xrange(len(z_grid)):
        z = z_grid[ii]
        integrand_grid[ii] = Mint_d2(Mlow, Mhigh, z, numerics, cosmology)
    return trapz(integrand_grid, x = z_grid)
#}}}


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

#################################
###### Map making functions #####
#################################
# ( only these functions should be called from outside )
def map_generate_dndOmega(numerics, cosmology, path):#{{{
    dndOmega = np.zeros((numerics['map_Npoints_M'], numerics['map_Npoints_z']))
    for ii in xrange(numerics['map_Npoints_M']):
        start = time.time()
        for jj in xrange(numerics['map_Npoints_z']):
            dndOmega[ii,jj] = zint_d2(
                    10.**numerics['map_logM_boundaries'][ii],
                    10.**numerics['map_logM_boundaries'][ii+1],
                    numerics['map_z_boundaries'][jj],
                    numerics['map_z_boundaries'][jj+1],
                    numerics,
                    cosmology
                    )
        end = time.time()
        if numerics['verbose'] :
            print str((numerics['map_Npoints_M']-ii)*(end-start)/60.) + ' minutes remaining in map_generate_dndOmega'
    # write to file
    np.savez(path + '/dndOmega.npz', dndOmega = dndOmega)
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
def map_generate_final_map(numerics, cosmology, path, index):#{{{
    start_total = time.time()
    f = np.load(path + '/dndOmega.npz')
    dndOmega = f['dndOmega']
    f = np.load(path + '/yprofiles.npz')
    thetas = f['thetas']
    yprofiles = f['yprofiles']
    # consistency checks#{{{
    if dndOmega.shape[0] != numerics['map_Npoints_M']:
        print 'dndOmega mass problem'
        print 'dndOmega has ' + str(dndOmega.shape[0]) + ' mass entries'
        print 'while we have ' + str(numerics['map_Npints_M']) + ' mass grid points'
        return
    if dndOmega.shape[1] != numerics['map_Npoints_z']:
        print 'dndOmega redshift problem'
        print 'dndOmega has ' + str(dndOmega.shape[1]) + ' redshift entries'
        print 'while we have ' + str(numerics['map_Npints_z']) + ' redshift grid points'
        return
    if len(yprofiles) != numerics['map_Npoints_M']:
        print 'yprofiles mass problem'
        print 'yprofiles has ' + str(len(yprofiles)) + ' mass entries'
        print 'while we have ' + str(numerics['map_Npoints_M']) + ' mass grid points'
        return
    if len(yprofiles[0]) != numerics['map_Npoints_z']:
        print 'yprofiles redshift problem'
        print 'yprofiles has ' + str(len(yprofiles[0])) + ' redshift entries'
        print 'while we have ' + str(numerics['map_Npoints_z']) + ' redshift grid points'
        return
    #}}}
    # prepare the final map
    final_map = np.zeros((int(numerics['map_size']/numerics['map_pixel_size']),int(numerics['map_size']/numerics['map_pixel_size'])))
    map_area = final_map.shape[0]*numerics['map_pixel_size']*final_map.shape[1]*numerics['map_pixel_size']
    for ii in xrange(numerics['map_Npoints_M']):
        if numerics['verbose'] :
            print str(ii)
        start = time.time()
        for jj in xrange(numerics['map_Npoints_z']):
            if numerics['map_Poisson']:
                cluster_number = np.random.poisson(dndOmega[ii,jj]*map_area)
            else :
                middle = dndOmega[ii,jj]*map_area
                lower = np.floor(middle)
                upper = np.ceil(middle)
                if np.random.rand() < (middle - lower):
                    cluster_number = int(upper)
                else:
                    cluster_number = int(lower)
            if cluster_number > 0:
                central_pixels_x = np.random.random_integers(0, final_map.shape[0]-1, size = cluster_number)
                central_pixels_y = np.random.random_integers(0, final_map.shape[1]-1, size = cluster_number)
                random_offset_x = np.random.rand()-0.5
                random_offset_y = np.random.rand()-0.5
                t = thetas[ii][jj]
                y = yprofiles[ii][jj]
                y_interpolator = interp1d(t, y, kind = 'cubic', bounds_error = False, fill_value = (max(y), 0.))
                T_of_theta = lambda theta: T(cosmology, y_interpolator(theta))
                this_map = np.zeros((2*int(max(t)/numerics['map_pixel_size'])+5, 2*int(max(t)/numerics['map_pixel_size'])+5)) # want central pixel to be on center of the cluster
                pixel_indices_x = np.linspace(-(this_map.shape[0]-1)/2.,(this_map.shape[0]-1)/2.,num = this_map.shape[0])
                pixel_indices_y = np.linspace(-(this_map.shape[1]-1)/2.,(this_map.shape[1]-1)/2.,num = this_map.shape[1])
                # average over angles
                nn = 0
                for kk in xrange(-numerics['map_grid_per_pixel'],numerics['map_grid_per_pixel']+1):
                    for ll in xrange(-numerics['map_grid_per_pixel'],numerics['map_grid_per_pixel']+1):
                        angles = numerics['map_pixel_size'] * np.sqrt(np.add.outer((pixel_indices_x + random_offset_x + float(kk)/float(numerics['map_grid_per_pixel']+0.5))**2.,(pixel_indices_y + random_offset_y + float(ll)/float(numerics['map_grid_per_pixel']+0.5))**2.))
                        this_map += T_of_theta(angles)
                        nn += 1
                this_map *= 1./float(nn)
                final_map = throw_clusters(cluster_number, final_map, this_map, central_pixels_x, central_pixels_y)
        end = time.time()
        if numerics['verbose'] :
            print str((numerics['map_Npoints_M']-ii)*(end-start)/60.) + ' minutes remaining in map_generate_final_map'
            print 'I am in index = ' + str(index)
    # need to take a subset of the final map, since otherwise we're getting a bias (centres of clusters are currently always in the map)
    spare_pixels_horizontal = int((1.-numerics['map_fraction'])/2.*final_map.shape[0])
    spare_pixels_vertical   = int((1.-numerics['map_fraction'])/2.*final_map.shape[1])
    final_map = final_map[spare_pixels_horizontal:-spare_pixels_horizontal-1,spare_pixels_vertical:-spare_pixels_vertical-1]
    end_total = time.time()
    if numerics['verbose'] :
        print 'used ' + str((end_total - start_total)/60.) + ' minutes in total'
#    np.savez(path + '/final_map_' + str(index) + '.npz', final_map = final_map)


#############################################
# TODO : this is only for validation purposes
    bin_boundaries = -np.arange(0., 100.)[::-1]
    hist, edges = np.histogram(final_map.flatten(), bin_boundaries)
    hist = hist/float(sum(hist))
    np.savez(path+'/p_' + str(index) + '.npz', p = hist)
#############################################
#}}}

