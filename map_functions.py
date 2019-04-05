import numpy as np
import time
from scipy.interpolate import interp1d
from numba import jit

# Compton-y --> temperature #{{{
def g(x):# Spectral Function
    return x*(np.tanh(0.5*x))**(-1.) - 4.
def T(cosmology, y):
    return cosmology['TCMB'] * cosmology['gnu'] * y * 1e6 # in uK
#}}}

#################################
###### Map making functions #####
#################################
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

# ( only these functions should be called from outside )
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
    np.savez(path + '/final_map_' + str(index) + '.npz', final_map = final_map)


#}}}
def map_generate_random_noise(numerics, cosmology, path, index) :#{{{
    noise_map = np.zeros((int(numerics['map_size']/numerics['map_pixel_size']),int(numerics['map_size']/numerics['map_pixel_size'])))
    onesvec = np.ones(noise_map.shape[0])
    inds = (np.arange(noise_map.shape[0])+0.5-noise_map.shape[0]/2.)/(noise_map.shape[0]-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    ell_scale_factor = 2.*np.pi/numerics['map_pixel_size']
    ell2d = R * ell_scale_factor
    Cell2d = cosmology['noise_power'](ell2d)
    random_array_for_T = np.random.normal(0,1,noise_map.shape)
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)
    FT_2d = np.sqrt(Cell2d) * FT_random_array_for_T
    y_map = np.fft.ifft2(np.fft.fftshift(FT_2d))
    y_map /= numerics['map_pixel_size']
    noise_map = np.real(y_map)
    noise_map = T(cosmology, noise_map)
    np.savez(path + '/noise_' + str(index) + '.npz', noise = noise_map)
#}}}
