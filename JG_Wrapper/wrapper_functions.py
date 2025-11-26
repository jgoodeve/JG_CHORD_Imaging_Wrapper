import numpy as np
import os
import ctypes
from numpy.ctypeslib import ndpointer
from astropy import wcs
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import astropy.units as units
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.linalg import ishermitian
from scipy.special import j1


c = 3e8

dms_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"dms_fixpoint.so"))
cuda_dirtymap_function = dms_lib.dirtymap_caller
#u, wavelengths, source_u, source_spectra, brightness_threshold, chord params, dm

class floatArray(ctypes.Structure):
    _fields_ = [("p",ctypes.POINTER((ctypes.c_float))),("l",ctypes.c_uint)]

def unpackArraytoStruct (arr):
    assert(arr.dtype==ctypes.c_float or arr.dtype==np.float32)
    return floatArray(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size)

class chordParams(ctypes.Structure):
    _fields_ = [("thetas",floatArray),
                ("centre_phi",ctypes.c_float),
                ("initial_phi_offset",ctypes.c_float),
		("m1", ctypes.c_uint),
                ("m2", ctypes.c_uint),
                ("L1", ctypes.c_float),
                ("L2", ctypes.c_float),
                ("CHORD_zenith_dec", ctypes.c_float),
                ("D", ctypes.c_float),
                ("delta_tau", ctypes.c_float),
                ("time_samples", ctypes.c_uint)]

def get_coarse (freq):
    return 300 + ((freq-300)//channel_width)*channel_width

def dirtymap_simulator_wrapper (u, wavelengths, source_u, source_spectra, brightness_threshold, chord_params):
    assert(source_u.shape[0] == source_spectra.shape[0])
    assert(wavelengths.shape[0] == source_spectra.shape[1])
    dirtymap = np.empty(u.shape[0]*wavelengths.shape[0], dtype = np.float32)
    source_u_float = source_u.flatten().astype(ctypes.c_float)
    source_spectra_float = source_spectra.flatten().astype(ctypes.c_float)
    u_flattened = u.flatten().copy()
    cuda_dirtymap_function(
        unpackArraytoStruct (u_flattened),
        unpackArraytoStruct (wavelengths),
        unpackArraytoStruct (source_u_float),
        unpackArraytoStruct(source_spectra_float),
        ctypes.c_float(brightness_threshold),
        chord_params,
        dirtymap
    )
    return dirtymap

cuda_dirtymap_function.argtypes = [floatArray, floatArray, floatArray, floatArray, ctypes.c_float, chordParams, ndpointer(dtype=ctypes.c_float)]

def normalize(dirtymap,noise,A_beam,B_beam,frequencies,M,N,beamthresh = 0.25):

    dirtymap/=M**2
    dirtymap/=N**2 ## necessary for normalization

    beam = A_beam.copy()

    for i in range(len(frequencies)):

        beammax = np.max(A_beam[:,:,i])
        noise[:,:,i] *= np.sqrt(B_beam[:,:,i])/beammax
        beam[:,:,i] /= beammax
        dirtymap[:,:,i] /= beammax
        (noise[:,:,i])[A_beam[:,:,i]<beamthresh] = np.nan
        (dirtymap[:,:,i])[A_beam[:,:,i]<beamthresh] = np.nan

    return dirtymap,noise,beam

def make_wcs(centre_ra_dec,cellsize,imsize):

    '''
    Create a wcs object corresponding to a given pixel scale, image size, and image centre.

    parameters:

    centre_ra_dec: 2-tuple containing the desired centre RA and dec, in degrees

    cellsize: size of an individual pixel, in degrees per pixel

    imsize: 2-tuple of image dimensions, first the 0th axis, then the first axis, in pixels
    '''
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [(imsize[0]+1)/2,(imsize[1]+1)/2] ### centre pixel
    w.wcs.cdelt = np.array([-cellsize,cellsize]) ### pixel scale. The first
                                                # entry is negative because
                                                # RA decreases going left -> right
                                                # in an image where north is up
    w.wcs.crval = centre_ra_dec ### centre RA and DEC in DEG
    w.wcs.ctype = ["RA---AZP", "DEC--AZP"]
    
    return w

def gen_image_u(centre_ra_dec,cellsize,imsize):

    '''
    given a central position in the sky and desired image dimensions, 
    return a list of pixel [X,Y,Z] vectors on the unit sphere that can 
    be input to Hans' code. 

    parameters:

    centre_ra_dec: 2-tuple of (ra,dec) in degrees 
    - the desired image centre position.

    cellsize: desired image pixel scale in degrees/pixel.

    imsize: 2-tupe of image sidelengths, in pixels
    '''
    
    w = make_wcs(centre_ra_dec,cellsize,imsize)
    x = np.linspace(0,imsize[0]-1,imsize[0])
    y = np.linspace(0,imsize[1]-1,imsize[1])

    xx,yy = np.meshgrid(x,y)
    
    z_ = np.stack((xx,yy),axis = -1)
    z_ = z_.reshape((imsize[0]*imsize[1],2))
    
    AX_0_pix = z_[:,0]
    AX_1_pix = z_[:,1]
    
    pos = w.pixel_to_world(AX_0_pix,AX_1_pix)
    
    X = pos.cartesian.x.value
    Y = pos.cartesian.y.value
    Z = pos.cartesian.z.value
    
    u = np.stack((X,Y,Z),axis = -1)
    
    return u.astype(np.float32),w

def recover_net_beam(u, centre_phi, init_phi_off, dphi, N_times, freqs, survey_dec, imsize, antenna_diam = 6):

    ### first figure out the angle between each point and each pointing
    angles_block = np.zeros((len(u),N_times))
    Xpix = u[:,0]
    Ypix = u[:,1]
    Zpix = u[:,2]
    RA_p = np.deg2rad(centre_phi-init_phi_off+dphi*np.arange(N_times))
    dec_p = 0*RA_p+np.deg2rad(survey_dec) ## add the RA to make the arrays the same size
    X_poin = np.cos(dec_p)*np.cos(RA_p)
    Y_poin = np.cos(dec_p)*np.sin(RA_p)
    Z_poin = np.sin(dec_p)
    for i in range(N_times):
        dp = Xpix*X_poin[i]+Ypix*Y_poin[i]+Zpix*Z_poin[i]
        dp[dp>1] = 1. ## avoid floating point errors ever so slightly above 1
        angles_block[:,i] = np.arccos(dp)
    angles_block = angles_block.ravel()
    A_beam = np.zeros((imsize[1],imsize[0],len(freqs)))
    beam = np.zeros((imsize[1],imsize[0],len(freqs)))
    for i in range(len(freqs)):
        airy_x = 2*np.pi*(antenna_diam/2)*angles_block*freqs[i]/c
        beam_block = 0*np.copy(airy_x)
        beam_block[airy_x > 0] = ((2*j1(airy_x[airy_x > 0])/airy_x[airy_x > 0])**2)
        beam_block[airy_x <= 0] = 1
        beam_block = beam_block.reshape(imsize[1],imsize[0],N_times)
        A_beam_block = beam_block**2
        A_beam[:,:,i] = np.sum(A_beam_block,axis = -1)
        beam[:,:,i] = np.sum(beam_block,axis = -1)

    return A_beam.astype('float32'),beam.astype('float32')

def u(ra,dec):

    c = SkyCoord(ra=ra*units.degree, dec=dec*units.degree, frame='icrs')
    
    X = c.cartesian.x.value
    Y = c.cartesian.y.value
    Z = c.cartesian.z.value

    u_ = np.array([X,Y,Z])
    
    return u_.astype(np.float32)

def u_vec(ra,dec):

    c = SkyCoord(ra=ra*units.degree, dec=dec*units.degree, frame='icrs')
    
    X = c.cartesian.x.value
    Y = c.cartesian.y.value
    Z = c.cartesian.z.value

    u = np.stack((X,Y,Z),axis = -1)
    
    return u.astype(np.float32)

def return_close_sources(centre_RA,survey_dec,phi_offset,N_times,dphi,RA,Dec,FJy,spec_idx,limit = 4):

    pointing_RAs = (np.arange(0,N_times,1)*dphi+centre_RA-phi_offset)
    pointing_Decs = 0*pointing_RAs+survey_dec

    pointings = SkyCoord(pointing_RAs*units.deg,pointing_Decs*units.deg,frame = 'icrs')
    ## check if close
    A = (pointing_RAs[0] - limit/np.cos(np.deg2rad(survey_dec)))
    B = (pointing_RAs[-1] + limit/np.cos(np.deg2rad(survey_dec)))
    if (B<A) or (limit/np.cos(np.deg2rad(survey_dec)) > 180):
        firstcut = (np.abs(survey_dec-Dec) < limit)
    else:
        if A < 0:
            firstcut = (np.logical_or(RA>A%360, RA<B)) & (np.abs(survey_dec-Dec) < limit)
        elif B > 360:
            firstcut = (np.logical_or(RA<B%360, RA>A)) & (np.abs(survey_dec-Dec) < limit)
        else:
            firstcut = (RA>A) & (RA<B) & (np.abs(survey_dec-Dec) < limit)
    
    RA_firstcut = RA[firstcut]
    Dec_firstcut = Dec[firstcut]
    FJy_firstcut = FJy[firstcut]
    spec_idx_firstcut = spec_idx[firstcut]

    #firstcut_positions = SkyCoord(RA_firstcut*units.deg,Dec_firstcut*units.deg,frame = 'icrs')

    source_u = u_vec(RA_firstcut,Dec_firstcut)

    return source_u,FJy_firstcut,spec_idx_firstcut

def get_spectra(frequencies,F,s):

    freqsdiv1p4 = frequencies/1400e6 ## normalize
    freqstack = np.zeros((len(F),len(frequencies)))
    freqstack += freqsdiv1p4
    s_array = 0*freqstack.copy()
    for i in range(len(s_array)):
        s_array[i] += s[i]
    spec_cofac = freqsdiv1p4**s_array ## array of factors by which the original Fnu is multiplied
    F_array = 0*spec_cofac.copy()
    for j in range(len(F_array)):
        F_array[j] += F[j] ## div by 1000 to compensate for mJy.

    spectra = spec_cofac*F_array

    return spectra

def assign_variability(ListOrNumber, bins, probs, varlength = 300):

    if isinstance(ListOrNumber,int):
        length = ListOrNumber         #### this chunk finds the number of variabilities and indices to generate
    else:
        length = len(ListOrNumber)

    bounds = np.cumsum(probs)

    outcomes = []
    
    bounds = np.cumsum(probs)
    bounds = np.concatenate((np.array([0]),bounds)) ## bounds sets the intervals where a random (0,1) float falls
                                                    ## so values can be drawn with the specified probability
    
    for i in range(length):
        trial = np.random.rand()
        for j in range(len(bins)):
            if (trial>bounds[j]) & (trial<bounds[j+1]):
                outcomes.append(bins[j])

    variability_fractions = np.array(outcomes)

    variability_indices = np.random.randint(0,varlength,length)

    variability_starts = np.random.randint(1,7300,length)

    return variability_indices,variability_fractions,variability_starts

def get_density(density_coeff,F1_mJy,F2_mJy):

    return density_coeff*(F1_mJy**-0.85-F2_mJy**-0.85)

def spec_idx_dist(x):

    a = 4.3
    b = 1.6

    xp = x.copy()
    y = 0*x.copy()
    xp[x<-0.6] = 4.3*(x[x<-0.6]+0.6)-0.6
    xp[x>=-0.6] = 1.6*(x[x>=-0.6]+0.6)-0.6

    N = np.sqrt(2)*a*b/np.sqrt(np.pi)/(a+b)

    return N*np.exp(-0.5*(xp+0.6)**2)

def gen_spectral_indices(N):

    a = 4.3
    b = 1.6
    
    spec_idxs = np.array([])
    
    integral = np.sqrt(2)*a*b/np.sqrt(np.pi)/(a+b)
    
    while len(spec_idxs) < N:
        X = 3.5*np.random.rand(N)-1.5
        Z = np.random.rand(N)
        keep = Z < spec_idx_dist(X)/integral
        spec_idxs = np.concatenate((spec_idxs,X[keep]))
    spec_idxs = spec_idxs[:N]

    return spec_idxs

def gen_faint_background(density_coeff,density_index,F1_mJy,F2_mJy,phi1,phi2,theta1,theta2,frequencies):

    density = get_density(density_coeff,F1_mJy,F2_mJy) ## per degree squared
    
    omega = (180/np.pi)**2*(np.deg2rad(phi2)-np.deg2rad(phi1))*(np.sin(np.deg2rad(theta2))-np.sin(np.deg2rad(theta1))) ## square degrees
    
    N = int(omega*density)
    
    phi = []
    theta = []
    F = []
    
    while len(phi) < N:
            
        phis = phi1+(phi2-phi1)*np.random.rand(N)
        thetas = theta1+(theta2-theta1)*np.random.rand(N)
        costhetas = np.cos(np.deg2rad(thetas))
        costhetas_norm = costhetas/np.max(costhetas)
        condition = np.random.rand(N) < costhetas_norm
        phis_inc = phis[condition]
        thetas_inc = thetas[condition]
        phi += list(phis_inc)
        theta += list(thetas_inc)
    
    while len(F) < N:
    
        Flux = F1_mJy+(F2_mJy-F1_mJy)*np.random.rand(N)
        pdf = Flux**(-1.85)
        pdf_norm = pdf / np.max(pdf)
        condition = np.random.rand(N) < pdf_norm
        Flux_inc = Flux[condition]/1000 ## divide by 1000 because we generated using mJy
        F += list(Flux_inc)
    
    dec = np.array(theta[:N])
    ra = np.array(phi[:N])
    F = np.array(F[:N])
    s = gen_spectral_indices(N)

    u = u_vec(ra,dec)
    spectra = get_spectra(frequencies,F,s)

    return u,spectra

def apply_variability(spectra,T_in_days,varlib,inds,starts,fracs):

    new_spectra = spectra.copy()

    for i in range(len(spectra)):

        tseries = np.roll(varlib[inds[i]],starts[i])

        new_spectra[i] *= (1+fracs[i]*varlib[inds[i]][T_in_days])

    return new_spectra
