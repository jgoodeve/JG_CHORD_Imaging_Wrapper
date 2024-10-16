import numpy as np
import ctypes

dms_lib = ctypes.CDLL("dms.so")
cuda_dirtymap_function = dms_lib.dirtymap

class floatArray(ctypes.Structure):
    _fields_ = [("p",ctypes.POINTER((ctypes.c_float))),("l",ctypes.c_uint)]

def unpackArraytoStruct (arr):
    return floatArray(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size)

class chordParams(ctypes.Structure):
    _fields_ = [("thetas",floatArray),
                ("initial_phi_offset",ctypes.c_float),
                ("m2", ctypes.c_ushort),
                ("L1", ctypes.c_double),
                ("L2", ctypes.c_double),
                ("CHORD_zenith_dec", ctypes.c_double),
                ("D", ctypes.c_double),
                ("delta_tau", ctypes.c_double),
                ("time_samples", ctypes.c_uint)]

cuda_dirtymap_function.argtypes = [floatArray, floatArray, floatArray, floatArray, ctypes.c_float, chordParams, ctypes.POINTER(ctypes.c_float)]
#u, wavelengths, source_u, source_spectra, brightness_threshold, chord params, dm

def ang2vec (theta,phi):
    return np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)]).T

def vec2ang(v):
    if v[2] > 0:
        theta = np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])
    elif v[2] < 0:
        theta = np.pi + np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])
    else:
        theta = np.pi/2
    
    if v[0] > 0:
        phi = np.arctan(v[1]/v[0])
    elif v[0] < 0 and v[1] >= 0:
        phi = np.arctan(v[1]/v[0]) + np.pi
    elif v[0] < 0 and v[1] < 0:
        phi = np.arctan(v[1]/v[0]) - np.pi
    elif v[0] == 0 and v[1] > 0:
        phi = np.pi/2
    elif v[0] == 0 and v[1] < 0:
        phi = np.pi/2
    else:
        phi = 0
    
    return theta, phi

def gaussian (x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

def generate_spectra (n,nchannels, center_theta, center_phi, angular_offset_scale_theta, angular_offset_scale_phi, seed=1234567):
    rng = np.random.default_rng(seed=seed)
    spectra = np.empty([n,nchannels])
    centers = rng.uniform(size=n)
    brightness =  rng.beta(2,5,size=n) * 12
    widths = rng.uniform(1.0/nchannels, 5.0/nchannels, size=n)
    x = np.linspace(0,1,nchannels)
    for i in range(n):
        spectra[i] = brightness[i] * gaussian(x, centers[i], widths[i])
    
    randomphi = rng.uniform(center_phi-angular_offset_scale_phi, center_phi+angular_offset_scale_phi, size=n)
    randomtheta = rng.uniform(center_theta-angular_offset_scale_theta, angular_offset_scale_theta, size=n) #yeah I know this is not supposed to be uniform. This function is just for demonstration purposes. It probably doesn't work at the north pole or something.
    us_output = np.empty([n,3])
    for i in range(n):
        us_output[i] = ang2vec(randomtheta[i],randomphi[i])
    
    return spectra, us_output

channel_width = (1500-300)/6000 #MHz
def get_coarse (freq):
    return 300 + ((freq-300)//channel_width)*channel_width

sol = 299792.458 #km/s
def z_to_center (z):
    return sol/210 / (1+z) #MHz

def get_tan_plane_pixelvecs (nx,ny, base_theta, base_phi, extent1, extent2):
    testvecs = np.empty([nx,ny,3])
    basevec = ang2vec(base_theta, base_phi)
    v1 = ang2vec(base_theta - np.pi/2, base_phi)
    v2 = np.cross(v1, basevec)
    
    xls = np.linspace(-1,1,nx) * np.tan(extent1)
    yls = np.linspace(-1,1,ny) * np.tan(extent2)
    testvecs = (basevec[np.newaxis, np.newaxis, :]
      + v1[np.newaxis, np.newaxis, :] * yls[:, np.newaxis, np.newaxis]
      + v2[np.newaxis, np.newaxis, :] * xls[np.newaxis, :, np.newaxis])

    #normalizing
    norms = np.linalg.norm(testvecs, axis=2)
    np.divide(testvecs[:,:,0],norms, testvecs[:,:,0])
    np.divide(testvecs[:,:,1],norms, testvecs[:,:,1])
    np.divide(testvecs[:,:,2],norms, testvecs[:,:,2])
    return testvecs

def dirtymap_simulator_wrapper (u, wavelengths, source_u, source_spectra, brightness_threshold, chord_params):
    dirtymap = np.empty(u.shape[0])
    cuda_dirtymap_function(
        unpackArraytoStruct (u.flatten()),
        unpackArraytoStruct (wavelengths),
        unpackArraytoStruct (source_u.flatten()),
        unpackArraytoStruct(source_spectra.flatten()),
        brightness_threshold,
        chord_params,
        unpackArraytoStruct (dirtymap)
    )
    return dirtymap

omega = 2*np.pi/(3600*24)

if __name__ == "__main__":
    nf = 2
    f = np.linspace(get_coarse(z_to_center(0.00))-nf*channel_width,get_coarse(z_to_center(0.00)),nf)

    base_theta = np.deg2rad(90-49.322)
    base_phi = 0
    nx = 200
    ny = 200
    extent1 = np.deg2rad(5.0/60 * nx)
    extent2 = np.deg2rad(5.0/60 * ny)

    spectra, source_us = generate_spectra (40,nf, base_theta, base_phi, extent2, extent1)

    chord_thetas = np.asarray([np.deg2rad(90-49.322)])
    cp = chordParams(thetas = unpackArraytoStruct(chord_thetas),
                    initial_phi_offset = np.deg2rad(10),
                     m1=22, m2=24, L1=8.5, L2=6.3, chord_zenith_dec = 49.322, D = 6.0
                    delta_tau = np.deg2rad(0.5)/omega, time_samples=20)
    
    u = get_tan_plane_pixelvecs(nx,ny, base_theta, base_phi, extent1, extent2)

    dirtymap = dirtymap_simulator_wrapper (u, f, source_us, spectra, 0.01, cp)
    np.save(dirtymap,"simulated dirtymap")