from dm_simulator_wrapper import dirtymap_simulator_wrapper, unpackArraytoStruct, chordParams
import numpy as np
import time
import pickle


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

channel_width = (1500-300)/6000 #MHz
def get_coarse (freq):
    return 300 + ((freq-300)//channel_width)*channel_width

sol = 299792.458 #km/s
def z_to_center (z):
    return sol/210 / (1+z) #MHz

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
    randomtheta = rng.uniform(center_theta-angular_offset_scale_theta, center_theta+angular_offset_scale_theta, size=n) #yeah I know this is not supposed to be uniform. This function is just for demonstration purposes. It probably doesn't work at the north pole or something.
    us_output = np.empty([n,3])
    for i in range(n):
        us_output[i] = ang2vec(randomtheta[i],randomphi[i])
    
    return spectra, us_output

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

def get_radec_pixelvecs (nx,ny, base_theta, base_phi, delta_theta, delta_phi):
	vecs = np.empty([nx*ny,3])
	phi_array = np.linspace(base_phi-delta_phi/2, base_phi+delta_phi/2, nx)
	theta_array = np.linspace(base_theta-delta_theta/2, base_theta+delta_theta/2, ny)
	for i in range(nx*ny):
		vecs[i] = ang2vec(theta_array[i//nx], phi_array[i%nx])
	return vecs

omega = 2*np.pi/(3600*24)

if __name__ == "__main__":
    t1 = time.time()
    nf = 32
    f = np.linspace(get_coarse(z_to_center(0.00))-nf*channel_width,get_coarse(z_to_center(0.00)),nf, dtype=np.float32)[::-1]
    wavelengths = sol*1e3/(f*1e6)
    #test_wavelengths = np.asarray([0.21], dtype=ctypes.c_float)

    base_theta = np.deg2rad(90-49.322)
    base_phi = 0
    nx = 200
    ny = 200
    extent1 = np.deg2rad(12) #np.deg2rad(5.0/60 * nx)
    extent2 = np.deg2rad(3) #np.deg2rad(5.0/60 * ny)

    spectra, source_us = generate_spectra (200,nf, base_theta, base_phi, extent2, extent1)
    #test_spectra = np.asarray([[6.0]],dtype=ctypes.c_float)
    #test_source_us = np.asarray([ang2vec(base_theta,0)], dtype=ctypes.c_float)

    chord_thetas = np.asarray([np.deg2rad(90-49.322)], dtype=np.float32)
    cp = chordParams(thetas = unpackArraytoStruct(chord_thetas),
                    initial_phi_offset = np.deg2rad(10),
                     m1=22, m2=24, L1=8.5, L2=6.3, chord_zenith_dec = 49.322, D = 6.0,
                    delta_tau = np.deg2rad(0.5)/omega, time_samples=41)

    #u = get_tan_plane_pixelvecs(nx,ny, base_theta, base_phi, extent1, extent2).reshape([nx*ny,3]).astype(np.float32)
    u = get_radec_pixelvecs(nx, ny, base_theta, base_phi, delta_theta, delta_phi)

    dirtymap = dirtymap_simulator_wrapper (u, wavelengths, source_us, spectra, 0.01, cp)
    t2 = time.time()
    print("Dirtymap simulator took", t2-t1, "seconds")

    dmDict = {
        "dirtymap": dirtymap,
        "freq": f,
        "nx": nx,
        "ny": ny
    }

    dmfile = open("dirtymap.pickle", "wb")
    pickle.dump(dmDict,dmfile)
    dmfile.close()
