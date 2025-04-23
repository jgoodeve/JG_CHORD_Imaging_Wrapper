import numpy as np
import time
import pickle

from dm_noise_simulator_wrapper import dm_noise_simulator_wrapper_gpu
from get_noise_distribution import not_autocorr_stdv

def get_baseline_counts():
    x = np.arange(m1)
    y = np.arange(m2)

    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()

    p = np.vstack([xx,yy]).T #positions

    baseline_counts = {}
    for i in range(m1*m2):
        for j in range(i,m1*m2):
            b = tuple(p[j]-p[i])
            if b in baseline_counts.keys():
                baseline_counts[b] += 1
            else:
                baseline_counts[b] = 1

    baselines = np.asarray(list(baseline_counts.keys()),dtype = np.float32) * np.array([L1, L2])
    counts = np.asarray(list(baseline_counts.values()),dtype=np.int32) 
    return baselines, counts

def ang2vec (theta,phi):
    return np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)]).T

def get_tan_plane_pixelvecs (nx,ny, base_theta, base_phi, extent1, extent2):
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

channel_width = (1500-300)/6000 #MHz
def get_coarse (freq):
    return 300 + ((freq-300)//channel_width)*channel_width

sol = 299792.458 #km/s
def z_to_center (z):
    return sol/210 / (1+z) #MHz

if __name__ == "__main__":
    t1 = time.time()
    
    m1 = 22 #ew
    m2 = 24 #ns
    L1 = 6.3
    L2 = 8.5
    baselines, baseline_counts = get_baseline_counts()
    baselines = baselines.astype(np.float32)

    nf = 32
    f = np.linspace(get_coarse(z_to_center(0.00))-nf*channel_width,get_coarse(z_to_center(0.00)),nf, dtype=np.float32)[::-1]
    wavelengths = sol*1e3/(f*1e6)
    
    base_theta = np.deg2rad(90-49.322)
    base_phi = 0
    nx = 600
    ny = 150
    extent1 = np.deg2rad(12)
    extent2 = np.deg2rad(3)
    
    chord_dec = 49.322
    dish_diameter = 6 #m
    ang_resolution = 2 #deg
    deg_distance_to_count = 8
    ntimesamples = int(360/ang_resolution)
    noise = not_autocorr_stdv(3600.0*24/ntimesamples) * 1e9 #1e9Jy

    #u = get_tan_plane_pixelvecs(nx,ny, base_theta, base_phi, extent1, extent2).reshape([nx*ny,3])
    u = get_radec_pixelvecs (nx,ny, base_theta, base_phi, extent2, extent1).astype(np.float32)
    #u = u[0][np.newaxis]
    
    dirtymap = dm_noise_simulator_wrapper_gpu(noise, u, baselines, baseline_counts, wavelengths, chord_dec, dish_diameter, deg_distance_to_count, ntimesamples)
    
    t2 = time.time()
    print("Dirtymap noise simulator took", t2-t1, "seconds")
    print("output:", dirtymap)

    dmDict = {
        "dirtymap": dirtymap,
        "freq": f,
        "nx": nx,
        "ny": ny
    }
    dmfile = open("dirtymap_noise.pickle", "wb")
    pickle.dump(dmDict,dmfile)
    dmfile.close()
