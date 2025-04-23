from dm_simulator_wrapper import dirtymap_simulator_wrapper, unpackArraytoStruct, chordParams
import numpy as np
import time
import pickle
import sys
sys.path.append("./dm_noise_simulator")
from dm_noise_simulator_wrapper import dm_noise_simulator_wrapper_gpu
from get_noise_distribution import not_autocorr_stdv

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

def generate_locations (n, center_theta, center_phi, angular_offset_scale_theta, angular_offset_scale_phi, seed):
    rng = np.random.default_rng(seed=seed)
    randomphi = rng.uniform(center_phi-angular_offset_scale_phi, center_phi+angular_offset_scale_phi, size=n)
    randomz = rng.uniform(np.cos(center_theta+angular_offset_scale_theta), np.cos(center_theta-angular_offset_scale_theta), size=n)
    us_output = np.empty([n,3])
    for i in range(n):
        us_output[i] = np.array([np.cos(randomphi[i])*np.sqrt(1-randomz[i]**2), np.sin(randomphi[i])*np.sqrt(1-randomz[i]**2), randomz[i]])
    return us_output

def get_radec_pixelvecs (nx,ny, base_theta, base_phi, delta_theta, delta_phi):
	vecs = np.empty([nx*ny,3])
	phi_array = np.linspace(base_phi-delta_phi/2, base_phi+delta_phi/2, nx)
	theta_array = np.linspace(base_theta-delta_theta/2, base_theta+delta_theta/2, ny)
	for i in range(nx*ny):
		vecs[i] = ang2vec(theta_array[i//nx], phi_array[i%nx])
	return vecs

sol = 299792458.0 #m/s
omega = 2*np.pi/(3600*24)

if __name__ == "__main__":
    t1 = time.time()
    spectra_file = np.load("test_spectra1000.npz")
    f = spectra_file["freq"].astype(np.float32)[:1024]
    wavelengths = sol*1e3/(f*1e6)

    chord_dec = 49.322
    dish_diameter = 6 #m
    m1 = 22 #ew
    m2 = 24 #ns
    L1 = 6.3
    L2 = 8.5
    seed = 1234

    base_theta = np.deg2rad(90-49.322)
    base_phi = 0
    nx = 800
    ny = 200
    extent1 = np.deg2rad(24)
    extent2 = np.deg2rad(6)

    spectra = spectra_file["spectra"][:,:1024]
    source_us = generate_locations (spectra.shape[0], base_theta, base_phi, extent2, extent1, seed)

    chord_thetas = np.asarray([np.deg2rad(90-chord_dec)], dtype=np.float32)
    cp = chordParams(thetas = unpackArraytoStruct(chord_thetas),
                    initial_phi_offset = np.deg2rad(10),
                     m1=m1, m2=m2, L1=L1, L2=L2, chord_zenith_dec = 49.322, D = dish_diameter, noise = 6.2522,
                    delta_tau = np.deg2rad(0.5)/omega, time_samples=41)

    u = get_radec_pixelvecs(nx, ny, base_theta, base_phi, extent2, extent1).astype(np.float32)

    source_dirtymap = dirtymap_simulator_wrapper (u, wavelengths, source_us, spectra, 0.01, cp)
    t2 = time.time()
    print("Source simulator took", t2-t1, "seconds")

    ang_resolution = 2 #deg
    deg_distance_to_count = 8
    ntimesamples = int(360/ang_resolution)
    noise = not_autocorr_stdv(3600.0*24/ntimesamples) /1000 #mJy
    baselines, baseline_counts = get_baseline_counts()
    baselines = baselines.astype(np.float32)

    noise_dirtymap = dm_noise_simulator_wrapper_gpu(noise, u, baselines, baseline_counts, wavelengths, chord_dec, dish_diameter, deg_distance_to_count, ntimesamples, seed)

    t3 = time.time()
    print("Noise simulator took", t3-t2, "seconds")

    combined_dirtymap = source_dirtymap + noise_dirtymap

    dmDict = {
        "dirtymap": combined_dirtymap,
        "freq": f,
        "nx": nx,
        "ny": ny
    }

    dmfile = open("dirtymap.pickle", "wb")
    pickle.dump(dmDict,dmfile)
    dmfile.close()
