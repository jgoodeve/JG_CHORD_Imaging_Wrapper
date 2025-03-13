import numpy as np
import os
import ctypes
from numpy.ctypeslib import ndpointer

dmns_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"dmns.so"))
cpp_dm_noise_sim_function = dmns_lib.dm_noise_sim

cpp_dm_noise_sim_function.argtypes = \
    [ctypes.c_double, ndpointer(dtype=ctypes.c_double), ctypes.c_int,
    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_int), ctypes.c_int,
    ndpointer(dtype=ctypes.c_double), ctypes.c_int,
    ctypes.c_float, ctypes.c_double, ctypes.c_int, 
    ndpointer(dtype=ctypes.c_double)]
    #noise, u, npixels,
    #baselines, baseline counts, nbaselines,
    #wavelengths, nwavelengths,
    #telescope dec, dish diameter, ntimesamples
    #output noise map


def dm_noise_simulator_wrapper (noise, u, baselines, baseline_counts, wavelengths, dec, dish_diameter, ntimesamples):
    assert(baselines.shape[0] == baseline_counts.shape[0])
    assert(u.shape[1] == 3)
    assert(baselines.shape[1] == 2)
    noise_map = np.empty(u.shape[0]*wavelengths.shape[0])
    u_flattened = u.flatten().copy()
    cpp_dm_noise_sim_function(noise, u_flattened, u_flattened.shape[0]//3,
                               baselines, baseline_counts, baselines.shape[0],
                               wavelengths, wavelengths.shape[0],
                               dec, dish_diameter, ntimesamples,
                               noise_map)
    return noise_map
