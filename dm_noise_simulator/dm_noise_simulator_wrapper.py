import numpy as np
import os
import ctypes
from numpy.ctypeslib import ndpointer

dmns_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"dmns.so"))
cpp_dm_noise_sim_function = dmns_lib.dm_noise_sim
cpp_dm_noise_sim_inst_function = dmns_lib.dm_noise_sim_instantaneous

cpp_dm_noise_sim_function.argtypes = \
    [ctypes.c_double, ndpointer(dtype=ctypes.c_double), ctypes.c_int,
    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_int), ctypes.c_int,
    ndpointer(dtype=ctypes.c_double), ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, 
    ndpointer(dtype=ctypes.c_double)]
    #noise, u, npixels,
    #baselines, baseline counts, nbaselines,
    #wavelengths, nwavelengths,
    #telescope dec, dish diameter, deg_distance_to_count, ntimesamples
    #output noise map

cpp_dm_noise_sim_inst_function.argtypes = \
    [ctypes.c_double, ndpointer(dtype=ctypes.c_double), ctypes.c_int,
    ndpointer(dtype=ctypes.c_double), ndpointer(dtype=ctypes.c_int), ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ndpointer(dtype=ctypes.c_double)]

def dm_noise_simulator_wrapper (noise, u, baselines, baseline_counts, wavelengths, dec, dish_diameter, deg_distance_to_count, ntimesamples):
    assert(baselines.shape[0] == baseline_counts.shape[0])
    assert(u.shape[1] == 3)
    assert(baselines.shape[1] == 2)
    noise_map = np.empty(u.shape[0]*wavelengths.shape[0])
    u_flattened = u.flatten().copy()
    cpp_dm_noise_sim_function(noise, u_flattened, u_flattened.shape[0]//3,
                               baselines, baseline_counts, baselines.shape[0],
                               wavelengths, wavelengths.shape[0],
                               dec, dish_diameter, deg_distance_to_count, ntimesamples,
                               noise_map)
    return noise_map
    
def dm_noise_simulator_instantaneous_wrapper (noise, u, baselines, baseline_counts, wavelength, dec, dish_diameter):
    assert(baselines.shape[0] == baseline_counts.shape[0])
    assert(u.shape[1] == 3)
    assert(baselines.shape[1] == 2)
    noise_map = np.empty(u.shape[0])
    u_flattened = u.flatten().copy()
    cpp_dm_noise_sim_inst_function(noise, u_flattened, u_flattened.shape[0]//3,
                               baselines, baseline_counts, baselines.shape[0],
                               wavelength,
                               dec, dish_diameter,
                               noise_map)
    return noise_map

try:
	gpu_dmns_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"gpu_dmns.so"))
	cuda_dm_noise_sim_function = gpu_dmns_lib.dm_noise_sim_caller
	cuda_dm_noise_sim_function.argtypes = \
    	[ctypes.c_float, ndpointer(dtype=ctypes.c_float), ctypes.c_int,
    	ndpointer(dtype=ctypes.c_float), ndpointer(dtype=ctypes.c_int), ctypes.c_int,
    	ndpointer(dtype=ctypes.c_float), ctypes.c_int,
    	ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, 
    	ndpointer(dtype=ctypes.c_float), ctypes.c_ulonglong]

	def dm_noise_simulator_wrapper_gpu (noise, u, baselines, baseline_counts, wavelengths, dec, dish_diameter, deg_distance_to_count, ntimesamples, seed=1234):
		assert(baselines.shape[0] == baseline_counts.shape[0])
		assert(u.shape[1] == 3)
		assert(baselines.shape[1] == 2)
		noise_map = np.empty(u.shape[0]*wavelengths.shape[0], dtype=np.float32)
		u_flattened = u.flatten().copy()
		cuda_dm_noise_sim_function(noise, u_flattened, u_flattened.shape[0]//3,
        	                       baselines, baseline_counts, baselines.shape[0],
        	                       wavelengths, wavelengths.shape[0],
        	                       dec, dish_diameter, deg_distance_to_count, ntimesamples,
        	                       noise_map, seed)
		return noise_map
except:
	pass


