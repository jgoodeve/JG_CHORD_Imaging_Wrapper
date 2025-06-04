import numpy as np
import os
import ctypes
from numpy.ctypeslib import ndpointer

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
