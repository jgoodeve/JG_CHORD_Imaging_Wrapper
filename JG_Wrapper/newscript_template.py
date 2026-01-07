import numpy as np
import os
import ctypes
import time
import pickle
from scipy import interpolate as interp
from scipy.integrate import quad
from numpy.ctypeslib import ndpointer
import JG_Wrapper as rev
import matplotlib.pyplot as plt

omega = 2*np.pi/(3600*24) ## in per s

start_time = time.time()

### IMAGING DETAILS ### (fill this out)
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

run_name = 'Test1' ## will appear in .fits name

foldername = 'Script_test_noise' ## this will be the name of the folder the .fits file is put in

print('Run:' + run_name) ## for slurm diagnostics

imcenter = (0,49.1) ## RA, dec of the centre of the image. This also sets the WCS
cellsize = 1/85 ## image pixel sidelength in deg
imsize = (900,300) ## pixels on a side
u,w = rev.gen_image_u(imcenter,cellsize,imsize) ## first argument is ra/dec in deg as tuple, then cellsize, then imsize. Angles in deg

### OBSERVATION SPECIFICS ### (set these; some are calculated for you from others)

M = 22 ## how many antennae in NS direction
N = 24 ## how many antennae in EW direction
L1 = 8.5 ## antenna spacing delta in NS direction (in m)
L2 = 6.3 ## antenna spacing delta in EW direction (in m)
chord_lat = 49.3 ## chord zenith declination (deg)
ant_diam = 6.0 ## antenna diameter (in m)
dtau = 120 ## integration time (in s)
dphi = 180/np.pi*omega*dtau ## degrees of RA per time step
centre_phi_RA_deg = 0 ## central RA, in deg, for the set of integrations
N_times = 31 ## number of integrations to do
initial_phi_offset = (N_times-1)/2*dphi ## Calculated for you. Don't worry about this

survey_dec = 49 ## Sets CHORD survey declination (degrees)
nu1 = 1300e6 ## first channel frequency (Hz)
nu2 = 1500e6  ## last channel frequency (Hz)
nchannels = 41 ## number of channels
dnu = (nu2-nu1)/nchannels ## (computed for you)
eta = 1 ## antenna power collection efficiency
SEFD = 6000 ## per antenna system equivalent flux density (in Jy)

frequencies = np.linspace(nu1,nu2,nchannels)
wavelengths = 3e8/frequencies

### WHAT TIMES (in whatever units your input times are spaced by) TO EVALUATE THE MAP AT? ###

Times = np.array([303]) ## must be called 'Times'

### WHAT SOURCE FIELDS TO INCLUDE?#############################################################
###############################################################################################

include_transients = False ## include the transient library. ##############

apply_transient_scintillation = True
trans_scin_range = (0.01,0.2)

include_AGN_background = True ## include the FIRST-derived, 'bright' background (>1mJy) ###########

apply_AGN_scintillation = True
AGN_scin_range = (0.01,0.2)

apply_AGN_variation = False ## apply realistic brightness variation statistics
### SOURCE VARIABILITY AMPLITUDES AND PROBABILITIES (These will not be used if backgrounds are not set to have variation) ###
varbins = np.array([0,0.02,0.1]) ## fractional brightness fluctuation standard deviation. 0.1 Corresponds to a 10% RMS flux density fluctuation
varprobs = np.array([318/370,40/370,12/370]) ## correspond probabilities. Must sum to 1

include_SFG_background = False ## include the randomly generated 'faint' background (10uJy - 1mJy) ###########

sim_noise_beam = True ## simulate the noise and beam?

### CHORD SETUP ### (this plugs in the information you entered above to Hans' classes)

chord_thetas = np.asarray([np.deg2rad(90-survey_dec)], dtype=ctypes.c_float)
cp = rev.chordParams(thetas = rev.unpackArraytoStruct(chord_thetas),
                    centre_phi = np.deg2rad(centre_phi_RA_deg),
                    initial_phi_offset = np.deg2rad(initial_phi_offset),
                     m1=M, m2=N, L1=L1, L2=L2, CHORD_zenith_dec = chord_lat, D = ant_diam,
                    delta_tau = dtau, time_samples=N_times)

### LOAD IN THE VARIABILITY: ###

scinlib = np.load('JG_Wrapper/Backgroundfiles/scintillation_library.npz')['varlib']
varlib = np.load('JG_Wrapper/Backgroundfiles/variation_library.npz')['varlib']

### NOW, THE AUTOMATED STUFF. YOU SHOULDN'T NEED TO TOUCH THINGS BELOW HERE ##########################################################
###########################################################################################################################################
###########################################################################################################################################

source_us_total = np.zeros((1,3))
source_us_total[0][0] = 1 ## so that this is actually a unit vector
spectra_total = np.zeros((608,1,len(frequencies)))

if include_transients:

    transient_data = np.load('JG_Wrapper/Backgroundfiles/JettedTDE_new.npz')
    transient_spectra = transient_data['data']/1000 ## divide by 1000 to get Jy from mJy
    transient_ra = transient_data['ra']
    transient_dec = transient_data['dec']

    new_source_us = rev.u_vec(transient_ra,transient_dec)
    condition = np.abs(transient_dec - imcenter[1]) < 5
    source_us_trans = new_source_us[condition]
    transient_spectra = transient_spectra[condition]
    transient_spectra = np.transpose(transient_spectra,(1,0,2)) ## whoops axis order was off
        
    if apply_transient_scintillation:
        
        trans_scin_indices, varfractions, trans_scin_starts = rev.assign_variability(source_us_trans,varbins,varprobs,scinlib)
        scinvarfracs = 10**((np.log10(trans_scin_range[1])-np.log10(trans_scin_range[0]))*np.random.rand(len(source_us_trans))+np.log10(trans_scin_range[0])) ## from 0.01 to 0.2 logarithmically
        transient_spectra = rev.apply_variability(transient_spectra,scinlib,trans_scin_indices,trans_scin_starts,scinvarfracs)
    
    spectra_total = np.concatenate((spectra_total,transient_spectra),axis = 1)
    source_us_total = np.concatenate((source_us_total,source_us_trans),axis = 0)

if include_AGN_background:

    ## leave this block alone, it reads in the sky background file ##
    background_data = np.load('JG_Wrapper/SKYMODEL_RA_dec_F_sidx.npz')
    RA = background_data['RA']
    Dec = background_data['dec']
    FJy = background_data['FmJy']/1000
    spec_idx = background_data['spec_idx']
    AGN_source_us,F,s = rev.return_close_sources(centre_phi_RA_deg,survey_dec,initial_phi_offset,N_times,dphi,RA,Dec,FJy,spec_idx)
    AGN_flat_spectra = rev.get_spectra(frequencies,F,s)
    AGN_spectra = np.zeros((608,AGN_flat_spectra.shape[0],AGN_flat_spectra.shape[1]))
    AGN_spectra[:] = AGN_flat_spectra
    
    if apply_AGN_variation:
        
        agn_var_inds, agn_var_fracs, agn_var_starts = rev.assign_variability(AGN_source_us,varbins,varprobs,varlib)
        AGN_spectra = rev.apply_variability(AGN_spectra,varlib,agn_var_inds,agn_var_starts,agn_var_fracs)
        
    if apply_AGN_scintillation:
        
        agn_scin_indices, varfractions, agn_scin_starts = rev.assign_variability(AGN_source_us,varbins,varprobs,scinlib)
        agnscinvarfracs = 10**((np.log10(AGN_scin_range[1])-np.log10(AGN_scin_range[0]))*np.random.rand(len(AGN_source_us))+np.log10(AGN_scin_range[0]))
        AGN_spectra = rev.apply_variability(AGN_spectra,scinlib,agn_scin_indices,agn_scin_starts,agnscinvarfracs)
        
    spectra_total = np.concatenate((spectra_total,AGN_spectra),axis = 1)
    source_us_total = np.concatenate((source_us_total,AGN_source_us),axis = 0)
    
    spectra_total = spectra_total[:,::10,:]
    source_us_total = source_us_total[::10]

if include_SFG_background:

    ### generate the faint background sources:
    phi1 = centre_phi_RA_deg - 0.5*N_times*dphi - 2/np.cos(np.deg2rad(imcenter[1]))
    phi2 = centre_phi_RA_deg + 0.5*N_times*dphi + 2/np.cos(np.deg2rad(imcenter[1]))

    if np.abs(phi2-phi1) > 360:
        phi2 = 360
        phi1 = 0
    faint_u, faint_spectra = rev.gen_faint_background_Matthews(0.1,1,
    phi1,
    phi2,
    imcenter[1]-2,
    np.min((imcenter[1]+2,90)),frequencies)
    SFG_spectra = np.zeros((608,faint_spectra.shape[0],faint_spectra.shape[1]))
    SFG_spectra[:] = faint_spectra
    
    spectra_total = np.concatenate((spectra_total,SFG_spectra),axis = 1)
    source_us_total = np.concatenate((source_us_total,faint_u),axis = 0)
    
spectra_total = spectra_total[:,1:,:].astype('float32') ## remove the unnecessary first guy
source_us_total = source_us_total[1:,:].astype('float32')

MAP = np.zeros((imsize[1],imsize[0],len(Times),len(frequencies))).astype('float32')
if sim_noise_beam:
    NOISE = MAP.copy().astype('float32')

for j in range(len(Times)):

    t1 = time.time()

    T = Times[j]
    
    spectra = spectra_total[T]

    ### RUN THE CODE ###

    dirtymap = rev.dirtymap_simulator_wrapper (u.astype(ctypes.c_float), wavelengths.astype(ctypes.c_float), source_us_total, spectra, 1e-9, cp)
    dirtymap = dirtymap.reshape(imsize[1],imsize[0],len(frequencies))

    t2 = time.time()
    
    print('loop: %d/%d' %(j,len(Times)))
    print('dirtymap: %.3fs' %(t2-t1))

    ### MAKE CORRESPONDING NOISE ###

    noise = 0*dirtymap.copy()
    
    if sim_noise_beam:
        
        for i in range(len(frequencies)):
            noise[:,:,i] = rev.FFT_noise_sim(M,N,L1,L2,chord_lat,survey_dec,N_times,dnu,dtau,SEFD,eta,frequencies[i],imsize,cellsize,ant_diam,applybeam = False)
    
    t3 = time.time()
    print('noise: %.3fs' %(t3-t2))
    ### RECOVER THE BEAM ###

    if j == 0: ## because this only has to be done once

        A_beam,B_beam = rev.recover_net_beam(u, centre_phi_RA_deg, initial_phi_offset, dphi, N_times, frequencies, survey_dec, imsize, antenna_diam = ant_diam)

    dirtymap,noise,beam = rev.normalize(dirtymap,noise,A_beam,B_beam,frequencies,M,N,beamthresh = 0.25)
    t4 = time.time()

    MAP[:,:,j,:] = dirtymap
    if sim_noise_beam:
        NOISE[:,:,j,:] = noise
        
MAP = np.transpose(MAP,(3,2,0,1))
NOISE = np.transpose(NOISE,(3,2,0,1))
beam = np.transpose(beam,(2,0,1))

end_time = time.time()

seconds = end_time - start_time

hours = 0
minutes = 0

while seconds >= 60:

    minutes += 1
    seconds -= 60

while minutes >= 60:

    hours += 1
    minutes -= 60

print('Total time: %dh%dm%ds' %(hours,minutes,seconds))

if not os.path.exists(f"output/{foldername}"):
        os.makedirs(f"output/{foldername}")

if sim_noise_beam:
    rev.writetofits(f"output/{foldername}/{run_name}",[('dirtymap',MAP,'Jy/bm'),('radiometer noise',NOISE,'Jy/bm'),('Effective Beam', beam, 'Fractional Power Efficiency')],w,overwrite = True)
else:
    rev.writetofits(f"output/{foldername}/{run_name}",[('dirtymap',MAP,'Jy/bm')],w,overwrite = True)