import numpy as np
import os
import ctypes
import time
from numpy.ctypeslib import ndpointer
from wrapper_functions import *
import reverb as rev

omega = 2*np.pi/(3600*24) ## in per s

if __name__ == "__main__": ## this 'if' statement prevents this code from running if not
                            # executed as a script
    
    ### IMAGING DETAILS ### (fill this out)

    run_name = '0528_template' ## will appear in .fits name

    date = '250528' ## this will be the name of the folder the .fits file is put in

    print('Run:' + run_name) ## for slurm diagnostics
    imcenter = (55,88) ## RA, dec of the centre of the image. This also sets the WCS
    cellsize = 1/120 ## image pixel sidelength in deg
    imsize = 1000 ## pixels on a side
    imparams = np.array([imcenter[0],imcenter[1],cellsize,imsize])
    u,w = rev.gen_image_u(imcenter,cellsize,imsize) ## first argument is ra/dec in deg as tuple, then cellsize, then imsize. Angles in deg

    ### OBSERVATION SPECIFICS ### (set these; some are calculated for you from others)

    M = 22 ## how many antennae in NS direction
    N = 24 ## how many antennae in EW direction
    L1 = 8.5 ## antenna spacing delta in NS direction (in m)
    L2 = 6.3 ## antenna spacing delta in EW direction (in m)
    chord_lat = 49.3 ## chord zenith declination (deg)
    ant_diam = 6.0 ## antenna diameter (in m)
    dphi = 2 ## degrees of RA per time step
    dtau = dphi*np.pi/180/omega ## integration time (in s; calculated for you if you set dphi)
    centre_phi_RA_deg = 55 ## central RA, in deg, for the set of integrations
    N_times = 201 ## number of integrations to do
    initial_phi_offset = (N_times-1)/2*dphi ## Calculated for you. Don't worry about this

    survey_dec = 88 ## Set's CHORD
    nu1 = 1000e6 ## first channel frequency (Hz)
    nu2 = 1500e6  ## last channel frequency (Hz)
    nchannels = 5 ## number of channels
    dnu = (nu2-nu1)/nchannels ## (computed for you)
    eta = 1 ## antenna power collection efficiency
    SEFD = 6000 ## per antenna system equivalent flux density (in Jy)
    
    ### CHORD SETUP ### (this plugs in the information you entered above to Hans' classes)

    chord_thetas = np.asarray([np.deg2rad(90-survey_dec)], dtype=ctypes.c_float)
    cp = chordParams(thetas = unpackArraytoStruct(chord_thetas),
                    centre_phi = np.deg2rad(centre_phi_RA_deg),
                    initial_phi_offset = np.deg2rad(initial_phi_offset),
                     m1=M, m2=N, L1=L1, L2=L2, CHORD_zenith_dec = chord_lat, D = ant_diam,
                    delta_tau = dtau, time_samples=N_times)
                    
    ### SOURCE SETUP ### (you only need to change this if you want non-background sources)

    t0 = time.time()

    ## leave this block alone, it reads in the sky background file##
    background_data = np.load('SKYMODEL_RA_dec_F_sidx.npz')
    RA = background_data['RA']
    Dec = background_data['dec']
    FJy = background_data['FmJy']/1000
    spec_idx = background_data['spec_idx']
    source_us,F,s = rev.return_close_sources(centre_phi_RA_deg,survey_dec,initial_phi_offset,N_times,dphi,RA,Dec,FJy,spec_idx)
    ################################################################

    frequencies = np.linspace(nu1,nu2,nchannels)
    spectra = rev.get_spectra(frequencies,F,s) ## generates spectra for all background objects
    wavelengths = 3e8/frequencies

    ### THIS IS WHERE YOU WOULD APPEND YOUR OWN SOURCES ###
    
    # The inputs you'd need to change are: 
    
    # 'source_us': Python array of unit vectors on the celestial sphere pointing to
    # the sources you want included in the simulation (RA = 0 is along the x axis). 
    # Shape is [(Number of sources) x 3]. If you haven't changed anything above, this already
    # includes all of the background sources

    # 'spectra': This is an array containing the source brightnesses at all of the
    # specified frequencies (in Hans' code the units are arbitrary; here it is set
    # up to interpret the input as a flux density in Janskys). The shape of this array
    # is [(Number of sources) x (Number of frequencies/channels)]. If I have a frequency
    # array with elements (1300e6, 1400e6, 1500e6), then spectra[0][0] specifies the
    # flux density for the first source in source_us at 1300MHz, spectra[0][1] specifies
    # the flux density at 1400MHz, and so on. spectra[10][2] would be the flux density of
    # the 11th source in source_us at 1500MHz.

    # Here are two example codes to add sources on. First, a single source
    # with a constant spectral index:

    u_new1 = np.array([rev.u(55,89)])
    F_new_1p4 = np.array([1]) ## Flux density [Jy] snapshot at 1.4GHz
    s_new = np.array([-1]) ## spectral index. Flux density for this source will go like frequency^{s}
    new_spectra1 = rev.get_spectra(frequencies,F_new_1p4,s_new)

    # now connect these on to the existing arrays:

    source_us = np.concatenate((source_us,u_new1),axis = 0)
    spectra = np.concatenate((spectra,new_spectra1),axis = 0)

    t1 = time.time()

    print("generating sources took", t1-t0, " seconds")

    ### RUN THE CODE ###

    dirtymap = dirtymap_simulator_wrapper (u.astype(ctypes.c_float), wavelengths.astype(ctypes.c_float), source_us, spectra, 1e-9, cp)
    dirtymap = dirtymap.reshape(imsize,imsize,len(frequencies))

    dirtymap /= M**2
    dirtymap /= N**2 ## for normalization

    t2 = time.time()

    print("Dirtymap simulator took", t2-t1, " seconds")

    ### MAKE CORRESPONDING NOISE ###

    noise = dirtymap.copy()
    for i in range(len(frequencies)):
    	noise[:,:,i] = rev.make_some_noise(M,N,L1,L2,chord_lat,survey_dec,N_times,dnu,dtau,SEFD,eta,frequencies[i],imsize,cellsize,ant_diam,applybeam = False)
    
    t3 = time.time()

    print("Generating noise took", t3-t2, " seconds")

    ### RECOVER THE BEAM ###

    A_beam,B_beam = rev.recover_net_beam(u, centre_phi_RA_deg, initial_phi_offset, dphi, N_times, frequencies, survey_dec, antenna_diam = ant_diam)

    ### apply the normalized raw beam to my own noise:

    for i in range(len(frequencies)):

        noise[:,:,i] *= A_beam[:,:,i] ## maximum will be divided out later!

    ## normalize the beam and dirtymap ##:
    for i in range(len(frequencies)):
        maxx = np.max(A_beam[:,:,i])
        A_beam[:,:,i] /= maxx
        dirtymap[:,:,i] /= maxx
        noise[:,:,i] /= maxx
        (noise[:,:,i])[A_beam[:,:,i]<0.25] = np.nan
        (dirtymap[:,:,i])[A_beam[:,:,i]<0.25] = np.nan

    t4 = time.time()
    print("Generating the beam took", t4-t3, " seconds")
    
    if not os.path.exists(f"output/{date}"):
        os.makedirs(f"output/{date}")
    rev.writetofits(f'output/{date}/{run_name}',w,dirtymap,noise,A_beam)
    #np.savez(f"output/{run_name}/data.npz", dirtymap=dirtymap, frequencies=frequencies, imparams = imparams)
    #np.savez(f"output/{run_name}/noise.npz", noise = noise)
    #np.savez(f"output/{run_name}/beam.npz", beam = A_beam)
    #txtfile = open(f"{run_name}.txt",'w')
    #txtfile.write(description)
    #txtfile.close() 

