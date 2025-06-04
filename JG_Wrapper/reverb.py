from astropy import wcs
import astropy.io.fits as fits
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as units
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.linalg import ishermitian
from scipy.special import j1

# This file is full of convenience functions written to help utilize the CHORD dirtymap simulator written by Hans Hopkins
# for the purposes of square imaging, simulating searches for slow radio transients. The functions here are meant to only
# be called by the python wrapper, using the various user inputs appropriately.

# Written by Josh Goodeve

c = 3e8

def cross(a,b):

    '''
    Cross product of two cartesian (x,y,z) three vectors a,b
    '''

    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

def dot(a,b):

    '''
    dot product of two cartesian (x,y,z) three vectors a,b
    '''

    return np.sum(a*b)

def mag(a):

    '''
    magnitude of a cartesian three vector a
    '''

    return np.sqrt(dot(a,a))

def Antenna_positions_cartbasis(positions_ground_basis,long,lat):

    '''
    The user inputs the longitude, latitude of the observatory and the easting, northing of each antenna relative to the origin - this converts those to positions in the 'cartesian' basis where the z axis points towards the celestial north pole.
    '''

    lat *= np.pi/180
    long *= np.pi/180

    primed_to_unprimed = np.array([[-np.sin(long),np.cos(long),0],
                                  [-np.sin(lat)*np.cos(long),np.sin(lat)*np.sin(long),np.cos(lat)],
                                  [np.cos(lat)*np.cos(long),np.cos(lat)*np.sin(long),np.sin(lat)]])

    unprimed_to_primed = np.linalg.inv(primed_to_unprimed)
    positions = []

    for i in range(len(positions_ground_basis)):
            positions.append(unprimed_to_primed@positions_ground_basis[i])
    positions = np.array(positions)

    return positions

def Visibility_noise(antenna_positions,s0,M,N,N_times,dnu,dt,SEFD = 6000,eta = 1,freq = 1e9):

    l_hat = cross(s0,np.array([0,0,1]))
    l_hat = l_hat/mag(l_hat)
    m_hat = cross(l_hat,s0)
    m_hat = m_hat/mag(m_hat) ## build the basis

    sigma = SEFD/eta/np.sqrt(2*dnu*dt)

    visibility_positions = []
    visibility_values = []

    antenna_positions_obsbasis = []
    
    for position in antenna_positions:
        newpos = np.array([dot(position,l_hat),dot(position,m_hat),dot(position,s0)]) * freq/c ## divide by the wavelength in km
        antenna_positions_obsbasis.append(newpos)
    
    antenna_positions_obsbasis = np.array(antenna_positions_obsbasis)
    
    R = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            R[m,n] = (M-m)*(N-n)
    R = R.reshape(M*N)
    
    for i in range(len(antenna_positions_obsbasis)):
        vis_value1 = np.sqrt(R[i])*sigma*np.random.randn()+1j*np.sqrt(R[i])*sigma*np.random.randn()
        vis_value2 = np.sqrt(R[i])*sigma*np.random.randn()+1j*np.sqrt(R[i])*sigma*np.random.randn()
        H1 = antenna_positions_obsbasis[i]-antenna_positions_obsbasis[0]
        H2 = H1.copy()
        H2[1] = -H1[1]
        H1 = H1[:-1]
        H2 = H2[:-1]
        visibility_positions.append(H1)
        visibility_values.append(vis_value1)
        if np.abs(H2[0]) >= 0.001:
            if np.abs(H2[1]) >= 0.001:
                visibility_positions.append(H2)
                visibility_values.append(vis_value2)
        if i != 0:
            visibility_positions.append(-H1)
            visibility_values.append(np.conjugate(vis_value1))
            if np.abs(H2[0]) >= 0.001:
                if np.abs(H2[1]) >= 0.001:
                    visibility_positions.append(-H2)
                    visibility_values.append(np.conjugate(vis_value2))
    visibility_positions = np.array(visibility_positions)
    visibility_values = np.array(visibility_values)

    return visibility_positions,visibility_values

def grid(pos,val,imsize,cell_size):

    dL_rad = np.pi/180*(cell_size/60)
    dL = cell_size
    umax = int(1/2/dL_rad)
    dU = 2*umax/imsize

    if imsize % 2 != 0:
        uu = np.arange(-int(imsize/2),int(imsize/2)+1)*dU
    elif imsize % 2 == 0:
        uu = np.arange(-np.round((imsize)/2),np.round(imsize/2))*dU

    udelta = uu[1]-uu[0]

    bin_edges = np.linspace(uu[0]-0.5*udelta,uu[-1]+0.5*udelta,len(uu)+1)

    vis_real = np.real(val)
    vis_imag = np.imag(val)

    x = pos[:,0]
    y = pos[:,1]

    grid_real,xedges,yedges = np.histogram2d(x,y,bins = bin_edges,weights = vis_real)
    grid_imag,xedges,yedges = np.histogram2d(x,y,bins = bin_edges,weights = vis_imag)

    gridded_vis = grid_real+1j*grid_imag

    return gridded_vis

def gen_noise_image(mags):

    noise = np.real(np.fft.ifft2(np.fft.ifftshift(mags,axes = (0,1)),axes = (0,1)))
    return noise

def Antenna_Positions_ForwardinTime(antenna_positions,t):

    '''
    Compute the positions of the antennas at some later time t (in days)
    given their initial position at time 0. Basically all that changes is their relative orientation as the Earth rotates.
    '''

    xp = antenna_positions[:,0]
    yp = antenna_positions[:,1]
    zp = antenna_positions[:,2]

    x = np.cos(2*np.pi*t)*xp + np.sin(2*np.pi*t)*yp
    y = -np.sin(2*np.pi*t)*xp + np.cos(2*np.pi*t)*yp

    new_antenna_positions = np.array([[x[i],y[i],zp[i]] for i in range(len(xp))],dtype = np.float64)

    return new_antenna_positions

def make_some_noise(M,N,L1,L2,lat,dec,N_times,dnu,dt,SEFD,eta,freq,imsize,cellsize_deg,antenna_diameter,applybeam = False):

    '''
    Main noise function, meant to be called in the python wrapper. The array is assumed to be regular.

    PARAMETERS:

    M: Size of the array in N/S direction (integer). 

    N: Size of the array in E/W direction (integer).

    L1: inter-antenna distance, in meters, in N/S direction (float)

    L2: inter-antenna distance, in meters, in E/W direction (float)

    lat: CHORD zenith declination, in degrees. Assuming that CHORD antennae are at the same elevation, this is the same as the observatory site latitute. When writing this, at first, it didn't occur to me that these weren't necessarily the same :(

    dec: CHORD observing declination, in degrees.

    N_times: Number of integrations in the observation.

    dnu: Channel bandwidth, in Hz

    dt: integration length, in seconds

    SEFD: Individual antenna system equivalent flux density, in Jy (~6000Jy for CHORD)

    eta: antenna power efficiency. Float from 0-1

    freq: frequency to generate noise at

    imsize: dimensions of the image to be made [assumed square] in number of pixels

    cellsize_deg: dimensions of an individual pixel, in degrees (make sure that a synthesized beam [i.e. resolution element] is at least a few pixels across!

    antenna_diameter: size of CHORD antennae, in m (6m for CHORD)

    applybeam: apply a single factor of the primary beam to the noise [boolean] (default False)
    '''

    antennae = np.zeros((N,M,3))
    
    for m in range(M):
        antennae[:,m,0] = m*L1
    
    for n in range(N):
        antennae[n,:,1] = n*L2

    antennae = antennae.reshape(M*N,3)
    s0 = np.array([np.cos(dec*np.pi/180),0,np.sin(dec*np.pi/180)])
    
    antennae = Antenna_positions_cartbasis(antennae,0,lat)
    pos = []
    magni = []
    for i in range(N_times):
        Delta_t_days = (-(N_times-1)/2+i)*(dt/86400)
        antenna_positions_shifted = Antenna_Positions_ForwardinTime(antennae,Delta_t_days)
        pos_new,magni_new = Visibility_noise(antenna_positions_shifted,s0,M,N,N_times = N_times,dnu=dnu,dt=dt,SEFD = SEFD,eta = eta,freq = freq)
        for j in range(len(pos_new)):
            pos.append(pos_new[j])
        for k in range(len(magni_new)):
            magni.append(magni_new[k])
    pos = np.array(pos)
    magni = np.array(magni)
    maggy = grid(pos,magni,imsize,cellsize_deg*60)
    noise = gen_noise_image(maggy)

    return noise.T*len(maggy.ravel())/M/N/(M*N-1)/N_times

def sightread(npzfile,noisefile = None,beamfile = None):

    '''
    I used to use this to read .npz files before I was storing everything in .fits files.
    '''

    g = np.load(npzfile)
    h = np.load(noisefile)
    i = np.load(beamfile)
    dirtymap = g['dirtymap']
    frequencies = g['frequencies']
    imparams = g['imparams']
    beam = i['beam']
    if noisefile:
        noise = h['noise']
    ra_cent,dec_cent,cellsize,imsize = imparams
    imsize = int(np.round((imsize)))
    u,w = gen_image_u((ra_cent,dec_cent),cellsize,imsize)

    if beamfile:

        if noisefile:
    
            return u,w,frequencies,imparams,dirtymap,noise,beam
    
        else:
    
            return u,w,frequencies,imparams,dirtymap,beam

    else:

        if noisefile:
    
            return u,w,frequencies,imparams,dirtymap,noise
    
        else:
    
            return u,w,frequencies,imparams,dirtymap

def make_wcs(centre_ra_dec,cellsize,imsize):

    '''
    Create a wcs object corresponding to a given pixel scale, image size, and image centre.

    parameters:

    centre_ra_dec: 2-tuple containing the desired centre RA and dec, in degrees

    cellsize: size of an individual pixel, in degrees per pixel
    '''
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [(imsize+1)/2,(imsize+1)/2] ### centre pixel
    w.wcs.cdelt = np.array([-cellsize,cellsize]) ### pixel scale
    w.wcs.crval = centre_ra_dec ### centre RA and DEC in DEG
    w.wcs.ctype = ["RA---AZP", "DEC--AZP"]
    
    return w

def writetofits(name,wcs,dmap,noise=None,beam=None):

    '''
    Write the dirtymap, plus noise and beam if provided, to a .fits file containing versions of the data that have been combined with noise, and the beam, or not.

    parameters:

    name: string; this will be the name of the .fits file

    wcs: the world coordinate system object that will be embedded in the .fits file

    dmap: the output dirtymap 

    noise (default none): the noise map

    beam (default none): the primary (A) beam
    '''

    header = wcs.to_header()
    hdul = []
    dirtymap = np.transpose(dmap,(2,0,1))
    hdul.append(fits.PrimaryHDU(header=header))
    hdul.append(fits.ImageHDU(header=header,data=dirtymap,name = 'dirtymap'))
    if np.any(noise):
        noisemap = np.transpose(noise,(2,0,1))
        dmap_noise = np.transpose(dmap+noise,(2,0,1))
        hdul.append(fits.ImageHDU(header=header,data=dmap_noise,name = 'dirtymap+noise'))
        hdul.append(fits.ImageHDU(header=header,data=noisemap,name = 'noise'))
    if np.any(beam):
        beammap = np.transpose(beam,(2,0,1))
        dmap_pbcor = np.transpose(dmap/beam,(2,0,1))
        hdul.append(fits.ImageHDU(header=header,data=dmap_pbcor,name = 'dirtymap_pbcor'))
        hdul.append(fits.ImageHDU(header=header,data=beammap,name = 'beam'))
    if np.any(noise) & np.any(beam):
        dmap_noise_pbcor = np.transpose((dmap+noise)/beam,(2,0,1))
        hdul.append(fits.ImageHDU(header=header,data=dmap_noise_pbcor,name = 'dirtymap_noise_pbcor'))

    hdul = fits.HDUList(hdul)
    
    if name[-5:] == '.fits':
        hdul.writeto(name)
    else:
        hdul.writeto(name+'.fits')

def gen_image_u(centre_ra_dec,cellsize,imsize):

    '''
    given a central position in the sky and desired image dimensions, return a list of pixel [X,Y,Z] vectors on the unit sphere that can be input to Hans' code. 

    parameters:

    centre_ra_dec: 2-tuple of (ra,dec) in degrees - the desired image centre position.

    cellsize: desired image pixel scale in degrees/pixel.

    imsize: the [square] sidelength of the image to be made, in pixels.
    '''
    
    w = make_wcs(centre_ra_dec,cellsize,imsize)
    x = np.linspace(0,imsize-1,imsize)

    xx,yy = np.meshgrid(x,x)
    
    z_ = np.stack((xx,yy),axis = -1)
    z_ = z_.reshape((imsize**2,2))
    
    AX_0_pix = z_[:,0]
    AX_1_pix = z_[:,1]
    
    pos = w.pixel_to_world(AX_0_pix,AX_1_pix)
    
    X = pos.cartesian.x.value
    Y = pos.cartesian.y.value
    Z = pos.cartesian.z.value
    
    u = np.stack((X,Y,Z),axis = -1)
    
    return u.astype(np.float32),w

def beam(u, centre_RA, centre_Dec, freqs, antenna_diam = 6):

    ### first figure out the angle between each point and each pointing
    imsize = int(np.round(np.sqrt(len(u))))
    Xpix = u[:,0]
    Ypix = u[:,1]
    Zpix = u[:,2]
    X_poin = np.cos(centre_Dec)*np.cos(centre_RA)
    Y_poin = np.cos(centre_Dec)*np.sin(centre_RA)
    Z_poin = np.sin(centre_Dec)
    dp = Xpix*X_poin+Ypix*Y_poin+Zpix*Z_poin
    dp[dp>1] = 1. ## avoid floating point errors ever so slightly above 1
    angles = np.arccos(dp)
    beam = np.zeros((imsize,imsize,len(freqs)))
    for i in range(len(freqs)):
        airy_x = 2*np.pi*(antenna_diam/2)*angles*freqs[i]/c
        beam_block = 0*np.copy(airy_x)
        beam_block[airy_x > 0] = ((2*j1(airy_x[airy_x > 0])/airy_x[airy_x > 0])**2)
        beam_block[airy_x <= 0] = 1
        beam_block = beam_block.reshape(imsize,imsize)
        beam[:,:,i] = beam_block

    return beam.astype('float32')

def recover_net_beam(u, centre_phi, init_phi_off, dphi, N_times, freqs, survey_dec,antenna_diam = 6):

    ### first figure out the angle between each point and each pointing
    imsize = int(np.round(np.sqrt(len(u))))
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
    A_beam = np.zeros((imsize,imsize,len(freqs)))
    beam = np.zeros((imsize,imsize,len(freqs)))
    for i in range(len(freqs)):
        airy_x = 2*np.pi*(antenna_diam/2)*angles_block*freqs[i]/c
        beam_block = 0*np.copy(airy_x)
        beam_block[airy_x > 0] = ((2*j1(airy_x[airy_x > 0])/airy_x[airy_x > 0])**2)
        beam_block[airy_x <= 0] = 1
        beam_block = beam_block.reshape(imsize,imsize,N_times)
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