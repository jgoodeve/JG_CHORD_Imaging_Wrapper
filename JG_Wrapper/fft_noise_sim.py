from astropy import wcs
import astropy.io.fits as fits
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.linalg import ishermitian

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

def Visibility_noise(antenna_positions,s0,M,N,dnu,dt,SEFD = 6000,eta = 1,freq = 1e9):

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

    imsize_rel = np.max(imsize) ## creating a square image, will crop later

    dL_rad = np.pi/180*(cell_size/60)
    dL = cell_size
    umax = int(1/2/dL_rad)
    dU = 2*umax/imsize_rel

    if imsize_rel % 2 != 0:
        uu = np.arange(-int(imsize_rel/2),int(imsize_rel/2)+1)*dU
    elif imsize_rel % 2 == 0:
        uu = np.arange(-np.round((imsize_rel)/2),np.round(imsize_rel/2))*dU

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

def FFT_noise_sim(M,N,L1,L2,lat,dec,N_times,dnu,dt,SEFD,eta,freq,imsize,cellsize_deg,antenna_diameter,applybeam = False):

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

    eta: electronic power efficiency. Float from 0-1

    freq: frequency to generate noise at

    imsize: dimensions of the image to be made in number of pixels as a 2-tuple (x,y)

    cellsize_deg: dimensions of an individual pixel, in degrees (make sure that a synthesized beam [i.e. resolution element] is at least a few pixels across!

    antenna_diameter: size of CHORD antennae, in m (6m for CHORD)

    '''

    antennae = np.zeros((N,M,3))
    
    for m in range(M):
        antennae[:,m,0] = m*L1
    
    for n in range(N):
        antennae[n,:,1] = n*L2

    antennae = antennae.reshape(M*N,3)
    s0 = np.array([np.cos(dec*np.pi/180),0,np.sin(dec*np.pi/180)])
    
    antennae = Antenna_positions_cartbasis(antennae,0,lat)
    pos,magni = Visibility_noise(antennae,s0,M,N,dnu=dnu,dt=dt,SEFD = SEFD,eta = eta,freq = freq)
    maggy = grid(pos,magni,imsize,cellsize_deg*60)
    noise = gen_noise_image(maggy)

    ### next bit crops the noise down to the size we wanted

    if noise.shape[0] > imsize[0]:
        noise = noise[:imsize[0]]
    if noise.shape[1] > imsize[1]:
        noise = noise[:,:imsize[1]]

    return noise.T*len(maggy.ravel())/M/N/(M*N-1)/np.sqrt(2)  # THESE EXTRA
                                                            # NORMALIZATION FACTORS ARE
                                                            # VERY IMPORTANT; THEY CORRECT
                                                            # THE NOISE NORMALIZATION TO BE
                                                            # IN INTENSITY UNITS (LIKE JY/BEAM)
                                                            # BY DIVIDING BY THE SUM OF WEIGHTS
                                                            # (NATURAL WEIGHTING ASSUMED, ALL WEIGHTS
                                                            # ARE 1) AND ADJUST FOR THE DEFAULT IFFT
                                                            # NORMALIZATION IN NUMPY.FFT. THE SQRT(2)
                                                            # ADJUSTS TO BE APPROPRIATE FOR ANY STOKES
                                                            # PARAMETER RATHER THAN A SINGLE POL)

def gen_noise_image(mags):

    noise = np.real(np.fft.ifft2(np.fft.ifftshift(mags,axes = (0,1)),axes = (0,1)))
    return noise