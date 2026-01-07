from astropy import wcs
import astropy.io.fits as fits
import numpy as np
from astropy.coordinates import SkyCoord

def make_wcs(centre_ra_dec,cellsize,imsize):

    '''
    Create a wcs object corresponding to a given pixel scale, image size, and image centre.

    parameters:

    centre_ra_dec: 2-tuple containing the desired centre RA and dec, in degrees

    cellsize: size of an individual pixel, in degrees per pixel

    imsize: 2-tuple of image dimensions, first the 0th axis, then the first axis, in pixels
    '''
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [(imsize[0]+1)/2,(imsize[1]+1)/2] ### centre pixel
    w.wcs.cdelt = np.array([-cellsize,cellsize]) ### pixel scale. The first
                                                # entry is negative because
                                                # RA decreases going left -> right
                                                # in an image where north is up
    w.wcs.crval = centre_ra_dec ### centre RA and DEC in DEG
    w.wcs.ctype = ["RA---AZP", "DEC--AZP"]
    
    return w

def writetofits(filename,data,wcs,overwrite = False):

    '''
    Write sets of data, with names and units, to .fits files
    for storage, viewing, and easy association with physical sky
    positions. Make sure to provide data with the axis order (RA, Dec, Freq)!

    Example:

    if filename = 'NAME',

    data = [(name1, data1, unit1), (name2, data2, unit2), ...],

    wcs = [a correct wcs object corresponding to the image field],

    and data1, data2, are numpy arrays with axis order (Dec, RA) or (Freq, Dec, RA),

    then this code should work!

    By default, and here, fits files are read with axes BACKWARDS, so if you load
    the resulting fits file into ds9, CARTA or something similar, the axes will now
    be (RA, Dec, Freq) or (RA, Dec). This axis order must be followed
    or the WCS will not correspond correctly.

    '''

    header = wcs.to_header()
    hdul = []
    hdul.append(fits.PrimaryHDU(header=header))

    for datum in data:

        HDUname = datum[0]
        HDUdata = datum[1]
        HDUunit = datum[2]
        newheader = wcs.to_header()
        newheader['BUNIT'] = HDUunit
        datashape = HDUdata.shape
        hdul.append(fits.ImageHDU(header = newheader,data = HDUdata,name = HDUname))

    hdul = fits.HDUList(hdul)
    
    if filename[-5:] == '.fits':
        hdul.writeto(filename,overwrite=overwrite)
    else:
        hdul.writeto(filename+'.fits',overwrite=overwrite)

def gen_image_u(centre_ra_dec,cellsize,imsize):

    '''
    given a central position in the sky and desired image dimensions, 
    return a list of pixel [X,Y,Z] vectors on the unit sphere that can 
    be input to Hans' code. 

    parameters:

    centre_ra_dec: 2-tuple of (ra,dec) in degrees 
    - the desired image centre position.

    cellsize: desired image pixel scale in degrees/pixel.

    imsize: 2-tupe of image sidelengths, in pixels
    '''
    
    w = make_wcs(centre_ra_dec,cellsize,imsize)
    x = np.linspace(0,imsize[0]-1,imsize[0])
    y = np.linspace(0,imsize[1]-1,imsize[1])

    xx,yy = np.meshgrid(x,y)
    
    z_ = np.stack((xx,yy),axis = -1)
    z_ = z_.reshape((imsize[0]*imsize[1],2))
    
    AX_0_pix = z_[:,0]
    AX_1_pix = z_[:,1]
    
    pos = w.pixel_to_world(AX_0_pix,AX_1_pix)
    
    X = pos.cartesian.x.value
    Y = pos.cartesian.y.value
    Z = pos.cartesian.z.value
    
    u = np.stack((X,Y,Z),axis = -1)
    
    return u.astype(np.float32),w