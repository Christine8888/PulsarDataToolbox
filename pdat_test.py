import sys
sys.path.append("/mnt/c/users/christine/dwg_tutorials")
import pdat
import numpy as np
import astropy.io.fits as fits
import fitsio as f
import os

def test_true():
    assert True

def test_false():
    assert False

def change_psrfits_shape(nchan=None, npol=None, nbin=None, nsubint=None, mode="astropy"):
    template = pdat.psrfits(None, template="PSR")
    template.set_subint_dims(nchan=nchan, npol=npol, nbin=nbin, nsubint=nsubint, nsblk=1)

    template.write_psrfits('temp.fits')
    template.close()

    if mode == "astropy":
        temp_shape = fits.open('temp.fits')['SUBINT'].data['DATA'].shape

    os.remove('temp.fits')
    return temp_shape

def test_change_psrfits_shape():
    nsubint = 1
    npol = 4
    nchan = 512
    nbin = 2048
    temp = change_psrfits_shape(nchan=nchan, npol=npol, nbin=nbin, nsubint=nsubint, mode="astropy")
    assert (nsubint, npol, nchan, nbin) == temp
