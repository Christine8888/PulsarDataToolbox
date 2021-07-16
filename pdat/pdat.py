# -*- coding: utf-8 -*-
# encoding=utf8
"""Main module."""

#Pulsar Data Toolbox. Based on fitsio package. See https://github.com/esheldon/fitsio for details.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pypulse as pp
import astropy.io.fits as fits
from astropy.io.fits import hdu
import collections, os, sys
import datetime
import warnings
import six

#package_path = os.path.dirname(__file__)
#template_dir = os.path.join(package_path, './templates/')

class psrfits(pp.Archive):

    def __init__(self, psrfits_path, mode='rw', from_template=False,
                 obs_mode=None, verbose=True):
        """
        Class which inherits fitsio.FITS() (Python wrapper for cfitsio) class's
        functionality, and add's new functionality to easily manipulate and make
        PSRFITS files.

        Parameters
        ----------

        from_template : bool, str
            Either a boolean which dictates if a copy would like to be made from
            a template, or a string which is the path to a user chosen template.

        psrfits_path : str
            Either the path to an existing PSRFITS file or the name for a new
            file.

        obs_mode : Same as OBS_MODE in a standard PSRFITS, either SEARCH, PSR or
            CAL for search mode, fold mode or calibration mode respectively.

        mode : str, {'r', 'rw, 'READONLY' or 'READWRITE'}
            Read/Write mode.

        """

        # define fields from constructor
        self.verbose = verbose
        self.psrfits_path = psrfits_path
        self.obs_mode = obs_mode

        dir_path = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(psrfits_path) and not from_template and verbose:
            print('Loading PSRFITS file from path:\n'
                  '    \'{0}\'.'.format(psrfits_path))

        #TODO If user enters 'READWRITE' (or 'rw') but from_template=False then
        # the template will track the changes in the loaded file and save them
        # as using the loaded .fits as the template...
        # or (from_template==False and mode='rw')
        elif from_template:
            if os.path.exists(psrfits_path):
                os.remove(psrfits_path)
                if verbose:
                    print('Removing older PSRFITS file from path:\n'
                          '   \'{0}\'.'.format(psrfits_path))

            if isinstance(from_template, six.string_types): # if from_template is a string
                template_path = from_template # then set the template path
            # elif isinstance(from_template, bool):
            #     template_path = filename #Path to template...
                #TODO: Make a template that this works for
                #dir_path + '/psrfits_template_' + obs_mode.lower() + '.fits'

            if mode in ['r','READONLY']:
                raise ValueError('Can not write new PSRFITS file if '
                                 'it is initialized in read-only mode!')

            self.written = False
            self.fits_template = pp.Archive(template_path, prepare=False, lowmem=True, baseline_removal=False, onlyheader=True) # [fitsio] creates FITS object

            if self.obs_mode is None:
                OBS = self.fits_template.header['OBS_MODE'].strip() # [fitsio] get OBS_MODE from template if not yet set
                self.obs_mode = OBS
            else:
                self.obs_mode = obs_mode


            # initializing more fields

            self.draft_hdrs = collections.OrderedDict()
            self.HDU_drafts = {}
            self.subint_dtype = None

            #Set the ImageHDU to be called primary.
            self.draft_hdrs['PRIMARY'] = self.fits_template.header # Astropy header
            self.n_hdrs = len(self.fits_template.keys) # [fitsio] one header for each HDU

            # ISSUE: Archive doesn't allow access by keys
            other_tables = 0

            for ii in np.arange(1,self.n_hdrs): # for each HDU/header
                hdr_key = self.fits_template.keys[ii] # [fitsio] get name
                # print(hdr_key)
                if hdr_key == "HISTORY":
                    self.draft_hdrs[hdr_key] = self.fits_template.history.header
                elif hdr_key == "PSRPARAM":
                    self.draft_hdrs[hdr_key] = self.fits_template.paramheader # dictionary
                elif hdr_key == "POLYCO":
                    self.draft_hdrs[hdr_key] = self.fits_template.polyco.header # dictionary
                elif hdr_key == "SUBINT":
                    self.draft_hdrs[hdr_key] = self.fits_template.subintheader # dictionary
                else:
                    self.draft_hdrs[hdr_key] = self.fits_template.tables[other_tables].header
                    other_tables += 1

                self.HDU_drafts[hdr_key] = None # no HDU yet
            self.draft_hdr_keys = list(self.draft_hdrs.keys()) # get all header keys used

            if verbose:
                msg = 'Making new {0} mode PSRFITS file '.format(self.obs_mode)
                msg += 'using template from path:\n'
                msg += '    \'{0}\'. \n'.format(template_path)
                msg += 'Writing to path: \n    \'{0}\''.format(psrfits_path)
                print(msg)


        # second issue: when you remove psrfits_path then this doesn't work
        super().__init__(psrfits_path, prepare=False, lowmem=True, baseline_removal=False, center_pulse=False, weight=False, wcfreq=False) # [fitsio] initialize a parent class object (FITS)?

        #If self.obs_mode is still None use loaded PSRFITS file
        if self.obs_mode is None and from_template: # i think this happens twice? maybe?
            OBS = self.fits_template.header['OBS_MODE'].strip() # [fitsio] get obs_mode from template
            self.obs_mode = OBS

        if from_template and verbose:
            print('The Binary Table HDU headers will be written as '
                  'they are added\n     to the PSRFITS file.')

        elif not from_template and (mode=='rw' or mode=='READWRITE'): # if you want to write stuff
            self.draft_hdrs = collections.OrderedDict()
            self.HDU_drafts = {}
            #Set the ImageHDU to be called primary.
            self.written = False

            self.draft_hdrs['PRIMARY'] = self.fits_template.header # Astropy header
            self.n_hdrs = len(self.fits_template.keys) # [fitsio] one header for each HDU

            for ii in range(self.n_hdrs-1): # for each HDU/header
                hdr_key = self.fits_template.keys[ii+1] # [fitsio] get name
                # print(hdr_key)
                if hdr_key == "HISTORY":
                    self.draft_hdrs[hdr_key] = self.fits_template.history.header
                elif hdr_key == "PSRPARAM":
                    self.draft_hdrs[hdr_key] = self.fits_template.paramheader # dictionary
                elif hdr_key == "POLYCO":
                    self.draft_hdrs[hdr_key] = self.fits_template.polyco.header # dictionary
                elif hdr_key == "SUBINT":
                    self.draft_hdrs[hdr_key] = self.fits_template.subintheader # dictionary
                else:
                    self.draft_hdrs[hdr_key] = self.fits_template.tables[other_tables].header
                    other_tables += 1

                self.HDU_drafts[hdr_key] = None # no HDU yet
            self.draft_hdr_keys = list(self.draft_hdrs.keys()) # get all header keys used

def list_arg(list_name, string):
    """Returns the index of a particular string in a list of strings."""
    return [x for x, y in enumerate(list_name) if y == string][0]

def convert2asciii(dictionary):
    """
    Changes all keys (i.e. assumes they are strings) to ASCII and
    values that are strings to ASCII. Specific to dictionaries.
    """
    return dict([(key.encode('ascii','ignore'),value.encode('ascii','ignore'))
                 if type(value) in [str,bytes] else
                 (key.encode('ascii','ignore'),value)
                 for key, value in dictionary.items()])
