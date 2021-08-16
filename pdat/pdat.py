# -*- coding: utf-8 -*-
# encoding=utf8
"""Main module."""

#Pulsar Data Toolbox. Based on fitsio package. See https://github.com/esheldon/fitsio for details.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import math
import pypulse as pp
import astropy.io.fits as pyfits
from astropy.io.fits import hdu
import collections, os, sys
import datetime
import warnings
import six
import gc as g
import time

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range



#package_path = os.path.dirname(__file__)
#template_dir = os.path.join(package_path, './templates/')

class psrfits(pp.Archive):


    def __init__(self, file_path=None, template=False, #  mode='rw',
                 obs_mode=None, verbose=True):
        """
        Class which inherits fitsio.FITS() (Python wrapper for cfitsio) class's
        functionality, and add's new functionality to easily manipulate and make
        PSRFITS files.

        Parameters
        ----------

        template : bool, str
            Either a boolean which dictates if a copy would like to be made from
            a template, or a string which is a template ("PSR", "SEARCH").

        file_path : str
            Either the path to an existing PSRFITS file or the name for a new
            file.

        obs_mode : Same as OBS_MODE in a standard PSRFITS, either SEARCH, PSR or
            CAL for search mode, fold mode or calibration mode respectively.

        """

        # define fields from constructor
        self.verbose = verbose
        self.file_path = file_path
        self.obs_mode = obs_mode


        if file_path is not None and not template:
            if os.path.exists(file_path):
                if verbose:
                    print('Loading PSRFITS file from path:\n'
                          '    \'{0}\'.'.format(file_path))
                super().__init__(file_path, prepare=False, lowmem=True, verbose = verbose, baseline_removal=False, center_pulse=False, weight=False, wcfreq=False)

        elif file_path is not None and template:
            if os.path.exists(file_path):
                if verbose:
                    print('Loading template PSRFITS file from path:\n'
                          '    \'{0}\'.'.format(file_path))

                super().__init__(file_path, onlyheader=True, verbose = verbose, baseline_removal=False, center_pulse=False, weight=False, wcfreq=False)

        elif file_path is None and template == "SEARCH":
            file_path = "/mnt/c/users/christine/dwg_tutorials/PulsarDataToolbox/pdat/templates/search_template.fits"
            if verbose:
                print('Loading template PSRFITS file from path:\n'
                      '    \'{0}\'.'.format(file_path))

            super().__init__(file_path, onlyheader=True, verbose = verbose, baseline_removal=False, center_pulse=False, weight=False, wcfreq=False)

        elif file_path is None and template == "PSR":
            file_path = "pdat/templates/search_template.fits" # replace with appropriate template
            if verbose:
                print('Loading template PSRFITS file from path:\n'
                      '    \'{0}\'.'.format(file_path))

            super().__init__(file_path, onlyheader=True, verbose = verbose, baseline_removal=False, center_pulse=False, weight=False, wcfreq=False)
        else:
            raise ValueError('Must provide filepath or choose a template')

        if self.obs_mode is None:
            OBS = self.header['OBS_MODE'].strip() # [fitsio] get OBS_MODE from template if not yet set
            self.obs_mode = OBS

        self.subint_dtype = None
        self.n_hdrs = len(self.keys)
        self.written = False

        if self.obs_mode == "PSR" or self.obs_mode == "CAL":
            # print('here')
            self.nsubint = self.history.getLatest("NSUB")
            self.npol = self.history.getLatest("NPOL")
            self.nchan = self.history.getLatest("NCHAN")
            self.nbin = self.history.getLatest("NBIN")
            self.nsblk = 1
        elif self.obs_mode == "SEARCH":
            self.nsubint = self.subintheader["NAXIS2"]
            self.npol = self.subintheader["NPOL"]
            self.nchan = self.subintheader["NCHAN"]
            self.nbin = 1
            self.nsblk = self.subintheader["NSBLK"]

    def __getitem__(self, arg, header=True):
        if arg == "PRIMARY":
            return self.header

        elif arg == "HISTORY":
            if header:
                return self.history.header
            return self.history

        elif arg == "POLYCO":
            if header:
                return self.polyco.header
            return self.polyco

        elif arg == "PSRPARAM":
            if header:
                return self.paramheader
            return self.params

        elif arg == "SUBINT":
            if header:
                return self.subintheader
            return self.subintinfo # could also return data

        else:
            msg = f'{arg} is not currently supported'
            raise ValueError(msg)

    def initialize_data(self, obs_mode = 'PSR', data=None):
        self.written = True

        self.freq = np.zeros((self.nsubint, self.nchan))
        self.weights = np.ones((self.nsubint, self.nchan))

        if (obs_mode == 'PSR' or obs_mode == 'CAL'):
            if data is not None:
                self._data = data
            else:
                self._data = np.zeros((self.nsubint, self.npol, self.nchan, self.nbin))
            self.weighted_data = self._data * self.weights[:, None, :, None]/np.nansum(self.weights)

        elif obs_mode == 'SEARCH':
            if data is not None:
                self._data = data
            else:
                self._data = np.zeros((self.nsubint, self.nsblk, self.npol, self.nchan, self.nbin))
            self.weighted_data = self._data * self.weights[:, None, None, :, None]/np.nansum(self.weights)

    def set_draft_header(self, ext_name, hdr_dict):
        """
        Set draft header entries for the new PSRFITS file from a dictionary.

        Parameters
        ----------
        psrfits_object : pdat.psrfits
            Pulsar Data Toolbox PSRFITS object.

        ext_name : str, {'PRIMARY','SUBINT','HISTORY','PSRPARAM','POLYCO'}
            Name of the header to replace the header entries.

        hdr_dict : dict
            Dictionary of header changes to be made to the template header.
            Template header entries are kept, unless replaced by this function.
        """
        for key in hdr_dict.keys():
            self.replace_FITS_Record(ext_name,key,hdr_dict[key])

    def replace_from_file(self,path,table='all'):
        """
        Method to replace single tables from other PSRFITS files.
        Note: Tables are appended directly to the original file. Make a copy
            before copying if you are unsure about appending. The array must
            match the columns (in the numpy.recarray sense) of the existing
            PSRFITS file.

        Parameters
        ----------

        path : str
            Path to the new PSRFITS file to be appended.

        table : list
            List of BinTable HDU headers to append from file. Defaults to
                appending all secondary BinTables.
                ['HISTORY','PSRPARAM','POLYCO','SUBINT']
        """
        PF2A = pyfits.open(path)

        if table=='all':
            PF2A_hdrs = []
            for ii in range(self.n_hdrs):
                hdr_key = PF2A[ii].name
                PF2A_hdrs.append(hdr_key)

            table=PF2A_hdrs

        if self.verbose:
            print("Appending from: {}".format(table))


        if "HISTORY" in table:
            for key in PF2A["HISTORY"].data.dtype.names:
                #if key in self.history.dictionary.keys():
                temp_data = PF2A["HISTORY"].data[key][0]
                temp_tuple = list(self.history.dictionary[key])
                temp_tuple[-1] = temp_data # REPLACING, NOT APPENDING
                self.history.dictionary[key] = temp_tuple


        if "POLYCO" in table:
            for key in PF2A["POLYCO"].data.dtype.names:
                #if key in self.polyco.dictionary.keys():
                temp_data = PF2A["POLYCO"].data[key][0]
                temp_tuple = list(self.polyco.dictionary[key])
                temp_tuple[-1] = temp_data # REPLACING, NOT APPENDING
                self.polyco.dictionary[key] = temp_tuple

        if "PSRPARAM" in table:
            size = int(PF2A['PSRPARAM'].data.shape[0])
            keys = [PF2A['PSRPARAM'].data[i][0].split(' ')[0] for i in range(size)]
            # print(keys)
            for key in keys:
                temp_data = PF2A['PSRPARAM'].data[list_arg(keys, key)][0].split(' ')[-1]
                if key in self.params.paramnames:
                    index = list_arg(self.params.paramnames, key)
                    self.params.paramlist[index].value = temp_data

        if "SUBINT" in table:
            for key in PF2A['SUBINT'].data.dtype.names[:-5]: # NOT currently replacing any data. if no need to replace data at all, could just use Archive object
                temp_data = PF2A['SUBINT'].data[key]
                #if key in test.subintinfo.keys():
                temp_tuple = list(self.subintinfo[key])
                temp_tuple[-1] = temp_data # REPLACING, NOT APPENDING
                self.subintinfo[key] = temp_tuple
                self.replace_subint_info(self.nsubint) # needs to adjust to be the right subint size if this is too small


    def replace_subint_info(self, nsubint):
        if self.obs_mode == "PSR" or self.obs_mode == "CAL":
            self.replace_FITS_tuple('SUBINT', 'INDEXVAL', self.subintinfo['INDEXVAL'][-1][:nsubint])
            self.replace_FITS_tuple('SUBINT', 'AUX_DM', self.subintinfo['AUX_DM'][-1][:nsubint])
            self.replace_FITS_tuple('SUBINT', 'AUX_RM', self.subintinfo['AUX_RM'][-1][:nsubint])

        self.replace_FITS_tuple('SUBINT', 'TSUBINT', self.subintinfo['TSUBINT'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'OFFS_SUB', self.subintinfo['OFFS_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'LST_SUB', self.subintinfo['LST_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'RA_SUB', self.subintinfo['RA_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'DEC_SUB', self.subintinfo['DEC_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'GLON_SUB', self.subintinfo['GLON_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'GLAT_SUB', self.subintinfo['GLAT_SUB'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'FD_ANG', self.subintinfo['FD_ANG'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'POS_ANG', self.subintinfo['POS_ANG'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'PAR_ANG', self.subintinfo['PAR_ANG'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'TEL_AZ', self.subintinfo['TEL_AZ'][-1][:nsubint])
        self.replace_FITS_tuple('SUBINT', 'TEL_ZEN', self.subintinfo['TEL_ZEN'][-1][:nsubint])


    def replace_FITS_tuple(self, hdr, name, new_value):
        if hdr == "HISTORY":
            temp = list(self.history.dictionary[name])
            temp[-1] = [new_value]
            self.history.dictionary[name] = temp
        elif hdr == "POLYCO":
            temp = list(self.polyco.dictionary[name])
            temp[-1] = [new_value]
            self.polyco.dictionary[name] = temp

        elif hdr == "SUBINT":
            temp = list(self.subintinfo[name])
            temp[-1] = new_value
            self.subintinfo[name] = temp


    def replace_FITS_Record(self, hdr, name, new_value):
        """
        Replace a Fits record with a new value in a fitsio.fitslib.FITSHDR
        object.

        Parameters
        ----------

        hdr : str
            Header name.

        name : FITS Record/Car
            FITS Record/Card name to replace.

        new_value : float, str
            The new value of the parameter.
        """
        # special_fields = ['TDIM17','TDIM20']

        if hdr == "PRIMARY":
            self.header[name] = new_value
        elif hdr == "HISTORY":
            self.history.header[name] = new_value
        elif hdr == "POLYCO":
            self.polyco.header[name] = new_value
        elif hdr == "PSRPARAM":
            self.paramheader[name] = new_value
        elif hdr == "SUBINT":
            self.subintheader[name] = new_value
        else:
            msg = f'{hdr} is not currently supported'
            raise ValueError(msg)

    def write_psrfits(self, save_path):

        if self.written:
            self.save(save_path)
        else:
            raise ValueError('Cannot save a file without data.')

    def save(self, filename):
        """Save the file to a new FITS file"""

        primaryhdu = pyfits.PrimaryHDU(header=self.header) #need to make alterations to header
        hdulist = pyfits.HDUList(primaryhdu)

        if self.history is not None:
            cols = []
            for name in self.history.namelist:
                fmt, unit, array = self.history.dictionary[name]
                #print name, fmt, unit, array
                col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
                cols.append(col)
            historyhdr = pyfits.Header()
            for key in self.history.headerlist:
                historyhdr[key] = self.history.header[key]
            historyhdu = pyfits.BinTableHDU.from_columns(cols, name='HISTORY', header=historyhdr)
            hdulist.append(historyhdu)
            # Need to add in PyPulse changes into a HISTORY
        #else: #else start a HISTORY table

        if self.params is not None:
            #PARAM and not PSRPARAM?:
            cols = [pyfits.Column(name='PSRPARAM', format='128A', array=self.params.filename)]
            paramhdr = pyfits.Header()
            for key in self.paramheaderlist:
                paramhdr[key] = self.paramheader[key]
            paramhdu = pyfits.BinTableHDU.from_columns(cols, name='PSRPARAM')
            hdulist.append(paramhdu)
            # Need to include mode for PSREPHEM

        if self.polyco is not None:
            cols = []
            for name in self.polyco.namelist:
                fmt, unit, array = self.polyco.dictionary[name]
                #print name, fmt, unit, array
                col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
                cols.append(col)
            polycohdr = pyfits.Header()
            for key in self.polyco.headerlist:
                polycohdr[key] = self.polyco.header[key]
            polycohdu = pyfits.BinTableHDU.from_columns(cols, name='POLYCO', header=polycohdr)
            hdulist.append(polycohdu)

        if len(self.tables) > 0:
            for table in self.tables:
                hdulist.append(table)

        if self.obs_mode == "PSR" or self.obs_mode == "CAL":
            nsubint, npol, nchan, nbin = self.shape(squeeze=False)
        elif self.obs_mode == "SEARCH":
            nsubint, nsblk, npol, nchan, nbin = self.shape(squeeze=False)

        # here fix this
        cols = []
        for name in self.subintinfolist:

            fmt, unit, array = self.subintinfo[name]
            if (array.shape[0] != nsubint and array.shape[0] != 1):
                msg = f'{name} has shape {array.shape} but file has {nsubint} subint.\n'
                msg += 'Will not save accurately shaped arrays.'
                raise ValueError(msg)


            col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
            cols.append(col)
            # finish writing out SUBINT!

        cols.append(pyfits.Column(name='DAT_FREQ', format='%iE'%np.shape(self.freq)[1], unit='MHz', array=self.freq)) #correct size? check units?
        cols.append(pyfits.Column(name='DAT_WTS', format='%iE'%np.shape(self.weights)[1], array=self.weights)) #call getWeights()

        DAT_OFFS = np.zeros((nsubint, npol*nchan), dtype=np.float32)
        DAT_SCL = np.zeros((nsubint, npol*nchan), dtype=np.float32)
        DATA = self.getData(squeeze=False, weight=False)
        saveDATA = np.zeros(self.shape(squeeze=False), dtype=np.int16)

        # Following Base/Formats/PSRFITS/unload_DigitiserCounts.C
        # TO DO: work for Search mode
        for i in xrange(nsubint):
            for j in xrange(npol):
                jnchan = j*nchan
                for k in xrange(nchan):
                    MIN = np.min(DATA[i, j, k, :])
                    MAX = np.max(DATA[i, j, k, :])
                    RANGE = MAX - MIN
                    if MAX == 0 and MIN == 0:
                        DAT_SCL[i, jnchan+k] = 1.0
                    else:
                        DAT_OFFS[i, jnchan+k] = 0.5*(MIN+MAX)
                        DAT_SCL[i, jnchan+k] = (MAX-MIN)/32766.0 #this is slightly off the original value? Results in slight change of data

                    saveDATA[i, j, k, :] = np.floor((DATA[i, j, k, :] - DAT_OFFS[i, jnchan+k])/DAT_SCL[i, jnchan+k] + 0.5) #why +0.5?


        offs_col = pyfits.Column(name='DAT_OFFS', format='%iE'%np.size(DAT_OFFS[0]), array=DAT_OFFS)
        cols.append(offs_col)

        scl_col = pyfits.Column(name='DAT_SCL', format='%iE'%np.size(DAT_SCL[0]), array=DAT_SCL)
        cols.append(scl_col)
        # print(scl_col.array.shape)
        dat_col = pyfits.Column(name='DATA', format='%iI'%np.size(saveDATA[0]), array=saveDATA, unit='Jy', dim='(%s,%s,%s)'%(nbin, nchan, npol))
        cols.append(dat_col) #replace the unit here
        # print(dat_col.array.shape)
        # print(nbin, nchan, npol)
        # print(saveDATA.shape)

        subinthdr = pyfits.Header()
        for key in self.subintheaderlist:
            # print(key)
            subinthdr[key] = self.subintheader[key]
        subinthdu = pyfits.BinTableHDU.from_columns(cols, name='SUBINT', header=subinthdr)
        hdulist.append(subinthdu)
        # print(subinthdr)
        # print(subinthdu.data['DATA'].shape)
        hdulist.writeto(filename, clobber=True)#clobber=True?
        return subinthdr, subinthdu

    def close(self):
        if self.verbose:
            t0 = time.time()
        self._data = None
        self.weighted_data = None
        self.weights = None

        if self.verbose:
            t1 = time.time()
            print("Unload time: %0.2f s" % (t1-t0))


        self.verbose = None
        self.file_path = None
        self.obs_mode = None
        self.subint_dtype = None
        self.n_hdrs = None
        self.nsubint = None
        self.npol = None
        self.nchan = None
        self.nbin = None

        g.collect()

    def make_HDU_rec_array(self, nrows, HDU_dtype_list, no_pad = True):
        """
        Makes a rec array with the set number of rows and data structure
        dictated by the dtype list.
        """
        #TODO Add in hdf5 type file format for large arrays?
        return np.empty(nrows, dtype=HDU_dtype_list)

    def set_data_from_array(self, data, option="strict"):
        """
        Takes in data of the form frequency vs. time produced by PSS
        options: "keep_shape", "pad", "no_pad"
        """
        self.npol = 1
        if self.verbose:
            print("Making {} subints with {} bins out of data".format(self.nsubint, self.nbin))



        if option == "strict":
            if data.shape[1] < self.nbin * self.nsubint or data.shape[1] > self.nbin*self.nsubint:
                err_msg = "Data does not match NSUBINT={} and NBIN={}.\n".format(self.nsubint, self.nbin)
                err_msg += "Expected data with shape ({}, {}), got data wth shape ({}, {})".format(self.nchan, self.nbin*self.nsubint, data.shape[0], data.shape[1])
                raise ValueError(err_msg)

            elif data.shape[0] != self.nchan:
                err_msg = "Data does not match NCHAN={}.".format(self.nchan)
                raise ValueError(err_msg)

        elif option=="keep_shape":

            if data.shape[1] < self.nbin * self.nsubint:
                missing = self.nbin*self.nsubint - data.shape[1]
                data = np.pad(data, ((0,0), (0, missing)))

            elif data.shape[1] > self.nbin*self.nsubint:
                data = data[:, :int(self.nbin*self.nsubint)]

        elif option=="pad":

            if data.shape[1] < self.nbin * self.nsubint:
                missing = self.nbin*self.nsubint - data.shape[1]
                data = np.pad(data, ((0,0), (0, missing)))

            elif data.shape[1] > self.nbin*self.nsubint:
                extra = data.shape[1] - self.nbin*self.nsubint
                self.nsubint += math.floor(float(extra)/self.nbin)
                extra = extra % self.nbin
                # print(extra)
                data = np.pad(data, ((0,0), (0,extra)))
                # print(data.shape)

        elif option=="nopad":

            if data.shape[1] < self.nbin * self.nsubint:
                missing = self.nbin*self.nsubint - data.shape[1]

                self.nsubint -= math.ceil(float(missing)/self.nbin)

                to_remove = math.ceil(float(missing)/self.nbin) * self.nbin
                data = data[:, :int(self.nbin*self.nsubint)]

            elif data.shape[1] > self.nbin*self.nsubint:
                data = data[:, :int(self.nbin*self.nsubint)]

        else:
            err_msg = "{} is not a valid reshaping option.".format(option)
            raise ValueError(err_msg)

        if data.shape[0] != self.nchan:
            if self.verbose:
                print("Setting NCHAN to {}".format(data.shape[0]))
            self.nchan = data.shape[0]

        data = np.reshape(data, (self.nchan, self.nsubint, self.nbin))
        data = np.reshape(data, (self.nsubint,  self.npol, self.nchan, self.nbin))
        self.set_subint_dims(self.nbin, self.nchan, self.npol, self.nsblk, self.nsubint, self.obs_mode, data)

        if self.verbose:
            print("NSUBINT is {}".format(self.nsubint))
        # set data
        # AND make sure everything else i.e. weighted data works


    def set_subint_dims(self, nbin=None, nchan=None, npol=None, nsblk=None,
                        nsubint=None, obs_mode=None, data=None, data_dtype='|u1'):
        """
        Method to set the appropriate parameters for the SUBINT BinTable of
            a PSRFITS file of the given dimensions.
        The parameters above are defined in the PSRFITS literature.
        The method automatically changes all the header information in the
            template dependent on these values. The header template is set to
            these values.
        A list version of a dtype array is made which has all the info needed
          to make a SUBINT recarray. This can then be written to a PSRFITS file,
          using the command write_prsfits().

        Parameters
        ----------

        nbin : int
            NBIN, number of bins. 1 for SEARCH mode data.

        nchan : int
            NCHAN, number of frequency channels.

        npol : int
            NPOL, number of polarization channels.

        nsblk : int
            NSBLK, size of the data chunks for search mode data. Set to 1 for
            PSR and CAL mode.

        nsubint : int
            NSUBINT or NAXIS2 . This is the number of rows or subintegrations
            in the PSRFITS file.

        obs_mode : str , {'SEARCH', 'PSR', 'CAL'}
            Observation mode.

        data_type : str
            Data type of the DATA array ('|u1'=int8 or '|u2'=int16).
        """
        self.nrows = self.nsubint = nsubint
        #Make a dtype list with defined dimensions and data type
        self._bytes_per_datum = np.dtype(data_dtype).itemsize

        if nsubint is not None:
            self.nsubint = nsubint
        if nbin is not None:
            self.nbin = nbin
        if nchan is not None:
            self.nchan = nchan
        if npol is not None:
            self.npol = npol
        if nsblk is not None:
            self.nsblk = nsblk


        nsubint = self.nsubint
        nbin = self.nbin
        nchan = self.nchan
        nsblk = self.nsblk
        npol = self.npol

        if obs_mode is None: obs_mode = self.obs_mode

        if obs_mode.upper() == 'SEARCH':

            if nbin != 1:
                err_msg = 'NBIN (set to {0}) parameter not set '.format(nbin)
                err_msg += 'to correct value for SEARCH mode.'
                raise ValueError(err_msg)

            self.nbits = 8 * self._bytes_per_datum
            #Set Header values dependent on data shape
            self.replace_FITS_Record('PRIMARY','BITPIX',8)
            self.replace_FITS_Record('SUBINT','BITPIX',8)
            self.replace_FITS_Record('SUBINT','NBITS',self.nbits)
            self.replace_FITS_Record('SUBINT','NBIN',nbin)
            self.replace_FITS_Record('SUBINT','NCHAN',nchan)
            self.replace_FITS_Record('PRIMARY','OBSNCHAN',nchan)
            self.replace_FITS_Record('SUBINT','NPOL',npol)
            self.replace_FITS_Record('SUBINT','NSBLK',nsblk)
            self.replace_FITS_Record('SUBINT','NAXIS2',nsubint)
            self.replace_FITS_Record('SUBINT','TFORM13',str(nchan)+'E')
            self.replace_FITS_Record('SUBINT','TFORM14',str(nchan)+'E')
            self.replace_FITS_Record('SUBINT','TFORM15',str(nchan*npol)+'E')
            self.replace_FITS_Record('SUBINT','TFORM16',str(nchan*npol)+'E')

            # self.replace_FITS_tuple('SUBINT', 'PERIOD', self.subintinfo['PERIOD'][-1][:nsubint])
            self.replace_subint_info(nsubint)

            #Calculate Number of Bytes in each row's DATA array
            tform17 = nbin*nchan*npol*nsblk
            self.replace_FITS_Record('SUBINT','TFORM17',str(tform17)+'B')

            #This is the number of bytes in TSUBINT, OFFS_SUB, LST_SUB, etc.
            bytes_in_lone_floats = 7*8 + 5*4

            naxis1 = tform17*self._bytes_per_datum + 2*nchan*4 + 2*nchan*npol*4
            naxis1 += bytes_in_lone_floats
            self.replace_FITS_Record('SUBINT','NAXIS1', naxis1)

            # Set the TDIM17 string-tuple
            tdim17 = '('+str(nbin)+', '+str(nchan)+', '
            tdim17 += str(npol)+', '+str(nsblk)+')'
            self.replace_FITS_Record('SUBINT','TDIM17', tdim17)

            self.initialize_data(obs_mode = obs_mode, data=data)
            # FIGURE OUT DATA FORMATS

            self.single_subint_floats=['TSUBINT','OFFS_SUB',
                                       'LST_SUB','RA_SUB',
                                       'DEC_SUB','GLON_SUB',
                                       'GLAT_SUB','FD_ANG',
                                       'POS_ANG','PAR_ANG',
                                       'TEL_AZ','TEL_ZEN']

        elif (obs_mode.upper() == 'PSR' or obs_mode.upper() == 'CAL'):


            if nsblk != 1:
                err_msg = 'NSBLK (set to {0}) parameter not set '.format(nsblk)
                err_msg += 'to correct value '
                err_msg += 'for {0} mode.'.format(obs_mode.upper())
                raise ValueError(err_msg)

            self.nbits = 1
            self.replace_FITS_Record('PRIMARY','BITPIX',8)
            self.replace_FITS_Record('SUBINT','BITPIX',8)
            self.replace_FITS_Record('SUBINT','NBITS',self.nbits)
            self.replace_FITS_Record('SUBINT','NBIN',nbin)
            self.replace_FITS_Record('SUBINT','NCHAN',nchan)
            self.replace_FITS_Record('PRIMARY','OBSNCHAN',nchan)
            self.replace_FITS_Record('SUBINT','NPOL',npol)
            self.replace_FITS_Record('SUBINT','NSBLK',nsblk)

            self.replace_FITS_tuple('HISTORY', 'NCHAN', nchan)
            self.replace_FITS_tuple('HISTORY', 'NSUB', nsubint)
            self.replace_FITS_tuple('HISTORY', 'NPOL', npol)
            self.replace_FITS_tuple('HISTORY', 'NBIN', nbin)
            self.replace_FITS_tuple('HISTORY', 'NBIN_PRD', nbin) # MAY NEED TO CHANGE
            self.replace_subint_info(nsubint)

            self.initialize_data(obs_mode = obs_mode, data=data)

            self.single_subint_floats=['TSUBINT','OFFS_SUB',
                                       'LST_SUB','RA_SUB',
                                       'DEC_SUB','GLON_SUB',
                                       'GLAT_SUB','FD_ANG',
                                       'POS_ANG','PAR_ANG',
                                       'TEL_AZ','TEL_ZEN',
                                       'AUX_DM','AUX_RM']

def list_arg(list_name, string):
    """Returns the index of a particular string in a list of strings."""
    return [x for x, y in enumerate(list_name) if y == string][0]
