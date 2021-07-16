# -*- coding: utf-8 -*-
# encoding=utf8
"""Main module."""

#Pulsar Data Toolbox. Based on fitsio package. See https://github.com/esheldon/fitsio for details.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import astropy.io.fits as fits
from astropy.io.fits import hdu
import collections, os, sys
import datetime
import warnings
import six

#package_path = os.path.dirname(__file__)
#template_dir = os.path.join(package_path, './templates/')

class psrfits(hdu.hdulist.HDUList):

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
            self.fits_template = fits.open(template_path, mode='readonly')

            if self.obs_mode is None:
                OBS = self.fits_template[0].header['OBS_MODE'].strip()
                self.obs_mode = OBS
            else:
                self.obs_mode = obs_mode


            # initializing more fields

            self.draft_hdrs = collections.OrderedDict()
            self.HDU_drafts = {}
            self.subint_dtype = None

            #Set the ImageHDU to be called primary.
            self.draft_hdrs['PRIMARY'] = self.fits_template[0].header
            self.n_hdrs = len(self)

            for ii in np.arange(1,self.n_hdrs): # for each HDU/header
                hdr_key = self.fits_template[ii].name
                self.draft_hdrs[hdr_key] = self.fits_template[ii].header
                self.HDU_drafts[hdr_key] = None # no HDU yet
            self.draft_hdr_keys = list(self.draft_hdrs.keys()) # get all header keys used

            if verbose:
                msg = 'Making new {0} mode PSRFITS file '.format(self.obs_mode)
                msg += 'using template from path:\n'
                msg += '    \'{0}\'. \n'.format(template_path)
                msg += 'Writing to path: \n    \'{0}\''.format(psrfits_path)
                print(msg)

        #fits_file = open(psrfits_path)
        #super().__init__(file = fits_file)
        temp = hdu.hdulist.fitsopen(name = psrfits_path)
        temp_hdus = [temp[i] for i in range(len(temp))]
        #print(temp_hdus)
        super().__init__(hdus = temp_hdus)

        #If self.obs_mode is still None use loaded PSRFITS file
        if self.obs_mode is None and from_template: # i think this happens twice? maybe?
            OBS = self.fits_template[0].header['OBS_MODE'].strip() # [fitsio] get obs_mode from template
            self.obs_mode = OBS

        if from_template and verbose:
            print('The Binary Table HDU headers will be written as '
                  'they are added\n     to the PSRFITS file.')

        elif not from_template and (mode=='rw' or mode=='READWRITE'): # if you want to write stuff
            self.draft_hdrs = collections.OrderedDict()
            self.HDU_drafts = {}
            #Set the ImageHDU to be called primary.
            #print(help(self))
            self.draft_hdrs['PRIMARY'] = self[0].header
            self.n_hdrs = len(self)
            self.written = False
            for ii in range(self.n_hdrs-1):
                hdr_key = self[ii+1].name
                self.draft_hdrs[hdr_key] = self[ii+1].header
                self.HDU_drafts[hdr_key] = None
            self.draft_hdr_keys = list(self.draft_hdrs.keys())

    def write_psrfits(self, HDUs=None, hdr_from_draft=True):
        """
        Function that takes the template headers and a dictionary of recarrays
            to make into PSRFITS HDU's. These should only include BinTable HDU
            Extensions, not the PRIMARY header (an ImageHDU). PRIMARY is dealt
            with a bit differently.

        Parameters
        ----------

        HDUs : dict, optional
            Dictionary of recarrays to make into HDUs. Default is set to
            HDU_drafts
        """
        if self.written:
            raise ValueError('PSRFITS file has already been written. '
                             'Can not write twice.')
        if HDUs is None:
            HDUs = self.HDU_drafts

        if any([val is None for val in HDUs.values()]):
            raise ValueError('One of HDU drafts is \"None\".')

        self.write_PrimaryHDU_info_dict(self.fits_template[0],self[0])
        self.set_hdr_from_draft('PRIMARY')
        for hdr in self.draft_hdr_keys[1:]:
            self.write_table(HDUs[hdr],extname=hdr, extver=1) # NEED TO FIX THIS
                             # header = self.draft_hdrs[hdr])
            if hdr_from_draft: self.set_hdr_from_draft(hdr)
        self.written = True

    # def write_psrfits_from_draft?(self):
    #     self.write_PrimaryHDU_info_dict(self.fits_template[0],self[0])
    #     self.set_hdr_from_draft('PRIMARY')
    #     #Might need to go into for loop if not true for all BinTables
    #     nrows = self.draft_hdrs['SUBINT']['NAXIS2']
    #     for jj, hdr in enumerate(self.draft_hdr_keys[1:]):
    #         HDU_dtype_list = self.get_HDU_dtypes(self.fits_template[jj+1])
    #         rec_array = self.make_HDU_rec_array(nrows, HDU_dtype_list)
    #         self.write_table(rec_array)
    #         self.set_hdr_from_draft(hdr)

    # def append_subint_array(self,table):
    #     """
    #     Method to append more subintegrations to a PSRFITS file from Python
    #      arrays.
    #     The array must match the columns (in the numpy.recarray sense)
    #      of the existing PSRFITS file.
    #     """
    #     fits_to_append = F.FITS(table)
    def append_from_file(self,path,table='all'):
        """
        Method to append more subintegrations to a PSRFITS file from other
        PSRFITS files.
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
        PF2A = fits.open(path, mode='readonly')
        PF2A_hdrs = []
        PF2A_hdrs.append('PRIMARY')
        for ii in range(self.n_hdrs-1):
            hdr_key = PF2A[ii+1].name
            PF2A_hdrs.append(hdr_key)
        if table=='all':
            if PF2A_hdrs!= self.draft_hdr_keys:
                if len(PF2A_hdrs)!= self.n_hdrs:
                    err_msg = '{0} and {1} do '.format(self.psrfits_path, path)
                    err_msg += 'not have the same number of BinTable HDUs.'
                    raise ValueError(err_msg)
                else:
                    err_msg = 'Original PSRFITS HDUs'
                    err_msg = ' ({0}) and PSRFITS'.format(self.draft_hdr_keys)
                    err_msg = ' to append ({1})'.format(PF2A_hdrs)
                    err_msg = ' have different BinTables or they are in'
                    err_msg = ' different orders. \nEnter a table list matching'
                    err_msg = ' the order of the orginal PSRFITS file.'
                    raise ValueError(err_msg)
            else:
                table=PF2A_hdrs # set table of stuff

        for hdr in self.draft_hdr_keys[1:]: # for each non-PRIMARY header
            rec_array = PF2A[list_arg(table,hdr)].data
            #print(self[list_arg(self.draft_hdr_keys,hdr)].data.shape)
            self[list_arg(self.draft_hdr_keys,hdr)].data = np.append(self[list_arg(self.draft_hdr_keys,hdr)].data, rec_array, axis=0)



    def get_colnames(self):
        """Returns the names of all of the columns of data needed for a PSRFITS
        file."""
        return self[1].columns.names

    def set_hdr_from_draft(self, hdr):
        """Sets a header of the PSRFITS file using the draft header derived from
        template."""
        keys = self.draft_hdr_keys
        if isinstance(hdr,int): # if hdr is an int
            hdr_name = keys[hdr]
        if isinstance(hdr, six.string_types): # if hdr i a string
            hdr_name = hdr.upper()
            hdr = list_arg(keys,hdr_name) # get index
        # with warnings.catch_warnings(): #This is very Dangerous
        #     warnings.simplefilter("ignore")

        for card in self.draft_hdrs[hdr_name].cards: # should not be modified directly
            print(card)
            #print(type(card.rawvalue))
            #card.rawvalue = card.value.encode('ascii', 'ignore')
            #card.keyword = card.keyword.encode('ascii', 'ignore')

        self[hdr].write_keys(self.draft_hdrs[hdr_name],clean=False)
        #Must set clean to False or the first keys are deleted!

    def get_FITS_card_dict(self, hdr, name):
        """
        Make a FITS card compatible dictionary from a template FITS header that
        matches the input name key in a standard FITS card/record. It is
        necessary to make a new FITS card/record to change values in the header.
        This function outputs a writeable dictionary which can then be used to
        change the value in the header using the hdr.add_record() method.

        Parameters
        ----------

        hdr : fitsio.fitslib.FITSHDR object [fitsio]
            Template for the card.

        name : str
            The name key in the FITS record you wish to make.
        """
        card = next((item for item in hdr.cards
                    if item[0] == name.upper()), False)
        if not card:
            err_msg = 'A FITS card named '
            err_msg += '{0} does not exist in this HDU.'.format(name)
            raise ValueError(err_msg)
        return card


        # STOPPING POINT
    def make_FITS_card(self, hdr, name, new_value):
        """
        Make a new FITS card/record using a FITS header as a template.
        This function makes a new card by finding the card/record in the
        template with the same name and replacing the value with new_value.
        Note: fitsio will set the dtype dependent on the form of the new_value
        for numbers.

        Parameters
        ----------

        hdr : fitsio.fitslib.FITSHDR [fitsio]
            A fitsio.fitslib.FITSHDR object, which acts as the template.

        name : str
            A string that matches the name key in the FITS record you wish to
            make.

        new_value : str, float
            The new value you would like to replace.
        """
        record = self.get_FITS_card_dict(hdr,name)
        record_value = record['value']
        dtype = record['dtype']

        string_dtypes = ['C']
        number_dtypes = ['I','F']

        def _fits_format(new_value,record_value):
            """
            Take in the new_value and record value, and format for searching
            card string. Change the shape of the string to fill out PSRFITS
            File Correctly.
            """
            try: #when new_value is a string
                if len(new_value)<=len(record_value):
                    str_len = len(record_value)
                    new_value = new_value.ljust(str_len) # left justify and fill
                card_string = record['card_string'].replace(record_value, # [fitsio] card
                                                            new_value)

            except TypeError: # When new_value is a number
                old_val_str = str(record_value)
                old_str_len = len(old_val_str)
                new_value = str(new_value)
                new_str_len = len(new_value)
                if new_str_len < old_str_len:
                    # If new value is shorter fill out with spaces.
                    new_value = new_value.rjust(old_str_len)
                elif new_str_len > old_str_len:
                    if new_str_len>20:
                        new_value=new_value[:20]
                        new_str_len = 20

                    # If new value is longer pull out more spaces.
                    old_val_str = old_val_str.rjust(new_str_len)
                card_string = record['card_string'].replace(old_val_str, # [fitsio] card
                                                            new_value) # replace, possibly keeping other stuff?
            return card_string

        def _replace_center_of_cardstring(new_value):
            """
            Replaces the entire center of the card string using the new value.
            """
            cardstring = record['card_string']
            equal_idx = old_cardstring.find('=') # what is old_cardstring?
            slash_idx = old_cardstring.find('/')
            len_center = slash_idx - equal_idx - 1
            new_center = str(new_value).rjust(len_center)
            cardstring[equal_idx+1, slash_idx] = new_center # just make the cardstring the value
            return cardstring

        #if isinstance(record['value'],tuple):
        #    record['value'] = str(record['value']).replace(' ','')
        # for TDIM17, TDIM20 in SUBINT HDU...
        # Could make more specific if necessary.
        special_fields = ['TDIM17','TDIM20']

        if record['name'] in special_fields: # if it's TDIM17 or TDIM20
            new_record = record
            record_value = str(record_value).replace(' ','')
            card_string = _fits_format(new_value.replace(' ',''), record_value)
            new_record['card_string'] = card_string.replace('\' (','\'(')
            new_record['value'] = new_value
            new_record['value_orig'] = new_record['value']

        #TODO Add error checking new value... and isinstance(new_value)
        #Usual Case
        elif str(record['value']) in record['card_string']: # if card_string has been made w/ value
            card_string = _fits_format(new_value, record_value) # replace it
            new_record = F.FITSRecord(card_string) # [fitsio]

        # IF NOT, then reformat

        #Special Case 1, Find Numbers with trailing zeros and writes string.
        elif ((str(record['value'])[-1]=='0')
              and (str(record['value'])[:-1] in record['card_string'])):

            record_value = str(record['value'])[:-1]
            #Adds decimal pt to end of string.
            if record_value[-1]=='.' and str(new_value)[-1]!='.':
                new_value = str(new_value) + '.'
            card_string = _fits_format(new_value, record_value)
            new_record = F.FITSRecord(card_string) # [fitsio]

        #Special Case 2, Find Numbers with upper/lower E in sci notation
        #that do not match exactly. Always use E in card string.
        elif (('dtype' in record.keys()) # if something of the type is in the keys
              and (record['dtype'] in number_dtypes) # and it's a number type
              and (('E' in str(record_value)) or ('e' in str(record_value)) # and it has an e somewhere
                    or ('E' in str(record['value_orig']))
                    or ('e' in str(record['value_orig'])))):

            new_value = str(new_value).upper()
            if str(record_value).upper() in record['card_string']: # test uppercase
                record_value = str(record_value).upper()
                card_string = _fits_format(new_value, record_value)
                new_record = F.FITSRecord(card_string) # [fitsio]
            elif str(record_value).lower() in record['card_string']: # test lowercase
                record_value =str(record_value).lower()
                card_string = _fits_format(new_value, record_value)
                new_record = F.FITSRecord(card_string) # [fitsio]
            else:
                card_string = _replace_center_of_cardstring(new_value) # just replace entire center
                new_record = F.FITSRecord(card_string) # [fitsio]
                msg = 'Old value cannot be found in card string. '
                msg += 'Entire center replaced.'
                print(msg)

        #Replace whole center if can't find value.
        else:
            card_string = _replace_center_of_cardstring(new_value)
            new_record = F.FITSRecord(card_string)
            msg = 'Old value cannot be found in card string. '
            msg += 'Entire center replaced.'
            print(msg)

        if new_record['value'] != new_record['value_orig']:
            new_record['value_orig'] = new_record['value'] # also change the value to match new cardstring

        return new_record

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
