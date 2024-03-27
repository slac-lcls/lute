#!/usr/bin/env python

import numpy as np
import psana
import time
from datetime import datetime
begin_job_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
import argparse
import socket
import os
import logging 
import requests
import sys
from glob import glob
from PIL import Image
from requests.auth import HTTPBasicAuth

########################################################## 
##
## User Input start --> 
##
########################################################## 
##########################################################
# functions for run dependant parameters
##########################################################

{% if getROIs is defined %}
# 1) REGIONS OF INTEREST
def getROIs(run):
    """ Set parameter for ROI analysis. Set writeArea to True to write the full ROI in the h5 file.
    See roi_rebin.py for more info
    """
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        roi_dict = {}
{% for detector, params in getROIs.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getAzIntParams is defined %}
# 2) AZIMUTHAL INTEGRATION
def getAzIntParams(run):
    """ Parameters for azimuthal integration
    See azimuthalBinning.py for more info
    """
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAzIntParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getAzIntPyFAIParams is defined %}
def getAzIntPyFAIParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAzIntPyFAIParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getPhotonParams is defined %}
# 3) PHOTON COUNTING AND DROPLET
# Photon
def getPhotonParams(run):
    """ Parameters for droplet algorithm
    See photons.py for more info
    """
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getPhotonsParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getDropletParams is defined %}
# Droplet algorithm
def getDropletParams(run):
    """ Parameters for droplet algorithm
    See droplet.py for more info
    """
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getDropletParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getDroplet2Photons is defined %}
# Droplet to photon algorithm (greedy guess)
def getDroplet2Photons(run):
    """ Set parameter for droplet2photon analysis. The analysis uses two functions, each with their
    own dictionary of argument:
        1. droplet: see droplet.py for args documentation
        2. photonize droplets: see droplet2Photons.py for args documentation
    """
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getDroplet2Photons.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getSvdParams is defined %}
# 4) WAVEFORM ANALYSIS (SVD, peak finding)
def getSvdParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getSvdParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getAutocorrParams is defined %}
# 5) AUTOCORRELATION
def getAutocorrParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAutocorrParams.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getProjection_ax0 is defined %}
# 6) PROJECTIONS (ROI or full detector)
def getProjection_ax0(run):
    if isinstance(run, str):
        run = int(run)
    ret_dict = {}

    if run > 0:
{% for detector, params in getProjection_ax0.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

{% if getProjection_ax1 is defined %}
def getProjection_ax1(run):
    if isinstance(run, str):
        run = int(run)
    ret_dict = {}

    if run > 0:
{% for detector, params in getProjection_ax1.items() %}
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}

##########################################################
# run independent parameters 
##########################################################
#aliases for experiment specific PVs go here
#epicsPV = ['slit_s1_hw']
{% if epicsPV is defined %}
epicsPV = {{ epicsPV }} #[]
{% else %}
epicsPV = []
{% endif %}
#fix timetool calibration if necessary
#ttCalib=[0.,2.,0.]
{% if ttCalib is defined %}
ttCalib = {{ ttCalib }} #[]
{% else %}
ttCalib = []
{% endif %}
#ttCalib=[1.860828, -0.002950]
#decide which analog input to save & give them nice names
#aioParams=[[1],['laser']]
{% if aioParams is defined %}
aioParams = {{ aioParams }} # []
{% else %}
aioParams = []
{% endif %}
########################################################## 
##
## <-- User Input end
##
##########################################################


# DEFINE DETECTOR AND ADD ANALYSIS FUNCTIONS
def define_dets(run):
    detnames = {{ detnames }} # ['jungfrau1M', 'epix_alc1'] # add detector here
    dets = []
    
    # Load DetObjectFunc parameters (if defined)
    try:
        ROIs = getROIs(run)
    except Exception as e:
        print(f'Can\'t instantiate ROI args: {e}')
        ROIs = []
    try:
        az = getAzIntParams(run)
    except Exception as e:
        print(f'Can\'t instantiate azimuthalBinning args: {e}')
        az = []
    try:
        az_pyfai = getAzIntPyFAIParams(run)
    except Exception as e:
        print(f'Can\'t instantiate AzIntPyFAI args: {e}')
        az_pyfai = []
    try:
        phot = getPhotonParams(run)
    except Exception as e:
        print(f'Can\'t instantiate Photon args: {e}')
        phot = []
    try:
        drop = getDropletParams(run)
    except Exception as e:
        print(f'Can\'t instantiate Droplet args: {e}')
        drop = []
    try:
        drop2phot = getDroplet2Photons(run)
    except Exception as e:
        print(f'Can\'t instantiate Droplet2Photons args: {e}')
        drop2phot = []
    try:
        auto = getAutocorrParams(run)
    except Exception as e:
        print(f'Can\'t instantiate Autocorrelation args: {e}')
        auto = []
    try:
        svd = getSvdParams(run)
    except Exception as e:
        print(f'Can\'t instantiate SVD args: {e}')
        svd = []

    # For retrieving projection parameters
    try:
        proj_ax0 = getProjection_ax0(run)
    except Exception as e:
        print(f"Can't instantiate Projection_ax1 args: {e}")
        proj_ax0 = []
    try:
        proj_ax1 = getProjection_ax1(run)
    except Exception as e:
        print(f"Can't instantiate Projection_ax1 args: {e}")
        proj_ax1 = []

    # Define detectors and their associated DetObjectFuncs
    for detname in detnames:
        havedet = checkDet(ds.env(), detname)
        # Common mode
        if havedet:
            if detname=='': 
                # change here to specify common mode for detname if desired. Else default is used
                common_mode=0
            else:
                common_mode=None
            det = DetObject(detname ,ds.env(), int(run), common_mode=common_mode)
            
            # Analysis functions
            # ROIs:
            if detname in ROIs:
                for iROI,ROI in enumerate(ROIs[detname]['ROIs']):
                    detROIFunc = ROIFunc(name=f"ROI_{iROI}",
                                         ROI=ROI,
                                         writeArea=ROIs[detname]['writeArea'],
                                         thresADU=ROIs[detname]['thresADU'])
                    # APPLYING PROJECTIONS TO ROIs
                    if detname in proj_ax0:
                        projax0Func = projectionFunc(**proj_ax0[detname])
                        detROIFunc.addFunc(projax0Func)
                    if detname in proj_ax1:
                        projax1Func = projectionFunc(**proj_ax1[detname])
                        detROIFunc.addFunc(projax1Func)
                    det.addFunc(detROIFunc)


            # Azimuthal binning
            if detname in az:
                det.addFunc(azimuthalBinning(**az[detname]))
            if detname in az_pyfai:
                det.addFunc(azav_pyfai(**az_pyfai[detname]))
            
            # psana photon count
            if detname in phot:
                det.addFunc(photonFunc(**phot[detname]))
            
            # Droplet algo
            if detname in drop:
                if nData in drop[detname]:
                    nData = drop[detname].pop('nData')
                else:
                    nData = None
                func = dropletFunc(**drop[detname])
                func.addFunc(roi.sparsifyFunc(nData=nData))
                det.addFunc(func)
            
            # Droplet to photons
            if detname in drop2phot:
                if 'nData' in drop2phot[detname]:
                    nData = drop2phot[detname].pop('nData')
                else:
                    nData = None
                # getp droplet dict
                droplet_dict = drop2phot[detname]['droplet']
                #get droplet2Photon dict
                d2p_dict = drop2phot[detname]['d2p']
                dropfunc = dropletFunc(**droplet_dict)
                drop2phot = droplet2Photons(**d2p_dict)
                # add sparsify to put photon coord to file
                sparsify = sparsifyFunc(nData=nData)
                # assemble function stack: droplet -> photon -> sparsify
                drop2phot.addFunc(sparsify)
                dropfunc.addFunc(drop2phot)
                det.addFunc(dropfunc)
            
            # Autocorrelation
            if detname in auto:
                det.addFunc(Autocorrelation(**auto[detname]))
            
            # SVD waveform analysis
            if detname in svd:
                det.addFunc(svdFit(**svd[detname]))

            det.storeSum(sumAlgo='calib')
            #det.storeSum(sumAlgo='calib_img')
            dets.append(det)
    return dets
