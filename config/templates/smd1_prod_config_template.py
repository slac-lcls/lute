import numpy as np

detnames = {{ detnames }}

{%- if getROIs is defined and getROIs %}
def getROIs(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        roi_dict = {}
{% for detector, params in getROIs.items() %}
        roi_dict['ROIs'] = {{ params['ROIs'] }}
        roi_dict['writeArea'] = {{ params['writeArea'] }}
        roi_dict['thresADU'] = {{ params['thresADU'] }}

        ret_dict['{{ detector }}'] = roi_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAzIntParams is defined and getAzIntParams %}
def getAzIntParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        az_dict = {}
{% for detector, params in getAzIntParams.items() %}
        az_dict['eBeam'] = {{ params['eBeam'] }}
        az_dict['center'] = {{ params['center'] }}
        az_dict['dis_to_sam'] = {{ params['dis_to_sam'] }}
        az_dict['tx'] = {{ params['tx'] }}
        az_dict['ty'] = {{ params['ty'] }}

        ret_dict['{{ detector }}'] = az_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAzIntPyFAIParams is defined and getAzIntPyFAIParams %}
def getAzIntPyFAIParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        az_dict = {}
{% for detector, params in getAzIntPyFAIParams.items() %}
{%- if params['poni_file'] -%}
        az_dict['poni_file'] = {{ params['poni_file'] }}
{% else %}
        ai_kwargs = {}
        ai_kwargs['dist'] = {{ params['ai_kwargs']['dist'] }}
        ai_kwargs['poni1'] = {{ params['ai_kwargs']['poni1'] }}
        ai_kwargs['poni2'] = {{ params['ai_kwargs']['poni2'] }}
        az_dict['ai_kwargs'] = ai_kwargs
{% endif %}
        az_dict['npts'] = {{ params['npts'] }}
        az_dict['npts_az'] = {{ params['npts_az'] }}
        az_dict['int_units'] = {{ params['2th_deg'] }}
        az_dict['return2d'] = {{ params['return2d'] }}

        ret_dict['{{ detector }}'] = az_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getPhotonParams is defined and getPhotonParams %}
def getPhotonParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        photon_dict = {}
{% for detector, params in getPhotonsParams.items() %}
        photon_dict['ADU_per_photon'] = {{ params['ADU_per_photon'] }}
        photon_dict['thresADU'] = {{ params['thresADU'] }}

        ret_dict['{{ detector }}'] = photon_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getDropletParams is defined and getDropletParams %}
def getDropletParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        droplet_dict = {}
{% for detector, params in getDropletParams.items() %}
        droplet_dict['name'] = {{ params['name'] }}
        #droplet_dict['mask'] = None # have to pass full array
        droplet_dict['threshold'] = {{ params['threshold'] }}
        droplet_dict['thresholdLow'] = {{ params['thresholdLow'] }}
        droplet_dict['thresADU'] = {{ params['thresADU'] }}
        droplet_dict['useRms'] = {{ params['useRms'] }}
        droplet_dict['nData'] = {{ params['nData'] }}

        ret_dict['{{ detector }}'] = droplet_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getDroplet2Photons is defined and getDroplet2Photons %}
def getDroplet2Photons(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        d2p_dict = {}
{% for detector, params in getDroplet2Photons.items() %}
        d2p_dict['droplet'] = {
            'threshold': {{ params['droplet']['threshold'] }},
            'thresholdLow': {{ params['droplet']['thresholdLow'] }},
            'thresADU': {{ params['droplet']['thresADU'] }},
            'useRms': {{ params['droplet']['useRms'] }},
        }

        d2p_dict['d2p'] = {
            'aduspphot': {{ params['aduspphot'] }},
            'cputime': {{ params['cputime'] }},
        }
        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getSvdParams is defined and getSvdParams %}
def getSvdParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        svd_dict = {}
{% for detector, params in getSvdParams.items() %}
        svd_dict['name'] = {{ params['name'] }}
        svd_dict['n_components'] = {{ params['n_components'] }}
        svd_dict['n_pulse'] = {{ params['n_pulse'] }}
        svd_dict['basis_file'] = {{ params['basis_file'] }}
        svd_dict['delay'] = {{ params['delay'] }}
        #svd_dict['mode'] = {{ params['mode'] }}
        svd_dict['return_reconstructed'] = {{ params['return_reconstructed'] }}

        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAutocorrParams is defined and getAutocorrParams %}
def getAutocorrParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
        autocorr_dict = {}
{% for detector, params in getAutocorrParams.items() %}
        autocorr_dict['name'] = {{ params['name'] }}
        autocorr_dict['thresADU'] = {{ params['thresADU'] }}
        autocorr_dict['save_lineout'] = {{ params['save_lineout'] }}
        autocorr_dict['save_range'] = {{ params['save_range'] }}
        autocorr_dict['illumination_correction'] = {{ params['illumination_correction'] }}

        ret_dict['{{ detector }}'] = {{ params }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getProjection_ax0 is defined %}
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


{%- if getProjection_ax1 is defined %}
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


{%- if detSumAlgos is defined and detSumAlgos %}
def getDetSums(run):
    if isinstance(run, str):
        run = int(run)
    ret_dict = {}
    if run > 0:
{% for detector, params in detSumAlgos.items() %}
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
{% if epicsOncePV is defined %}
epicsOncePV = {{ epicsOncePV }}
{% else %}
epicsOncePV = []
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
