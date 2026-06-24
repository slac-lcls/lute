{%- from "macros.jinja" import step_parameters, step_value -%}
import numpy as np

detnames = {{ detnames }}

{%- if getROIs is defined and getROIs %}
def getROIs(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getROIs.items() %}
        roi_dict = {}
{{- step_parameters("roi_dict", detector, params) }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAzIntParams is defined and getAzIntParams %}
def getAzIntParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAzIntParams.items() %}
        az_dict = {}
        {%- if 'userMask' in params %}
        az_dict['userMask'] = np.load("{{ params['userMask'] }}")
        {{- step_parameters("az_dict", detector, params|rejectattr("userMask")) }}
        {% else %}
        {{- step_parameters("az_dict", detector, params) }}
        {% endif %}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAzIntPyFAIParams is defined and getAzIntPyFAIParams %}
def getAzIntPyFAIParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAzIntPyFAIParams.items() %}
        az_dict = {}
{{- step_parameters("az_dict", detector, params) }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getPhotonParams is defined and getPhotonParams %}
def getPhotonParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getPhotonsParams.items() %}
        photon_dict = {}
{{- step_parameters("photon_dict", detector, params) }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getDropletParams is defined and getDropletParams %}
def getDropletParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getDropletParams.items() %}
        droplet_dict = {}
{{- step_parameters("droplet_dict", detector, params) }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getDroplet2Photons is defined and getDroplet2Photons %}
def getDroplet2Photons(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getDroplet2Photons.items() %}
        d2p_dict = {}
        droplet_dict = {}
{{- step_parameters("droplet_dict", detector, params['droplet'], False) }}
        d2p_dict['droplet'] = droplet_dict
        d2p_dict['d2p'] = {
            'aduspphot': {{ params['aduspphot'] }},
            'cputime': {{ params['cputime'] }},
        }
        ret_dict['{{ detector }}'] = d2p_dict
{% endfor %}
    return ret_dict
{% endif %}


{%- if getSvdParams is defined and getSvdParams %}
def getSvdParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getSvdParams.items() %}
        svd_dict = {}
{{- step_parameters("svd_dict", detector, params) }}
{% endfor %}
    return ret_dict
{% endif %}


{%- if getAutocorrParams is defined and getAutocorrParams %}
def getAutocorrParams(run):
    if isinstance(run,str):
        run=int(run)
    ret_dict = {}
    if run>0:
{% for detector, params in getAutocorrParams.items() %}
        autocorr_dict = {}
{{- step_parameters("autocorr_dict", detector, params) }}
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
{{- step_value(detector, params) }}
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
{{- step_value(detector, params) }}
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
{{- step_value(detector, params) }}
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
