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
# These lists are either PV names, aliases, or tuples with both.
# epicsPV = ['las_fs14_controller_time']
# epicsOncePV = ['m0c0_vset', ('TMO:PRO2:MPOD:01:M2:C3:VoltageMeasure', 'MyAlias'),
#               'IM4K4:PPM:SPM:VOLT_RBV', "FOO:BAR:BAZ", ("X:Y:Z", "MCBTest"), "A:B:C"]
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
# This is a list of float to fix the Timetool calibration if necessary. 
{% if ttCalib is defined %}
ttCalib = {{ ttCalib }} #[]
{% else %}
ttCalib = []
{% endif %}
# This is a list of analog inputs to save and give them a name.
# aioParams = [[1], ['laser']]
{% if aioParams is defined %}
aioParams = {{ aioParams }} # []
{% else %}
aioParams = []
{% endif %}
