import numpy as np

# These lists are needed, do not delete them
# If no detector in a given category, leave the corresponding
# list empty.
# detectors = ['jungfrau','epix100']
{%- if detnames is defined and detnames %}
detectors = {{ detnames }}
{% else %}
detectors = []
{% endif %}
{%- if integrating_detectors is defined and integrating_detectors %}
integrating_detectors = {{ integrating_detectors }}
{% else %}
integrating_detectors = []
{% endif %}

def get_intg(run):
    """
    Returns
    -------
    intg_main (str):  This detector ill be passed to the psana datasource. It should be the SLOWEST of
                all integrating detectors in the data
    intg_addl (list of str): The detectors in this list will be analyzed as integrating detectors. It is
                             important that the readout frequency of these detectors is commensurate and
                             in-phase with intg_main.
    """
    run = int(run)
    intg_main = ""
    intg_addl = []
    if run > 0:
{%- if get_intg is defined and get_intg %}
    {%- if get_intg["intg_main"] and get_intg["intg_main"] %}
        intg_main = "{{ get_intg['intg_main'] }}"
    {% endif %}
    {%- if get_intg["intg_addl"] and get_intg["intg_addl"] %}
        {%- for det in get_intg["intg_addl"] %}
        intg_addl.append("{{ det }}")
        {% endfor %}
    {% endif %}
{% endif %}
    return intg_main, intg_addl

{%- if getROIs is defined and getROIs %}
def getROIs(run):
    ret_dict = {}

    jungfrau_roi = {"thresADU": None, "writeArea": True, "calcPars": False, "ROI": None}
    epix100_roi = {"thresADU": None, "writeArea": True, "calcPars": False, "ROI": None}

    if run > 0:
        roi_dict = {}
{% for detector, params in getROIs.items() %}
        roi_dict["ROIs"] = {{ params["ROIs"] }}
        roi_dict["writeArea"] = {{ params["writeArea"] }}
        roi_dict["thresADU"] = {{ params["thresADU"] }}
        roi_dict["calcPars"] = {{ params["calcPars"] }}

        ret_dict["{{ detector }}"] = roi_dict
{% endfor %}
        ...
    return ret_dict
{% endif %}


{%- if getDroplet2Photons is defined and getDroplet2Photons %}

def get_droplet2photon(run):
    ret_dict = {}

    if run > 0:
        d2p_dict = {}
{% for detector, params in getDroplet2Photons.items() %}
        d2p_dict["droplet"] = {
            "threshold": {{ params["droplet"]["threshold"] }},
            "thresholdLow": {{ params["droplet"]["thresholdLow"] }},
            "thresADU": {{ params["droplet"]["thresADU"] }},
            "useRms": {{ params["droplet"]["useRms"] }},
        }
        d2p_dict["d2p"] = {
            "aduspphot": 20,
            "cputime": {{ params["cputime"] }},
        }
        ret_dict["{{ detector }}"] = {{ params }}
        d2p_dict["nData"] = None
        d2p_dict["get_photon_img"] = False

        ret_dict["epix100"] = d2p_dict
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
epicsPV = {{ epicsPV }}
{% else %}
epicsPV = []
{% if epicsOncePV is defined %}
epicsOncePV = {{ epicsOncePV }}
{% else %}
epicsOncePV = []
{% endif %}

##########################################################
# psplot config
##########################################################

import psplot
