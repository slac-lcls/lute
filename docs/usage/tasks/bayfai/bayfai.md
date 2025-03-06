
# BayFAI User Documentation {: #top}

<a name="toc"></a> **Jump to:**
- [`BayFAI Configuration`](#bayfai-configuration)
- [`Running BayFAI from the Command-Line`](#running-bayfai-from-the-command-line)
- [`Running BayFAI from the eLog`](#running-bayfai-from-the-elog)
- [`Running only BayFAI Geometry Calibration`](#running-only-bayfai-geometry-calibration)

---
## BayFAI Configuration

### Preliminaries `lute`

BayFAI is run within the newer version of `btx`, `lute` standing for LCLS Unified Task Executor. This next iteration of `btx` is still in development.
Due to its recent implement, BayFAI has not yet been merged in the main branch of `lute`.

A stable and up to date version with BayFAI can be found at `/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/lute`.

### Preliminaries `smalldata`

BayFAI needs a powder image to perform the calibration. `smalldata` does it for us.
A stable and up to date version working with BayFAI can be found at `/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/smalldata_tools`.

### Experiment Configuration

Each experiment requires a customized YAML configuration file.

1. Navigate to the experiment scratch folder and create a lute folder:
    ```bash
    (base) [lconreux@sdfiana002 ~] cd /sdf/data/lcls/ds/<hutch>/<experiment>/scratch/
    (base) [lconreux@sdfiana002 scratch] mkdir bayfai
    ```

2. Navigate to the lute working directory and create useful subfolders:
    ```bash
    (base) [lconreux@sdfiana002 scratch] cd bayfai
    (base) [lconreux@sdfiana002 bayfai] mkdir yamls
    (base) [lconreux@sdfiana002 bayfai] mkdir launchpad
    (base) [lconreux@sdfiana002 bayfai] mkdir smd_output
    ```

3. Fetch a config yaml:
    A template config yaml can be found at : `/sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/yamls/config.yaml`.
    Copy this config yaml to the scratch folder with appropriate experiment tag:
    ```bash
    (base) [lconreux@sdfiana002 bayfai] cp /sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/yamls/config.yaml yamls/<experiment>.yaml
    ```
    At this point, the working directory should look like this: 
    ```bash
    (base) [lconreux@sdfiana002 bayfai]$ tree
    .
    ├── launchpad
    ├── smd_output
    └── yamls
        └── <experiment>.yaml
    ```

4. Fill in the blanks in the config yaml:
    A template config yaml has been created but the user needs to fill in some important information. Here what the template config file looks like:
```bash
date: 2023/10/25
lute_version: 0.1
experiment: <experiment> # If launch from eLog, erase that line
run: <run>               # If launch from eLog, erase that line
task_timeout: 1200
title: LUTE Task Configuration
work_dir: /sdf/data/lcls/ds/<hutch>/<experiment>/scratch/bayfai/ # Fill this line 
---
OptimizePyFAIGeometry:
bo_params:
    bounds:
    dist: <guess distance> # Fill this line with guessed detector distance
    poni1:
    - -0.01
    - 0.01
    poni2:
    - -0.01
    - 0.01
    res: 0.0002         
calibrant: <calibrant> # Fill this line with calibrant name (AgBh, LaB6...)
det_type: <detector>   # Fill this line with detector name (epix10k2M, jungfrau4M, Rayonix...)
SubmitSMD:
detSumAlgos:
    Rayonix:
    - calib_skipFirst_thresADU1
    - calib_skipFirst_max
    all:
    - calib
    - calib_dropped
    - calib_dropped_square
    - calib_thresADU1
    epix10k2M:
    - calib_thresADU5
    - calib_max
    jungfrau4M:
    - calib_thresADU5
    - calib_max
detnames:
- <detector> # Fill this line with detector name (epix10k2M, jungfrau4M, Rayonix...)
directory: /sdf/data/lcls/ds/<hutch>/<experiment>/scratch/lute_output/smd_output/ # Fill this line 
producer: /sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/smalldata_tools/lcls1_producers/smd_producer.py # If no smalldata cloned in experiment folder, use this one!
...
```
    BayFAI's config template is divided into three parts, the `lute_config`: basic experiment configuration (top top of the yaml), `OptimizePyFAIGeometry`: BayFAI required parameters, `SMDSubmit`: smalldata required parameters
    a. `lute_config`:
        - If launched from eLog, erase the experiment and run lines.
        - Fill in the correct working directory `/sdf/data/lcls/ds/<hutch>/<experiment>/scratch/bayfai/`.
    b. `OptimizePyFAIGeometry`:
        - Fill in a guessed detector distance, BayFAI will scan around that distance in the following manner [guess-50mm; guess+50mm] with a step size of 1mm.
        - Fill in the calibrant name, (usually AgBh or LaB6) (list of all calibrant available: [ressources](https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration)).
        - Fill in the detector type name, as it is defined in the psana environment (epix10k2M, jungfrau4M, Rayonix, Epix10kaQuad...).
    c. `SubmitSMD`:
        - Fill the output directory for smalldata.
        - Don't touch the producer if no smalldata repo was cloned in your experiment (I'd recommend even to never touch it!).

## Running BayFAI from the Command-Line

Skip this section if you are interested in launching BayFAI from the [eLog](#running-bayfai-from-the-elog)

## Running BayFAI from the eLog

1. Go to the Workflow Definition Panel:
    - Click on Add a Workflow
    - Fill in the Workflow Definition informations:
        - Name: BayFAI
        - Trigger: Manually triggered
        - Location: S3DF
        - Executable: /sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/lute/launch_scripts/submit_launch_airflow.sh
        - Parameters: /sdf/data/lcls/ds/prj/prjlute22/results/benchmarks/geom_opt/lute/launch_scripts/launch_airflow.py -w bayfai -c /sdf/data/lcls/ds/<hutch>/<experiment>/scratch/bayfai/yamls/<experiment>.yaml --partition=milano --ntasks=102 --account=lcls:<experiment> --nodes=1 --test
2. Go to the Workflow Control Panel:
    - Trigger BayFAI for the desired run!

| ![BayFAI workflow controls from the eLog](images/bayfai-controls.png) | 
|:---------------------------------------------------------------:| 
|            __BayFAI workflow controls from the eLog.__             |

3. Monitor the Results (after a couple of minutes!):
    - Geometry is posted to the eLog along with the Resolution range covered by the detector.
    - Fitting plots along with BayFAI metrics can be found in the "Summaries" page
The measured distance between sample and detector will eventually be reported in the Workflow controls tab. 

| ![BayFAI reporting of geometry inferred from Silver Behenate run](images/bayfai-geom.png) | 
|:-----------------------------------------------------------------------------------:| 
|                      __BayFAI reporting of geometry inferred from Silver Behenate run.__                       |

Fitting plots will can be found in the "Summaries" page (go to ***runs > r0010*** where 10 is the run number).


| !BayFAI summary of geometry inferred from Silver Behenate run | 
|:-----------------------------------------------------------------------------------------:| 
|              __BayFAI summary of geometry inferred from Silver Behenate run.__               |


## Running only BayFAI Geometry Calibration

---