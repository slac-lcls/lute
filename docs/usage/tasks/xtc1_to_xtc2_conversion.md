# Converting XTC1 to XTC2 Files - `ReadXtc1` and `WriteXtc2`
## Introduction - Overview of the XTC2 Format with Chunks
As the conversion from the XTC1 file format to the XTC2 format requires access to two different environments (i.e. psana1 and psana2), the actual conversion requires running **two** `Task`s in parallel with a workflow. The two basic `Task`s are:

- `ReadXtc1`
- `WriteXtc2`

A working DAG/workflow, currently requires that the two **managed** `Task`s be given an equivalent number of ranks and are launched in parallel. The conversion process then proceeds in the following fashion:

1. `WriteXtc2` binds a socket on rank 0.
2. `WriteXtc2` publishes the port that was bound to the workflow manager `maestro`.
3. `ReadXtc1` queries `maestro` to discover which port to use.
4. **Every** rank from `ReadXtc1` then connects to a socket, transmitting data via ZMQ to `WriteXtc2`. Each rank tags its transmitted data so that the `WriteXtc2` processes can determine who will write it.
5. As only rank 0 for `WriteXtc2` has bound the socket, rank 0 internally redistributes the data using MPI IPC to the correct rank using the tag that was provided by `ReadXtc1`.


Ultimately, each rank of `WriteXtc2` will write a separate XTC2 **chunk** file (`-cXYZ` file). The XTC2 file format, or rather, more specifically, psana2 requires that the separate chunk files and the corresponding `.smd.xtc2` file be constructed as follows:

1. Rank 0 opens chunk file `-c000.xtc2` and the `-c000.smd.xtc2` file. There will only ever be **one** `.smd.xtc2` file, as required by psana2.
2. Rank 0 writes the initial transitions (Configure, BeginRun, etc.). It will also include any **calibration constants** as modified `SlowUpdate` transitions during this phase (`SlowUpdate` normally would contain EPICS data, but psana2 has been modified to understand these special transitions containing calibration constants).
3. All ranks then write their `-cXYZ.xtc2` files in parallel. The data is distributed chronologically: the `L1Accept` (event) data in `-c001.xtc2` comes directly after the `L1Accept` data in `-c000.xtc2`, and before that in `-c002.xtc2`.
4. As the ranks write their data, they record the timestamp, local offset of the datagram in their big data chunk file, and the size of the datagram they're writing. This will be used at the end to finalize the `.smd.xtc2` file.
5. When a rank has written all its corresponding event data, it then will either write a `Disable`, `EndStep`, `BeginStep` and `Enable` transition if it is a rank less than `(MPI_SIZE - 1)`. The `Enable` transition is special in that it will also include a special `chunkinfo` XTC which points to the next chunk file to read. E.g., `-c000.xtc2`'s final `Enable` transition will have a chunkinfo that includes a `chunkid = 1` and the filename for the `-c001.xtc2` file.
6. On the other hand, when `rank == (MPI_SIZE - 1)` completes, it instead writes the following transitions to complete the run: `Disable`, `EndStep`, `EndRun`.
7. Finally, all the ranks will send the recorded tuples of (`timestamp`, `offset`, `dgram_size`) to rank 0. Rank 0 will then write them out to the `.smd.xtc2` file. This is fast since it is small data.

## Command-line help

As a reminder, the `lute_help` command-line utility may be used to inspect the full set of command-line arguments. After sourcing the activation script upon building LUTE you will have the utility in your PATH.

```bash
# Assume the current working directory is the top of the LUTE repo, and the build
# script was run
> source install/bin/activate_installation
> lute_help -T <task>
```

For help on the conversion `Task`s, you can run:

```bash
> lute_help -T ReadXtc1
DEBUG:lute.execution.ipc:Preference of ZMQ usage not specified. Defaulting to using ZMQ.
INFO:lute_utilities.help.task_parameters:Fetching parameter information for ReadXtc1.
ReadXtc1
--------
Parameters for the xtc conversion Task.


Required Parameters:
--------------------
lute_config (AnalysisHeader)
	Unknown description.

xtc1_access_pattern (object)
	Provides information for how to access the data in XTC1. The top level keys will be used as the detector names in the XTC2 data.



All Parameters:
---------------
lute_config (AnalysisHeader)
	Unknown description.

eventfile (string) - Default: <Empty String> - May be populated by validator
	CSV file with event numbers. Otherwise will process all events.

nevents (integer)
	Optionally specify the number of events to use. If providing eventfile as well, that option will supercede this one.

xtc1_access_pattern (object)
	Provides information for how to access the data in XTC1. The top level keys will be used as the detector names in the XTC2 data.

Template Parameters:
--------------------
ConversionSpecification:
xtc2_attr_name (string)
	The name this field will have in the XTC2 file.

object_name (string)
	The name used to access the object in psana1. E.g. `epix10k2M` if you would create the object as `psana.Detector('epix10k2M')`

object_type (string)
	The psana1 object type used. E.g. `psana.Detector`

object_field_name ()
	The field (or fields) on the constructed psana1 object used to get the per event data. E.g. `'calib'` if using `det.calib(evt)`. Or `('get', 'ebeamPhotonEnergy')` if using `det.get(evt).ebeamPhotonEnergy()`


> lute_help -T WriteXtc2
DEBUG:lute.execution.ipc:Preference of ZMQ usage not specified. Defaulting to using ZMQ.
INFO:lute_utilities.help.task_parameters:Fetching parameter information for WriteXtc2.
WriteXtc2
---------
Parameters for the xtc conversion Task.


Required Parameters:
--------------------
lute_config (AnalysisHeader)
	Unknown description.

output_file (string)
	Where to write the output XTC2 file.

xtc1_access_pattern (object)
	Provides information for how to access the data in XTC1. The top level keys will be used as the detector names in the XTC2 data.



All Parameters:
---------------
lute_config (AnalysisHeader)
	Unknown description.

node_id (string) - Default: 1
	Node ID for the detector

output_file (string)
	Where to write the output XTC2 file.

xtc1_access_pattern (object)
	Provides information for how to access the data in XTC1. The top level keys will be used as the detector names in the XTC2 data.

Template Parameters:
--------------------
ConversionSpecification:
xtc2_attr_name (string)
	The name this field will have in the XTC2 file.

object_name (string)
	The name used to access the object in psana1. E.g. `epix10k2M` if you would create the object as `psana.Detector('epix10k2M')`

object_type (string)
	The psana1 object type used. E.g. `psana.Detector`

object_field_name ()
	The field (or fields) on the constructed psana1 object used to get the per event data. E.g. `'calib'` if using `det.calib(evt)`. Or `('get', 'ebeamPhotonEnergy')` if using `det.get(evt).ebeamPhotonEnergy()`


```

## Managed `Task`s
There are currently only 2 **managed** `Task`s that must be run in parallel. Each will run in the respective psana1 or psana2 environment to run their `Task`:

- `Xtc1Reader`
- `Xtc2Writer`

## DAG/Workflow

The following DAG shows an example setup for running the two `Task`s in parallel:

```yaml
!LUTE_DAG
- task_name: "Xtc1Reader"
  slurm_params: "--psana1 --nodes=1 --tasks-per-node=11 --partition=milano --account=lcls:<EXPERIMENT>"
  next: []
- task_name: "Xtc2Writer"
  slurm_params: "--nodes=1 --tasks-per-node=11 --partition=milano --account=lcls:<EXPERIMENT>"
  next: []
```

In this example, ten ranks will be used. (Remember: the `Executor` uses 1 core, so passing 11 to `ntasks-per-node` means 10 CPU cores will be available to the `Task`).

**NOTE:** It is currently a **REQUIREMENT** that the number of cores in each `Task`s request be identical! (11 in both cases, in this example)

## Configuration
A starting YAML for the conversion process may look like:

```yaml
ReadXtc1:               # All variables are given as strings
  # eventfile: ""
  nevents: 200          # Uncomment to do only some events - otherwise whole run.
  xtc1_access_pattern:
    jungfrau1M: # Name of the detector in the converted XTC2
    # You can have a list of attributes you will convert that will be stored in
    # this detector
      - xtc2_attr_name: "calib"          # Name of this attribute in xtc2
        object_name: "jungfrau1M" # Name of the detector in psana1
        object_type: "psana.Detector"    # Name of the object type in psana1
        object_field_name: "calib"       # Name of the per-event method to use in psana1
    lxt_fast:
      - xtc2_attr_name: "lxt_fast"
        object_name: "lxt_fast"
        object_type: "psana.Detector"
        object_field_name: "__call__"

WriteXtc2:                    # All variables are given as strings
  node_id: "1"                # Node ID for the detector
  output_file: "{{ XTC2_FILE_PATH }}/{{ XTC2_FILE_NAME }}"
  xtc1_access_pattern:
    jungfrau1M: # Name of the detector in the converted XTC2
    # You can have a list of attributes you will convert that will be stored in
    # this detector
      - xtc2_attr_name: "calib"          # Name of this attribute in xtc2
        object_name: "jungfrau1M" # Name of the detector in psana1
        object_type: "psana.Detector"    # Name of the object type in psana1
        object_field_name: "calib"       # Name of the per-event method to use in psana1
    lxt_fast:
      - xtc2_attr_name: "lxt_fast"
        object_name: "lxt_fast"
        object_type: "psana.Detector"
        object_field_name: "__call__"

```

For the most part, `node_id` can be left as is. We will cover the various other parameters in detail below. The most important being the `xtc1_access_pattern` covered at the end. This **MUST** be provided to **both** `Task`s in the current setup.

The documentation here assumes familiarity with both psana1 and psana2.

### Implicit parameters

Like many other `Task`s, this conversion `Task` will select data based on the global `experiment` and `run` provided. This are either provided explicitly in the header document of the YAML file, or are retrieved by environment variable when running from the ARP/eLog. Alternatively, they are passed on the command-line using `-e` and `-r` command-line arguments.

### Selecting events: `eventfile` and `nevents`

Under the configuration for `ReadXtc1` you can select how many events will be converted.

The default behaviour for the `Task` is to convert all the events in the XTC1 file(s) for the LCLS1 experiment to XTC2. You can optionally select a portion of these events using `eventfile` or `nevents`.

- `eventfile`: Is the path to a file that contains a list of **indices** for the events. This file should be a **CSV** file (comma-separated file).
- `nevents`: As an alternative, a specific number of events can instead be provided. This should be a single integer.

If both `eventfile` and `nevents` are provided, `eventfile` takes precedence.

### Output location: `output_file`

Under the `WriteXtc2` configuration, you can specify where to write the output.

A single large XTC2 file will be written by this `Task`, and this parameter determines where that file will be written.

**NOTE:** There are restrictions on the format of this path!

1. The filename must have a structure like: `<exp>-r<run:04d>-s000-c000.xtc2`
2. The path must include the standard psana2 search structure, namely the final portion must be: `<hutch>/<experiment>/xtc/<filename>`.

  - For example, a MFX experiment MFX12345, running on run 42, should have a filename and output path that looks like: `/rest_of_path/mfx/mfx12345/xtc/mfx12345-r0042-s000-c000.xtc2` . The rest of the path at the beginning is free to choose (as long as it is a valid path).

**NOTE:** The expectation is that you provide only the name of the file for chunk 0 (`-c000.xtc2`). The `Task` will then modify this base filename appropriately for each chunk file that it writes.

### Specifying what to convert and how: `xtc1_access_pattern`

**NOTE:** Currently, this information must be duplicated in the configuration for **BOTH** `ReadXtc1` and `WriteXtc2` as both require knowledge of it. A future update may allow for this information to be transmitted via `maestro` (as the port number is). However, that is not currently implemented, so the configuration should be written for one `Task` and copied to the other.

The main portion of the configuration is under this parameter. To understand how it works we will work from the provided example above:

```yaml
  xtc1_access_pattern:
    jungfrau1M: # Name of the detector in the converted XTC2
    # You can have a list of attributes you will convert that will be stored in
    # this detector
      - xtc2_attr_name: "calib"          # Name of this attribute in xtc2
        object_name: "jungfrau1M"        # Name of the detector in psana1
        object_type: "psana.Detector"    # Name of the object type in psana1
        object_field_name: "calib"       # Name of the per-event method to use in psana1
    lxt_fast:
      - xtc2_attr_name: "lxt_fast"
        object_name: "lxt_fast"
        object_type: "psana.Detector"
        object_field_name: "__call__"

```

The top level keys, `lxt_fast` and `jungfrau1M`, describe what you want the names of the detectors to be in the new converted XTC2 files. I.e. you will be able to access the data from psana2 using `run.Detector("jungfrau1M")` and `run.Detector("lxt_fast")`.

Under each detector you define a **list of dictionaries**. Each **dictionary** in the list describes one component from the XTC1 data that you will want to translate into XTC2, and the manner in which you will do the translation. The keys of the dictionary are:

1. `xtc2_attr_name`: This will be the name of the method used to access the data from **psana2**.
2. `object_name`: This is the name of the object in **psana1**
3. `object_type`: This is the type of object in **psana1**
4. `object_field_name`: This is the method (or methods) used to access the data in **psana1**.

So, taking the example for `jungfrau1M` above, we are translating a single method from XTC1 to XTC2. The specification says that we will take the `calib` method from a psana1 object `psana.Detector("jungfrau1M")` and attach it to a `calib` method for a psana2 object.

Alternatively, in pseudo-Python we have:

```python
# Take this psana1 data
object_type("object_name").object_field_name

# And translate it to:
jungfrau1M.xtc1dump.calib
```

**NOTE:** The converted data is always provided on a special "algorithm" called `xtc1dump`. So for the example above, you are creating the `calib` method for that algorithm. I.e.:

```python
# psana1 access of the form
det = psana.Detector("jungfrau1M")
data = det.calib(evt)

# is translated into psana2 access of the form
det = run.Detector("jungfrau1M")
data = det.xtc1dump.calib(evt)
```

The example for `lxt_fast` is analogous, however it uses a different method to access the data:

```python
# psana1 access of the form
det = psana.Detector("lxt_fast")
data = det(evt) # __call__ is identical to usage as a function

# Is now translated into psana2 access of the form
det = run.Detector("lxt_fast")
lxt_fast.xtc1dump.lxt_fast(evt)
```

A more involved case is exemplified by dealing with BLD. For example, the `EBeam` BLD could be accessed using the following YAML configuration:

```yaml
    EBeam:
      - xtc2_attr_name: "photon_energy"
        object_name: "EBeam"
        object_type: "psana.Detector"
        object_field_name: ["get","ebeamPhotonEnergy"]
```

In Python, this would correspond to the following psana1 access pattern:

```python
det = psana.Detector("EBeam")
data = det.get(evt).ebeamPhotonEnergy()
```

That is, the first item in the list of `object_field_name` is the method that takes the psana1 event object. The second object in the list is called without argument.
