# [ADR-6] Third-party Config Files Managed by Templates Rendered by `ThirdPartyTask`s

**Date:** 2024-02-12

## Status
**Proposed**

## Context and Problem Statement
- While many third-party executables of interest to the LUTE platform can be fully configured via command-line arguments, some also require management of an additional config file.
- Config files may use a variety of languages and methods. E.g. YAML, TOML, JSON, or even direct management of Python scripts.
  - From the perspective of a generic interface to manage these files this poses a challenge.
- Ideally all aspects of configuraiton could be managed from the single LUTE configuration file.

## Decision
Templates will be used for the third party configuration files. A generic interface to heterogenous templates will be provided through a combination of pydantic models and the `ThirdPartyTask` implementation. The pydantic models will label extra arguments to `ThirdPartyTask`s as being `TemplateParameters`. I.e. any extra parameters are considered to be for a templated configuration file. The `ThirdPartyTask` will find the necessary template and render it if any extra parameters are found. This puts the burden of correct parsing on the template definition itself.

### Decision Drivers
* Need to be able to configure the necessary files from within the LUTE framework.
* Configuration files take many forms so need a generic interface to disparate file types.
* Want to maintain as simple a `Task` interface as possible - but due to the above, need a way of handling multiple output files.
  * Text substiution provides a means to do this.

### Considered Options
* Separate configuration `Task` to be run before the main `ThirdPartyTask`.
* Generate the configuration file in its entirety from within the `Task`.
  * This removes the simplicity in allowing all `ThirdPartyTask`s to be run as instances of a single class.

## Consequences
* Can configure and run third party tasks which require the use of a configuration file.
* Must manage templates in addition to the standard configuration parsing code.
  * The templates themselves provide the specific "programming" for filling them in. I.e. the Python interface assumes that the template will properly handle the block of parameters it is sent.
  * Due to the above, template errors can be fatal, and appropriate attention to template creation is necessary.
* Allowing for template parameters in the general configuration file requires accepting the possiblity of extra parameters not defined in the data validation (pydantic) models.
  * Extra parameters are not validated in the same way as standard parameters.
  * We have to assume the template will properly deal with them.

## Compliance


## Metadata
- This ADR WILL be revisited during the post-mortem of the first prototype.
- Compliance section will be updated as prototype evolves.
