# [ADR-7] `Task` Configuration is Stored in a Database Managed by `Executor`s

**Date:** 2024-02-12

## Status
**Proposed**

## Context and Problem Statement
- For metadata publishing reasons, need a mechanism to maintain a history of `Task` parameter configurations.
- Each `Task`'s code is designed to be independent of other `Task`'s aside from code shared by inheritance.
  - Dependencies between `Task`s are intended to be defined only at the level of workflows.
  - Nonetheless, some `Task`s may have implicit dependencies on others. E.g. one `Task` may use the output files of another, and so could benefit from having knowledge of where they were written.

## Decision
Upon `Task` completion the managing `Executor` will write the `AnalysisConfig` object, including `TaskParameters`, results and generic configuration information to a database. Some entries from this database can be retrieved to provide default files for `TaskParameter` fields; however, the `Task` itself has no knowledge, and does not access to the database.

### Decision Drivers
* Want to reduce explicit dependencies between `Task`s while allowing information to be shared between them.
* Have `Task`-independent IO be managed solely at the `Executor` level.

### Considered Options
* `Task`s write the database.
* `Task`s pass information through other mechanisms, such as Airflow.

## Consequences
* Requires a database.
  * Additional dependency, although at least one backend can be the standard `sqlite` which should make everything transferrable.
* Allows for information to be passed between `Task`s without any explicit code dependencies/linkages between them.
  * The dependency is still mostly determined by the workflow definition. Default values can be provided by the database if needed.

## Compliance


## Metadata
- This ADR WILL be revisited during the post-mortem of the first prototype.
- Compliance section will be updated as prototype evolves.
