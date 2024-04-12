# Architecture Decision Records
- This directory contains a list of architecture and major feature decisions.
- Please refer to the `madr_template.md` for creating new ADRs. This template was adapted from the [MADR template](https://adr.github.io/madr/) (MIT License).
- A table of ADRs is provided below.

| ADR No. | Record Date | Title                                                                       | Status       |
|:-------:|:-----------:|:----------------------------------------------------------------------------|:------------:|
| 1       | 2023-11-06  | All analysis `Task`s inherit from a base class                              | **Accepted** |
| 2       | 2023-11-06  | Analysis `Task` submission and communication is performed via `Executor`s   | **Accepted** |
| 3       | 2023-11-06  | `Executor`s will run all `Task`s via subprocess                             | **Proposed** |
| 4       | 2023-11-06  | Airflow `Operator`s and LUTE `Executor`s are separate entities.             | **Proposed** |
| 5       | 2023-12-06  | Task-Executor IPC is Managed by Communicator Objects                        | **Proposed** |
| 6       | 2024-02-12  | Third-party Config Files Managed by Templates Rendered by `ThirdPartyTask`s | **Proposed** |
| 7       | 2024-02-12  | `Task` Configuration is Stored in a Database Managed by `Executor`s         | **Proposed** |
|         |             |                                                                             |              |
