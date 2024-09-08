# [ADR-10] Tasklet functions provide supplemental/auxiliary analysis for `ThirdPartyTask`s

**Date:** 2024-04-15

## Status
**Proposed**

## Context and Problem Statement
- Often need to extract some information for presentation after running some third-party analysis code.
- There is non-zero over-head associated with running a `Task`.
- Complexity of a `Task` is often overkill for simple operations such as parsing a text file, concatenating files or sending some small amoumnt of information to the eLog.

## Decision
The `Executor` will provide a mechanism to run some small functions either before or after the main `Task` has executed. These "tasklets" are Python functions which perform an action and optionally return a value to the `Executor`. Depending on the type of the returned value (if any), the `Executor` may take a number of actions, e.g. posting a message, creating a plot, moving files.

### Decision Drivers
* Need to provide a mechanism to summarize information as `Task`s complete. Just running the `Task` provides no feedback to the user on its own.
* Want to present information to the eLog as an option.
* Some functions need to be run before the `Task` starts, and others after it has finished.
* Want a mechanism to change which functions are run.

### Considered Options
* Any operation at all must be a fully-implemented `Task`.
  * There is overhead to launching a new batch job just to run a single function.
* Additional options in the configuration of the pydantic model for a third-party `Task` to include auxiliary functions to run.
  * This doesn't easily allow the function to be run after the `Task` has completed.

## Consequences
* Additional complexity for configuring `Executor`.
  * This adds additional knobs to turn, points of failure, and makes on boarding new developers more challenging.
* Adding new tasklets is straightforward.
* Tasklets can be used in first-party `Task` code as they are just Python functions.

## Compliance


## Metadata
- This ADR WILL be revisited during the post-mortem of the first prototype.
- Compliance section will be updated as prototype evolves.
