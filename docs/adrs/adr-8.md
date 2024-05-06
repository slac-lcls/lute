# [ADR-8] Airflow credentials/authorization requires special launch program

**Date:** 2024-03-18

## Status
**Proposed**

## Context and Problem Statement
- Airflow is used as the workflow manager.
- Airflow does not currently support multi-tenancy, and LDAP is not currently supported for authentication.
- Multiple users will be expected to run the software and thus need to authenticate against the Airflow API.
  - We require a mechanism to control shared credentials for multiple users.
  - The credentials are admin credentials, so we do not want unconstrained access to them.
    - We want users to run workflows, for instance, but not to have free access to add and remove workflows.

## Decision
A closed-source `lute_launcher` program will be used to run the Airflow launch scripts. This program accesses credentials with the correct permissions. Users should otherwise not have access to the credentials. This will help ensure the credentials can be used by everyone but only to run workflows and not perform restricted admin activities.

### Decision Drivers
* Need shared access to credentials for the purpose of launching jobs.
* Restricted access to credentials for administrative activities.
* Ease of use for users
  * Authentication should be automatic - users can not be asked for passwords etc, for jobs that need to run automatically upon data acquisition

### Considered Options
* LDAP - this may be used in the future, but requires backend work outside of our control. We will revisit the implementation arising from this ADR in the future if LDAP is supported.
*

## Consequences
* Complexity

## Compliance


## Metadata
- This ADR WILL be revisited during the post-mortem of the first prototype.
- Compliance section will be updated as prototype evolves.
