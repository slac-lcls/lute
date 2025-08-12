import os


class DatabaseError(Exception):
    """General LUTE database error."""

    ...


LUTE_DB_CURRENT_SPEC_VERSION: int = 0x000002
LUTE_DB_DEFAULT_SPEC_VERSION: int = 0x000001

LUTE_DB_SPEC_VERSION: int = int(
    os.getenv("LUTE_DB_SPEC_VERSION", LUTE_DB_DEFAULT_SPEC_VERSION)
)

if LUTE_DB_SPEC_VERSION == 0x000001:
    from lute.io.db.v1.api import *  # noqa: F403
elif LUTE_DB_SPEC_VERSION == 0x000002:
    from lute.io.db.v2.api import *  # noqa: F403
else:
    raise DatabaseError(
        "Unrecognized database specification version! Set LUTE_DB_SPEC_VERSION appropriately! "
        "Supported versions: 0x000001 and 0x000002"
    )
