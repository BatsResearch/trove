"""Manage global library configuration."""

import os
from contextlib import contextmanager
from uuid import uuid4

from datasets.fingerprint import Hasher

## Defaults

LOG_LEVEL = "INFO"
# disable rich log handler
TROVE_LOGGING_DISABLE_RICH = False
# disable logging module
TROVE_LOGGING_DISABLE = False

# incomplete files are written to this directory (from file_utils.atomic_write function)
TROVE_INCOMPLETE_FILE_DIRNAME = "trove_incomplete_files"

# root directory where all cache files are saved.
# we pass this value to 'hf_hub.cached_assets_path(assets_dir=config.TROVE_CACHE_DIR)'
# If you set this to None, the cache dir is determined by hf_hub library
# (current it gives us a subdir in 'HUB_CACHE/assets')
TROVE_CACHE_DIR = None

# Create a unique ID for this specific process
# You can use it for example to create a temporary directory for
# all temporary files in this process
hasher = Hasher()
hasher.update(os.getpid())
hasher.update(uuid4().hex)
PROCESS_UUID = hasher.hexdigest()
del hasher

## Update default values if necessary

if os.getenv("TROVE_LOG_LEVEL", None) is not None:
    LOG_LEVEL = os.environ["TROVE_LOG_LEVEL"]

if os.getenv("TROVE_LOGGING_DISABLE_RICH", None) is not None:
    TROVE_LOGGING_DISABLE_RICH = (
        os.environ["TROVE_LOGGING_DISABLE_RICH"].lower() == "true"
    )

if (
    os.getenv("TROVE_CACHE_DIR", None) is not None
    and os.environ["TROVE_CACHE_DIR"].strip() != ""
):
    TROVE_CACHE_DIR = os.environ["TROVE_CACHE_DIR"]


def disable_logging() -> None:
    global TROVE_LOGGING_DISABLE
    TROVE_LOGGING_DISABLE = True


def enable_logging() -> None:
    global TROVE_LOGGING_DISABLE
    TROVE_LOGGING_DISABLE = False


@contextmanager
def logging_context(disable: bool = False):
    global TROVE_LOGGING_DISABLE
    orig_value = TROVE_LOGGING_DISABLE
    try:
        TROVE_LOGGING_DISABLE = disable
        yield
    finally:
        TROVE_LOGGING_DISABLE = orig_value
