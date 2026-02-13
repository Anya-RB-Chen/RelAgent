"""
Default paths and configuration for the RelAgent baseline.
"""
import os

# Project root: parent of relagent/ package
_RELAGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_RELAGENT_DIR)

DATA_DIR = os.environ.get("RELAGENT_DATA_DIR", os.path.join(_ROOT, "data"))
MOLGROUND_DIR = os.path.join(DATA_DIR, "molground")
MOLGENIE_DIR = os.path.join(DATA_DIR, "molgenie")
MOLONTO_CACHE_DIR = os.path.join(DATA_DIR, "molonto")

DEFAULT_TEST_PATH = os.path.join(MOLGROUND_DIR, "molground_test.json")
DEFAULT_VAL_PATH = os.path.join(MOLGROUND_DIR, "molground_val.json")

OUTPUT_DIR = os.environ.get("RELAGENT_OUTPUT_DIR", os.path.join(_ROOT, "outputs"))
