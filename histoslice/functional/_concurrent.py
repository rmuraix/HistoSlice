# Use spawn to avoid fork-related hangs with libvips/pyvips on Python 3.10/3.11.
# In Python 3.12+, fork is deprecated anyway.
DEFAULT_START_METHOD = "spawn"
