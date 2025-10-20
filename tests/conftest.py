"""Pytest configuration for histoslice tests."""

# Pre-import scipy to avoid subprocess interference issues
# This ensures scipy is loaded in the parent process before any subprocess tests run
try:
    import scipy  # noqa: F401
    import scipy.cluster  # noqa: F401
except ImportError:
    pass

# Pre-import matplotlib backend to avoid subprocess interference
try:
    import matplotlib  # noqa: F401
    import matplotlib.backends.backend_agg  # noqa: F401
except ImportError:
    pass
