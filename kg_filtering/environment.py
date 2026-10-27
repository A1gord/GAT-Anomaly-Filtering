import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def runtime_descriptor():
    packages = {}
    for distribution in ('torch', 'torch-geometric', 'scikit-learn'):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {'python': sys.version, 'implementation': platform.python_implementation(),
        'platform': platform.platform(), 'machine': platform.machine(), 'packages': packages}
