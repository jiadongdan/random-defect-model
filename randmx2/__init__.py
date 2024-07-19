# Version should always be readily available.
__version__ = '0.1.0'

# Lazy loading for sub-packages.
class _LazyLoader:
    def __init__(self, package_name):
        self._package_name = package_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = __import__(self._package_name, globals(), locals(), ['*'])
        return self._module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __dir__(self):
        module = self._load()
        return dir(module)

# Setup lazy loading for sub-packages.
random = _LazyLoader('randmx2.random')

# Explicit imports for frequently used functions or classes
# These are assumed to be lightweight and commonly used enough to justify immediate loading.
from randmx2.random.random_defect_model import VacRandomModel   # Assuming this is lightweight
from randmx2.rdf._rdf import get_rdf
from randmx2.rdf._rdf import get_rdf_ij
from randmx2.rdf._rdf import get_rdf_cnts

__all__ = [
    'random',
    'VacRandomModel',
    'get_rdf',
    'get_rdf_ij',
    'get_rdf_cnts',
]