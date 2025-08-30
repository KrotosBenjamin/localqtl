import importlib.metadata
from .cis import map_cis, map_nominal, map_independent
from .haplotypeio import RFMixReader
from .genotypeio import PlinkReader
from .pgen import PgenReader
from .post import (
    calculate_afc,
    annotate_genes,
    calculate_qvalues,
    get_significant_pairs
)

__version__ = importlib.metadata.version(__name__)
