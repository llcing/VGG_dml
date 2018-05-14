from __future__ import print_function, absolute_import

from .SoftmaxNeigLoss import SoftmaxNeigLoss
from .NCA import NCA 
from .NeighbourLoss import NeighbourLoss
from .triplet import Triplet
from .CenterTriplet import CenterTripletLoss
from .GaussianMetric import GaussianMetricLoss
from .HistogramLoss import HistogramLoss
from .NeighbourLoss import NeighbourLoss
from .DistanceMatchLoss import DistanceMatchLoss
from .NeighbourHardLoss import NeighbourHardLoss
from .DistWeightLoss import DistWeightLoss
from .BinDevianceLoss import BinDevianceLoss
from .BinBranchLoss import BinBranchLoss
from .MarginDevianceLoss import MarginDevianceLoss
from .MarginPositiveLoss import MarginPositiveLoss
from .ContrastiveLoss import ContrastiveLoss
from .DistWeightContrastiveLoss import DistWeightContrastiveLoss
from .DistWeightDevianceLoss import DistWeightBinDevianceLoss
from .DistWeightDevBranchLoss import DistWeightDevBranchLoss
from .DistWeightNeighbourLoss import DistWeightNeighbourLoss
from .BDWNeighbourLoss import BDWNeighbourLoss
from .EnsembleDWNeighbourLoss import EnsembleDWNeighbourLoss
# from .BranchKNNSoftmax import BranchKNNSoftmax
# from .JSDivKNNSoftmaxLoss import JSDivKNNSoftmaxLoss
from .A_triplet import ATriplet
from .Batchall import BatchAll
from .ABatchall import ABatchAll
from .A_hard_pair import AHardPair
from .Grad_NCA import Grad_NCA
from .CenterNCALoss import CenterNCALoss
from .ClusterNCALoss import ClusterNCALoss
from .MCALoss import MCALoss

__factory = {
    'softneig': SoftmaxNeigLoss,
    'nca': NCA,
    'neighbour': NeighbourLoss,
    'histogram': HistogramLoss,
    'gaussian': GaussianMetricLoss,
    'bin': BinDevianceLoss,
    'binbranch': BinBranchLoss,
    'margin': MarginDevianceLoss,
    'positive': MarginPositiveLoss,
    'con': ContrastiveLoss,
    'distweight': DistWeightLoss,
    'distance_match': DistanceMatchLoss,
    'dwcon': DistWeightContrastiveLoss,
    'dwdev': DistWeightBinDevianceLoss,
    'dwneig': DistWeightNeighbourLoss,
    'dwdevbranch': DistWeightDevBranchLoss,
    'bdwneig': BDWNeighbourLoss,
    'edwneig': EnsembleDWNeighbourLoss,
  #  'branchKS': BranchKNNSoftmax,
  #  'JSDivKS': JSDivKNNSoftmaxLoss,
    'triplet': Triplet,
    'Atriplet': ATriplet,
    'batchall': BatchAll,
    'Abatchall': ABatchAll,
    'Ahardpair': AHardPair,
    'Grad_nca': Grad_NCA,
    'center-nca': CenterNCALoss,
    'cluster-nca': ClusterNCALoss,
    'mca': MCALoss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)
