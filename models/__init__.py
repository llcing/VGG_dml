from __future__ import print_function, absolute_import
from .BN_Inception import BNInception
from .VGG import vgg16_bn

__factory = {
    'bn': BNInception,
    'vgg': vgg16_bn,
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
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
