from .CUB200 import CUB200
from .Car196 import Car196
from .Products import Products
from .In_shop_clothes import InShopClothes
from .JD_Fashion import JD_Fashion
# from .transforms import *
__factory = {
    'cub': CUB200,
    'car': Car196,
    'product': Products,
    'shop': InShopClothes,
    'jd': JD_Fashion,
}


def names():
    return sorted(__factory.keys())


def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)
