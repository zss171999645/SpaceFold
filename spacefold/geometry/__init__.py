"""SpaceFold geometry codec and multiresolution distance fields."""

from importlib import import_module

_MODULES = {
    "TreeBatch": ".model",
    "UnifiedOctreeVAE": ".model",
    "UnifiedTUDF": ".octree",
    "load_geometry_checkpoint": ".checkpoint",
    "register_volfill_source": "._source",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(_MODULES[name], __name__), name)
