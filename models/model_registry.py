from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """Registry for magnetic model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""
