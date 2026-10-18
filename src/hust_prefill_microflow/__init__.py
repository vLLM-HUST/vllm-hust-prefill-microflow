"""Independent Prefill Microflow ECPA bundle; import has no runtime effects."""

__version__ = "0.1.0.dev0"


def register():
    """Selected by ECPA through vLLM's general-plugin entry point."""
    from .activation import register as activate

    return activate()
