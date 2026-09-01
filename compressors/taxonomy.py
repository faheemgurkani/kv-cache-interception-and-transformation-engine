"""Compression method taxonomy (Phase 4).

Classifies KV transformations by mechanism rather than treating every method as
generic "KV compression".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompressionCategory(str, Enum):
    """Primary compression mechanism categories from RESEARCH_REDESIGN_PLAN Phase 4."""

    EVICTION = "eviction"  # A — drop tokens; retained tokens stay dense
    QUANTIZATION = "quantization"  # B — lower bitwidth on tensor elements
    PROJECTION = "projection"  # C — dimensionality reduction / low-rank latent cache
    HYBRID = "hybrid"  # D — combine eviction + other transforms
    MODIFIED_ATTENTION = "modified_attention"  # E — attention kernel changes


@dataclass(frozen=True)
class MethodTaxonomy:
    """Taxonomy metadata for one compression plug-in."""

    name: str
    primary: CompressionCategory
    secondary: tuple[CompressionCategory, ...] = ()
    description: str = ""
    modifies_attention: bool = False
    calibration_free: bool = True
    compression_unit: str = "tensor"  # token | head | layer | tensor

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "primary": self.primary.value,
            "secondary": [item.value for item in self.secondary],
            "description": self.description,
            "modifies_attention": self.modifies_attention,
            "calibration_free": self.calibration_free,
            "compression_unit": self.compression_unit,
        }


METHOD_TAXONOMY: dict[str, MethodTaxonomy] = {
    "identity": MethodTaxonomy(
        name="identity",
        primary=CompressionCategory.QUANTIZATION,
        description="Uncompressed FP16 baseline (no transformation).",
        compression_unit="tensor",
    ),
    "turboquant": MethodTaxonomy(
        name="turboquant",
        primary=CompressionCategory.QUANTIZATION,
        description="WHT + Lloyd-Max vector quantization with optional QJL residual on values.",
        calibration_free=False,
        compression_unit="tensor",
    ),
    "qjl": MethodTaxonomy(
        name="qjl",
        primary=CompressionCategory.QUANTIZATION,
        secondary=(CompressionCategory.MODIFIED_ATTENTION,),
        description="1-bit sign sketch on keys; asymmetric ProdQJL attention estimator.",
        modifies_attention=True,
        compression_unit="head",
    ),
    "rocketkv": MethodTaxonomy(
        name="rocketkv",
        primary=CompressionCategory.HYBRID,
        secondary=(CompressionCategory.EVICTION, CompressionCategory.MODIFIED_ATTENTION),
        description="SnapKV-style permanent filter + hybrid sparse attention (HSA).",
        modifies_attention=True,
        compression_unit="token",
    ),
    "snapkv": MethodTaxonomy(
        name="snapkv",
        primary=CompressionCategory.EVICTION,
        description="Prefill-only observation-window voting + pooled top-k token eviction.",
        compression_unit="head",
    ),
    "palu": MethodTaxonomy(
        name="palu",
        primary=CompressionCategory.PROJECTION,
        secondary=(CompressionCategory.MODIFIED_ATTENTION,),
        description="G-LRD low-rank latent KV cache; dynamic K reconstruction under RoPE.",
        modifies_attention=True,
        calibration_free=False,
        compression_unit="head",
    ),
    "kivi": MethodTaxonomy(
        name="kivi",
        primary=CompressionCategory.QUANTIZATION,
        description="Asymmetric INT2 KV quantization (stub).",
        compression_unit="tensor",
    ),
}


def get_method_taxonomy(compressor_name: str) -> MethodTaxonomy | None:
    return METHOD_TAXONOMY.get(compressor_name)


# KIVI is registered but unimplemented — excluded from live eval / taxonomy smokes.
STUB_METHODS: frozenset[str] = frozenset({"kivi"})


def active_eval_methods() -> tuple[str, ...]:
    """Plug-ins that are implemented and should appear in taxonomy-coverage smokes."""
    return tuple(name for name in METHOD_TAXONOMY if name not in STUB_METHODS)


def taxonomy_categories_covered(method_names: list[str] | tuple[str, ...]) -> set[CompressionCategory]:
    covered: set[CompressionCategory] = set()
    for name in method_names:
        meta = METHOD_TAXONOMY.get(name)
        if meta is None:
            continue
        covered.add(meta.primary)
        covered.update(meta.secondary)
    return covered
