"""judge-reliability-kit - decide whether an LLM judge panel can be trusted."""

from .agreement import (
    cohen_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
    pairwise_kappas,
    percent_agreement,
)
from .decompose import (
    AMBIGUOUS,
    CLEAN,
    UNDERSPECIFIED,
    UNSTABLE,
    Decomposition,
    ItemVerdict,
    decompose,
)
from .probes import ProbeResult, position_bias, self_enhancement_bias, verbosity_bias
from .report import markdown_report

__version__ = "0.1.0"

__all__ = [
    "AMBIGUOUS",
    "CLEAN",
    "UNDERSPECIFIED",
    "UNSTABLE",
    "Decomposition",
    "ItemVerdict",
    "ProbeResult",
    "__version__",
    "cohen_kappa",
    "decompose",
    "fleiss_kappa",
    "interpret_kappa",
    "krippendorff_alpha",
    "markdown_report",
    "pairwise_kappas",
    "percent_agreement",
    "position_bias",
    "self_enhancement_bias",
    "verbosity_bias",
]
