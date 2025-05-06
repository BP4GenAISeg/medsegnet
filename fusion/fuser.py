from .weighted_softmax import WeightedSoftmaxFusion
from .weighted_majority import WeightedMajorityFusion
from .only_final import OnlyFinalFusion


class OutputFuser:
    """
    Handles the fusion of multiple model outputs (logits) from different resolutions.
    """

    VALID_MODES = {
        "weighted_softmax": WeightedSoftmaxFusion,
        "weighted_majority": WeightedMajorityFusion,
        "no_fuse_only_final": OnlyFinalFusion,
    }

    def __init__(self, mode: str, weights=None):
        _mode_lower = mode.lower()
        if _mode_lower not in self.VALID_MODES:
            raise ValueError(
                f"Invalid fusion mode '{mode}'. Valid modes are: {list(self.VALID_MODES.keys())}"
            )

        if _mode_lower == "only_final":
            self.strategy = self.VALID_MODES[_mode_lower]()
        else:
            self.strategy = self.VALID_MODES[_mode_lower](weights)

    def fuse(self, outputs):
        return self.strategy.fuse(outputs)

    def __call__(self, outputs):
        return self.fuse(outputs)
