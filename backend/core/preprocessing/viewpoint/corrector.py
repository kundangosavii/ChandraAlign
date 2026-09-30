import numpy as np


class ViewpointCorrector:

    def __init__(
        self,
        lowes_ratio: float = 0.75,
        mutual: bool = False,
    ):
        pass

    def process(
        self,
        reference: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        return source.copy()


def viewpoint_correction(
    reference: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:

    corrector = ViewpointCorrector()

    return corrector.process(
        reference,
        source,
    )