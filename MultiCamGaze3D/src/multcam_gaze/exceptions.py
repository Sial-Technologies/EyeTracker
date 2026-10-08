"""Domain exceptions."""


class MultiCamGazeError(Exception):
    """Base error."""


class MissingIntrinsicsError(MultiCamGazeError):
    """Required camera intrinsics file is absent."""


class PoseEstimateFailed(MultiCamGazeError):
    """Could not estimate pose from image."""


class ConfigValidationError(MultiCamGazeError):
    """Invalid configuration JSON."""


class CalibrationSchemaError(MultiCamGazeError):
    """Calibration file schema mismatch."""
