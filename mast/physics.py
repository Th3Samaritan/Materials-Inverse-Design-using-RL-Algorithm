"""Small physics primitives for validation/retraining research, not fitted plant models."""
import numpy as np


def martensite_fraction(ms_c, final_temperature_c, available_austenite_fraction, coefficient_per_c=.011):
    """Koistinen-Marburger form applied only to the available austenite.

    Austenite availability must come from a separately validated diffusional
    transformation model. This function does not infer it from a quench label.
    """
    ms, end, available = np.broadcast_arrays(ms_c, final_temperature_c, available_austenite_fraction)
    if not all(np.isfinite(x).all() for x in (ms, end, available)) or not np.isfinite(coefficient_per_c) or coefficient_per_c <= 0:
        raise ValueError('Inputs and coefficient must be finite; coefficient must be positive.')
    if ((available < 0) | (available > 1)).any():
        raise ValueError('Available austenite fraction must lie in [0, 1].')
    return available * (-np.expm1(-coefficient_per_c * np.maximum(ms-end, 0)))
