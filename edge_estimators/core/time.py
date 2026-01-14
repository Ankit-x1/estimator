"""Time handling: dt, timestamp, async logic."""


def compute_dt(t_current: float, t_previous: float) -> float:
    """
    Compute time delta between timestamps.
    
    Args:
        t_current: Current timestamp
        t_previous: Previous timestamp
    
    Returns:
        Time delta in seconds
    """
    dt = t_current - t_previous
    if dt < 0:
        raise ValueError(f"Negative dt: {dt} (t_current={t_current}, t_previous={t_previous})")
    return dt


def clamp_dt(dt: float, dt_min: float = 1e-6, dt_max: float = 1.0) -> float:
    """
    Clamp dt to reasonable bounds to prevent numerical issues.
    
    Args:
        dt: Time delta
        dt_min: Minimum allowed dt (default: 1e-6 seconds)
        dt_max: Maximum allowed dt (default: 1.0 seconds)
    
    Returns:
        Clamped dt
    """
    if dt < dt_min:
        return dt_min
    if dt > dt_max:
        return dt_max
    return dt

