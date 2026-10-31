from numbers import Real

FEATURE_LIMITS = {
    "N": (0, 300),
    "P": (0, 300),
    "K": (0, 300),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 5000),
}

def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    cleaned = {}
    missing = [name for name in FEATURE_LIMITS if name not in payload]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    for name, (low, high) in FEATURE_LIMITS.items():
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a numeric value.")
        value = float(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
        cleaned[name] = value
    return cleaned
