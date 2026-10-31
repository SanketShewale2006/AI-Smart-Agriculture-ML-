import pytest
from src.validation import validate_payload

VALID = {
    "N": 90, "P": 42, "K": 43, "temperature": 24.5,
    "humidity": 80, "ph": 6.5, "rainfall": 200
}

def test_valid_payload_is_normalized():
    result = validate_payload(VALID)
    assert result["N"] == 90.0
    assert result["ph"] == 6.5

def test_missing_field_is_rejected():
    with pytest.raises(ValueError, match="Missing"):
        validate_payload({"N": 90})

def test_out_of_range_value_is_rejected():
    bad = {**VALID, "humidity": 101}
    with pytest.raises(ValueError, match="humidity"):
        validate_payload(bad)

def test_string_value_is_rejected():
    bad = {**VALID, "N": "a lot"}
    with pytest.raises(ValueError, match="numeric"):
        validate_payload(bad)
