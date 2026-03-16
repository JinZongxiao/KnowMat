import math

from knowmat.schema_converter import SchemaConverter


converter = SchemaConverter()


def test_parse_temperature_to_k_from_at_k():
    assert converter.parse_temperature_to_k("measured at 298 K in air") == 298
    assert converter.parse_temperature_to_k("AT 873 K; compression") == 873


def test_parse_temperature_to_k_from_celsius():
    # 25 °C ≈ 298.15 K
    assert converter.parse_temperature_to_k("tested at 25 °C") == 298
    # plain 'c' should still be treated as Celsius when clearly temperature
    assert converter.parse_temperature_to_k("at 100 c in vacuum") == 373


def test_parse_temperature_to_k_from_k_without_at():
    assert converter.parse_temperature_to_k("873K tensile test") == 873


def test_parse_temperature_to_k_returns_none_when_no_temperature():
    assert converter.parse_temperature_to_k("room temperature test") is None


def test_validate_composition_json_warns_on_sum_far_from_100():
    comp = {"Ti": 30.0, "Nb": 30.0, "Zr": 30.0}
    cleaned, warnings = converter.validate_composition_json(comp, "Ti30Nb30Zr30")
    assert cleaned == comp
    assert any("Composition sum" in w for w in warnings)


def test_validate_composition_json_drops_invalid_element():
    comp = {"Ti": 50.0, "Xx": 50.0}
    cleaned, warnings = converter.validate_composition_json(comp, "Ti50Xx50")
    assert "Xx" not in cleaned
    assert any("Invalid element" in w for w in warnings)

