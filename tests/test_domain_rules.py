from pathlib import Path

from knowmat.domain_rules import DomainRules, default_rules


def test_default_rules_loaded_from_yaml():
    # Basic smoke test: default_rules should have non-empty core fields
    assert default_rules.valid_elements
    assert isinstance(next(iter(default_rules.valid_elements)), str)
    assert isinstance(default_rules.phase_patterns, dict)
    assert isinstance(default_rules.precipitate_keywords, list)


def test_domain_rules_from_yaml_custom_path(tmp_path: Path):
    yaml_content = """
valid_elements:
  - Ti
  - Nb
phase_patterns:
  bcc: BCC
precipitate_keywords:
  - sigma
property_name_mapping:
  hardness: Hardness
process_category_keywords:
  AM_DED:
    - directed energy deposition
parameter_patterns:
  Laser_Power_W:
    - "(\\\\d+\\\\.\\\\d+) W"
"""
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(yaml_content, encoding="utf-8")

    rules = DomainRules.from_yaml(rules_path)

    assert rules.valid_elements == {"Ti", "Nb"}
    assert rules.phase_patterns.get("bcc") == "BCC"
    assert "sigma" in rules.precipitate_keywords
    assert rules.property_name_mapping.get("hardness") == "Hardness"
    assert "AM_DED" in rules.process_category_keywords
    assert "Laser_Power_W" in rules.parameter_patterns
    # parameter patterns should be compiled
    assert "Laser_Power_W" in rules._compiled_param_patterns
