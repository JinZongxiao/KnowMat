"""
Schema converter for KnowMat 2.0.

Transforms the internal LLM extraction format into the target HEA dataset
schema.  All domain heuristics (phase inference, precipitate detection,
property-name normalisation, process-category classification, etc.) are
driven by :class:`~knowmat.domain_rules.DomainRules` so that materials
scientists can adjust behaviour via ``domain_rules.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from knowmat.domain_rules import DomainRules, default_rules
from knowmat.states import TargetSchema

logger = logging.getLogger(__name__)


class SchemaConverter:
    """Convert LLM-extracted data to the target HEA dataset schema."""

    def __init__(self, rules: DomainRules | None = None) -> None:
        self.rules = rules or default_rules

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        data: dict,
        source_path: str,
        paper_text: Optional[str] = None,
        document_metadata: Optional[dict] = None,
    ) -> TargetSchema:
        """Convert internal extraction dict to target HEA schema.

        Mirrors the original ``_to_target_schema`` logic with 3-tier DOI
        priority, Key_Params_JSON population, dedicated microstructure text,
        grain-size backfill, and build-orientation encoding.
        """
        if not isinstance(data, dict):
            return {"Dataset_Description": "High Entropy Alloy Data Extraction Template", "Materials": []}
        if "Materials" in data and "Dataset_Description" in data:
            return data  # type: ignore[return-value]

        compositions = data.get("compositions", []) or []
        source_name = os.path.basename(source_path)

        ocr_doi = (document_metadata or {}).get("doi") if document_metadata else None
        regex_doi = self.extract_first_doi(paper_text)
        global_doi = ocr_doi or regex_doi or ""

        target_compositions = [c for c in compositions if c.get("role", "Target") == "Target"]

        groups: Dict[str, list] = defaultdict(list)
        for comp in target_compositions:
            comp_raw = comp.get("composition", "") or ""
            formula_norm = comp.get("composition_normalized")
            if not formula_norm:
                formula_norm = comp_raw.replace(" ", "")
            base_formula = re.sub(r"\s*\[.*?\]\s*$", "", formula_norm).strip()
            if not base_formula:
                base_formula = formula_norm
            groups[base_formula].append(comp)

        materials: List[Dict[str, Any]] = []

        for mat_idx, (base_formula, comps) in enumerate(sorted(groups.items()), start=1):
            first_comp = comps[0]
            comp_raw_first = first_comp.get("composition", "") or ""
            material_doi = first_comp.get("source_doi") or global_doi

            samples: List[Dict[str, Any]] = []
            for s_idx, comp in enumerate(comps, start=1):
                samples.append(self._build_sample(comp, mat_idx, s_idx))

            raw_comp_json = self.build_composition_json(base_formula)
            validated_comp_json, comp_warnings = self.validate_composition_json(
                raw_comp_json, base_formula
            )
            for warning in comp_warnings:
                logger.warning("[COMPOSITION] %s", warning)

            material = {
                "description": f"--- Material {mat_idx}: {base_formula} ---",
                "Mat_ID": f"M{mat_idx:03d}",
                "Alloy_Name_Raw": comp_raw_first,
                "Formula_Normalized": base_formula,
                "Composition_JSON": validated_comp_json,
                "Source_DOI": material_doi or "",
                "Source_File": source_name,
                "Processed_Samples": samples,
            }
            materials.append(material)

        return {
            "Dataset_Description": "High Entropy Alloy Data Extraction Template",
            "schema_version": "2.2",
            "pipeline_version": "knowmat-2.1.0",
            "Materials": materials,
        }

    # ------------------------------------------------------------------
    # Helpers – each is independently testable
    # ------------------------------------------------------------------

    @staticmethod
    def parse_temperature_to_k(measurement_condition: Optional[str]) -> Optional[int]:
        """Best-effort parse of temperature from measurement condition text.

        Prioritises ``at XXX K`` format, then general patterns.
        Returns integer Kelvin.
        """
        if not measurement_condition:
            return None
        txt = measurement_condition.lower()

        m_at = re.search(r"at\s+(-?\d+(?:\.\d+)?)\s*k\b", txt)
        if m_at:
            return round(float(m_at.group(1)))

        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(k|°c|c)\b", txt)
        if not m:
            return None
        value = float(m.group(1))
        unit = m.group(2)
        if unit in {"°c", "c"}:
            return round(value + 273)
        return round(value)

    @staticmethod
    def build_composition_json(formula: str) -> dict:
        """Convert formula string like ``Ti42Hf21Nb21V16`` to a composition dict."""
        comp: Dict[str, float] = {}
        if not formula:
            return comp
        cleaned = re.sub(r"[()[\]{}]", "", formula)
        for element, amount in re.findall(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)", cleaned):
            comp[element] = float(amount)
        return comp

    def validate_composition_json(
        self, comp_json: dict, formula_source: str = ""
    ) -> Tuple[dict, List[str]]:
        """Validate element symbols and check that at% sums to ~100."""
        warnings: List[str] = []
        cleaned: Dict[str, float] = {}

        for elem, val in comp_json.items():
            if elem not in self.rules.valid_elements:
                warnings.append(f"Invalid element '{elem}' removed from {formula_source}")
                continue
            cleaned[elem] = val

        if cleaned:
            total = sum(cleaned.values())
            if abs(total - 100) > 2:
                warnings.append(
                    f"Composition sum = {total:.2f} at%, expected ~100 (±2%). Formula: {formula_source}"
                )
        return cleaned, warnings

    @staticmethod
    def extract_first_doi(text: Optional[str]) -> str:
        """Extract first DOI-like token from *text*."""
        if not text:
            return ""
        m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
        return m.group(0) if m else ""

    def infer_main_phase(self, char_text: str) -> str:
        """Infer main phase from characterisation text using rule patterns."""
        if not char_text:
            return ""
        ctext = char_text.lower()
        phases: List[str] = []
        seen: set = set()
        for pattern, label in self.rules.phase_patterns.items():
            if pattern in ctext and label not in seen:
                phases.append(label)
                seen.add(label)
        if not phases:
            return ""
        return " + ".join(phases[:3])

    def infer_precipitates(self, char_text: str) -> bool:
        """Return *True* if precipitate keywords are found in *char_text*."""
        if not char_text:
            return False
        ctext = char_text.lower()
        return any(kw in ctext for kw in self.rules.precipitate_keywords)

    def parse_key_params(
        self, process_text: str, processing_params: Optional[dict] = None
    ) -> dict:
        """Build structured ``Key_Params_JSON`` from LLM output or regex."""
        params: Dict[str, Any] = {}

        if processing_params and isinstance(processing_params, dict):
            for k, v in processing_params.items():
                if v is not None:
                    params[k] = v
        if params:
            return params

        if not process_text:
            return params

        for param_name, compiled in self.rules._compiled_param_patterns.items():
            for pat in compiled:
                m = pat.search(process_text)
                if m:
                    params[param_name] = float(m.group(1))
                    break

        if re.search(r"\bargon\b", process_text, re.IGNORECASE):
            params["Shielding_Gas"] = "Ar"
        elif re.search(r"\bnitrogen\b", process_text, re.IGNORECASE):
            params["Shielding_Gas"] = "N2"

        return params

    @staticmethod
    def build_microstructure_text(comp: dict) -> str:
        """Build ``Microstructure_Text_For_AI`` from dedicated fields or legacy characterisation."""
        micro_desc = comp.get("microstructure_description")
        if micro_desc and micro_desc.strip():
            return micro_desc.strip()

        char = comp.get("characterisation")
        if isinstance(char, dict):
            parts = []
            for key in ("Microstructure", "EBSD", "SEM", "TEM"):
                val = char.get(key)
                if val and isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val.strip()}")
            if parts:
                return "; ".join(parts)
            all_parts = [f"{k}: {v}" for k, v in char.items() if isinstance(v, str) and v.strip()]
            if all_parts:
                return "; ".join(all_parts)

        return "not provided"

    def normalize_property_name(self, name: Optional[str]) -> Optional[str]:
        """Map a free-text property name to a standardised column name."""
        if not name:
            return name
        key = name.strip().lower()
        mapped = self.rules.property_name_mapping.get(key)
        if mapped:
            return mapped
        return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")

    def infer_process_category(self, process_text: str) -> str:
        """Classify manufacturing process from free text."""
        t = (process_text or "").lower()
        for category, keywords in self.rules.process_category_keywords.items():
            if any(kw in t for kw in keywords):
                return category
        return "Unknown"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sample(self, comp: dict, mat_idx: int, s_idx: int) -> Dict[str, Any]:
        """Build a single ``Processed_Sample`` entry from a composition dict."""
        comp_raw = comp.get("composition", "") or ""

        match = re.search(r"\[(.*?)\]\s*$", comp_raw)
        bracket_suffix = match.group(1).replace(" ", "_") if match else ""
        orientation = comp.get("build_orientation") or ""
        orientation_suffix = orientation.replace(" ", "_").replace("/", "-") if orientation else ""

        if bracket_suffix and orientation_suffix:
            condition_suffix = f"{bracket_suffix}_{orientation_suffix}"
        elif bracket_suffix:
            condition_suffix = bracket_suffix
        elif orientation_suffix:
            condition_suffix = orientation_suffix
        else:
            condition_suffix = "Default"

        props = comp.get("properties_of_composition", []) or []
        tests: List[Dict[str, Any]] = []
        grain_size_from_props: Optional[float] = None
        for j, p in enumerate(props, start=1):
            numeric_val = p.get("value_numeric")
            prop_std_name = self.normalize_property_name(p.get("property_name"))

            if prop_std_name == "Grain_Size" and numeric_val is not None:
                unit = (p.get("unit") or "").lower().strip()
                if "nm" in unit:
                    grain_size_from_props = numeric_val / 1000.0
                else:
                    grain_size_from_props = numeric_val

            if numeric_val is None:
                continue
            tests.append({
                "Test_ID": f"T{j:03d}",
                "Test_Temperature_K": self.parse_temperature_to_k(p.get("measurement_condition")),
                "Property_Type": prop_std_name,
                "Property_Value": numeric_val,
                "Property_Unit": p.get("unit"),
            })

        process_text = comp.get("processing_conditions")
        if isinstance(process_text, dict):
            process_text = json.dumps(process_text, ensure_ascii=False)

        microstructure_text = self.build_microstructure_text(comp)

        main_phase = comp.get("main_phase")
        if not main_phase:
            main_phase = self.infer_main_phase(microstructure_text)

        has_precipitates_val = comp.get("has_precipitates")
        if has_precipitates_val is None:
            has_precipitates_val = self.infer_precipitates(microstructure_text)

        grain_size_um = comp.get("grain_size_avg_um")
        if grain_size_um is None and grain_size_from_props is not None:
            grain_size_um = grain_size_from_props

        process_category = comp.get("process_category")
        if not process_category or process_category == "Unknown":
            process_category = self.infer_process_category(process_text or "")

        key_params = self.parse_key_params(process_text or "", comp.get("processing_params"))
        if orientation and "Build_Orientation" not in key_params:
            key_params["Build_Orientation"] = orientation

        return {
            "Sample_ID": f"S{mat_idx:03d}_{s_idx:02d}_{condition_suffix}",
            "Process_Category": process_category,
            "Process_Text_For_AI": process_text or "not provided",
            "Key_Params_JSON": key_params,
            "Main_Phase": main_phase or "",
            "Microstructure_Text_For_AI": microstructure_text,
            "Has_Precipitates": has_precipitates_val,
            "Grain_Size_avg_um": grain_size_um,
            "Performance_Tests": tests,
        }
