"""
Top-level assembly of the KnowMat 2.0 LangGraph workflow.

This module wires together the individual processing nodes and drives the
end-to-end extraction pipeline for a single PDF document.  Domain-specific
conversion logic lives in :mod:`knowmat.schema_converter` and report
generation in :mod:`knowmat.report_writer`.
"""

import json
import os
import uuid
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from knowmat.states import KnowMatState
from knowmat.report_writer import write_comprehensive_report
from knowmat.nodes.paddleocrvl_parse_pdf import parse_pdf_with_paddleocrvl
from knowmat.nodes.subfield_detection import detect_sub_field
from knowmat.nodes.extraction import extract_data
from knowmat.nodes.evaluation import evaluate_data
from knowmat.nodes.aggregator import aggregate_runs
from knowmat.nodes.validator import validate_and_correct
from knowmat.nodes.flagging import assess_final_quality
from knowmat.nodes.standardize import standardize_properties
from knowmat.nodes.schema_convert import convert_to_target_schema
from knowmat.app_config import Settings, settings
from knowmat.config import _env_path


def evaluation_condition(state: KnowMatState) -> str:
    """Decide whether to rerun extraction or proceed to aggregation.

    Called by the graph when the evaluation node completes.  If
    ``state['needs_rerun']`` is true and ``state['run_count']`` is less
    than ``state['max_runs']`` the function returns the name of the
    extraction node to trigger another cycle.  Otherwise it returns
    ``aggregate_runs`` to begin the two-stage manager process.
    """
    run_count = state.get("run_count", 0)
    max_runs = state.get("max_runs", 3)
    needs_rerun = state.get("needs_rerun", False)
    if needs_rerun and run_count < max_runs:
        return "extract_data"
    return "aggregate_runs"


def build_graph(full_pipeline: bool = True) -> StateGraph:
    """Construct the LangGraph for KnowMat 2.0 with two-stage manager.

    When *full_pipeline* is True the graph includes subfield detection,
    iterative evaluation, aggregation, validation, flagging, optional
    property standardization, and schema conversion -- all as tracked
    graph nodes with checkpoint support.
    """
    builder = StateGraph(KnowMatState)

    builder.add_node("parse_pdf", parse_pdf_with_paddleocrvl)
    builder.add_node("extract_data", extract_data)

    builder.add_edge(START, "parse_pdf")
    if not full_pipeline:
        builder.add_edge("parse_pdf", "extract_data")
        builder.add_node("convert_schema", convert_to_target_schema)
        builder.add_edge("extract_data", "convert_schema")
        builder.add_edge("convert_schema", END)
    else:
        builder.add_node("detect_sub_field", detect_sub_field)
        builder.add_node("evaluate_data", evaluate_data)
        builder.add_node("aggregate_runs", aggregate_runs)
        builder.add_node("validate_and_correct", validate_and_correct)
        builder.add_node("assess_final_quality", assess_final_quality)
        builder.add_node("standardize_properties", standardize_properties)
        builder.add_node("convert_schema", convert_to_target_schema)

        builder.add_edge("parse_pdf", "detect_sub_field")
        builder.add_edge("detect_sub_field", "extract_data")
        builder.add_edge("extract_data", "evaluate_data")
        builder.add_conditional_edges(
            "evaluate_data", evaluation_condition, ["extract_data", "aggregate_runs"]
        )
        builder.add_edge("aggregate_runs", "validate_and_correct")
        builder.add_edge("validate_and_correct", "assess_final_quality")
        builder.add_edge("assess_final_quality", "standardize_properties")
        builder.add_edge("standardize_properties", "convert_schema")
        builder.add_edge("convert_schema", END)

    return builder.compile(checkpointer=MemorySaver())


def run(
    pdf_path: str,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    max_runs: int = 3,
    subfield_model: Optional[str] = None,
    extraction_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
    manager_model: Optional[str] = None,
    flagging_model: Optional[str] = None,
    full_pipeline: bool = False,
    enable_property_standardization: bool = False,
) -> dict:
    """Run the full KnowMat 2.0 pipeline on a given input file and write results.

    Parameters
    ----------
    pdf_path : str
        Path to the materials science paper in ``.pdf`` or ``.txt`` format.
    output_dir : Optional[str]
        Directory where results will be saved.
    model_name : Optional[str]
        Override the base model (e.g., "gpt-4", "gpt-5-mini").
    max_runs : int
        Maximum number of extraction/evaluation cycles.
    subfield_model, extraction_model, evaluation_model, manager_model, flagging_model
        Per-agent model overrides.
    full_pipeline : bool
        If True, run subfield/evaluation/aggregation/validation stages.
    enable_property_standardization : bool
        If True, run optional property post-processing (extra LLM calls).

    Returns
    -------
    dict
        Results dictionary containing final_data, flag, output_dir, etc.
    """

    if _env_path:
        print(f"Loaded environment variables from: {_env_path}")

    # Build an isolated Settings instance for this run to avoid mutating
    # the module-level singleton in concurrent executions.
    overrides = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if model_name:
        overrides["model_name"] = model_name
    if subfield_model:
        overrides["subfield_model"] = subfield_model
    if extraction_model:
        overrides["extraction_model"] = extraction_model
    if evaluation_model:
        overrides["evaluation_model"] = evaluation_model
    if manager_model:
        overrides["manager_model"] = manager_model
    if flagging_model:
        overrides["flagging_model"] = flagging_model

    if overrides:
        effective_settings = Settings(**{**settings.model_dump(), **overrides})
    else:
        effective_settings = settings

    # Default all per-agent models to model_name when not overridden.
    if not subfield_model and not overrides.get("subfield_model"):
        effective_settings.subfield_model = effective_settings.model_name
    if not extraction_model and not overrides.get("extraction_model"):
        effective_settings.extraction_model = effective_settings.model_name
    if not evaluation_model and not overrides.get("evaluation_model"):
        effective_settings.evaluation_model = effective_settings.model_name
    if not manager_model and not overrides.get("manager_model"):
        effective_settings.manager_model = effective_settings.model_name
    if not flagging_model and not overrides.get("flagging_model"):
        effective_settings.flagging_model = effective_settings.model_name

    print(f"\nModel Configuration:")
    print(f"   Subfield Detection: {effective_settings.subfield_model}")
    print(f"   Extraction:         {effective_settings.extraction_model}")
    print(f"   Evaluation:         {effective_settings.evaluation_model}")
    print(f"   Aggregation:        rule-based (no LLM)")
    print(f"   Validation:         {effective_settings.manager_model}")
    print(f"   Flagging:           {effective_settings.flagging_model}")

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_output_dir = os.path.join(effective_settings.output_dir, base_name)
    os.makedirs(paper_output_dir, exist_ok=True)

    print(f"\nOutput directory: {paper_output_dir}\n")

    state: KnowMatState = {
        "pdf_path": pdf_path,
        "output_dir": paper_output_dir,
        "run_count": 0,
        "run_results": [],
        "max_runs": max_runs,
        "enable_property_standardization": enable_property_standardization,
    }

    thread_id = f"knowmat2_{base_name}_{uuid.uuid4().hex[:8]}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    graph = build_graph(full_pipeline=full_pipeline)
    for _ in graph.stream(state, thread_config, stream_mode="values"):
        pass

    final_state = graph.get_state(thread_config).values
    final_data = final_state.get("final_data", {})
    if not final_data:
        final_data = final_state.get("latest_extracted_data", {})
    flag = final_state.get("flag", False) if full_pipeline else False

    # --- Write outputs to disk ---
    output_path = os.path.join(paper_output_dir, f"{base_name}_extraction.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"Saved extraction to {output_path}")

    report_path = os.path.join(paper_output_dir, f"{base_name}_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        write_comprehensive_report(f, final_state)
    print(f"Saved analysis report to {report_path}")

    runs_path = os.path.join(paper_output_dir, f"{base_name}_runs.json")
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(final_state.get("run_results", []), f, ensure_ascii=False, indent=2)

    # --- Generate QA Report (pure computation, no LLM) ---
    qa_report = _build_qa_report(base_name, final_data, final_state)

    qa_path = os.path.join(paper_output_dir, f"{base_name}_qa_report.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, ensure_ascii=False, indent=2)
    print(f"Saved QA report to {qa_path}")

    if qa_report.get("red_line_triggers"):
        triggers = qa_report["red_line_triggers"]
        print(f"\n[RED LINE] QA check failed: {', '.join(triggers)}")
        print(f"           Human review REQUIRED for {base_name}")

    return {
        "final_data": final_data,
        "flag": flag,
        "output_dir": paper_output_dir,
        "final_confidence_score": final_state.get("final_confidence_score"),
        "aggregation_rationale": final_state.get("aggregation_rationale"),
        "human_review_guide": final_state.get("human_review_guide"),
    }


def _build_qa_report(base_name: str, final_data: dict, final_state: dict) -> dict:
    """Build the QA report dict from final extraction results."""
    materials = final_data.get("Materials", [])
    all_samples = [s for m in materials for s in m.get("Processed_Samples", [])]
    all_tests = [t for s in all_samples for t in s.get("Performance_Tests", [])]

    unknown_process_count = sum(
        1 for s in all_samples if s.get("Process_Category") == "Unknown"
    )
    phase_filled_count = sum(1 for s in all_samples if s.get("Main_Phase"))
    phase_filled_rate = phase_filled_count / len(all_samples) if all_samples else 0

    red_line_triggers = []
    if len(materials) == 0:
        red_line_triggers.append("NO_TARGET_MATERIALS")
    if len(all_tests) == 0:
        red_line_triggers.append("NO_PROPERTIES")
    if all_samples:
        unknown_ratio = unknown_process_count / len(all_samples)
        if unknown_ratio > 0.5:
            red_line_triggers.append("HIGH_UNKNOWN_PROCESS_RATIO")

    return {
        "paper_name": base_name,
        "pipeline_version": "knowmat-2.0.1",
        "materials_target_count": len(materials),
        "samples_count": len(all_samples),
        "properties_count": len(all_tests),
        "unknown_process_count": unknown_process_count,
        "phase_filled_rate": round(phase_filled_rate, 3),
        "missing_doi": 1 if not materials or not materials[0].get("Source_DOI") else 0,
        "needs_review": len(red_line_triggers) > 0,
        "red_line_triggers": red_line_triggers,
        "final_confidence_score": final_state.get("final_confidence_score"),
    }
