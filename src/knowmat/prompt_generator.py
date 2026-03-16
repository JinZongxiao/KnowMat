"""
Prompt generation utilities for KnowMat 2.0.

This module centralises the construction of system and user prompts for the
extraction agent. The original KnowMat pipeline relied on a fixed system
prompt specifying detailed instructions for extracting compositions,
processing conditions, characterisation techniques and properties. The
agentic version reuses these instructions but allows them to be modified
based on the detected sub‑field and evaluation feedback.

Clients should call :func:`generate_system_prompt` to obtain the core
extraction instructions and :func:`generate_user_prompt` to wrap the
paper text. Both prompts are now loaded from templates in ``prompts/``
via :mod:`knowmat.prompt_loader`, so prompt content can be maintained
without changing Python logic.
"""

from typing import Optional

from knowmat.prompt_loader import load_text_template


def generate_system_prompt(sub_field: Optional[str] = None) -> str:
    """Return the system prompt for the extraction agent.

    If a ``sub_field`` is provided it will be inserted near the top of the
    instructions to encourage the LLM to pay special attention to that
    domain.  The rest of the prompt closely follows the original
    specification from KnowMat v1.

    Parameters
    ----------
    sub_field: Optional[str]
        A detected sub‑field of materials science.  If provided, a line
        stating ``"Sub‑field: <sub_field>"`` will be included near the
        beginning of the prompt.  Otherwise this line is omitted.

    Returns
    -------
    str
        A detailed set of instructions for the LLM.
    """
    sub_field_line = ""
    if sub_field:
        sub_field_line = f"Sub-field: {sub_field.capitalize()}\n\n"
    template = load_text_template("extraction_system_template.txt")
    # Use explicit token replacement instead of str.format():
    # the template includes literal JSON braces that would otherwise be
    # interpreted as format fields and raise KeyError.
    return template.replace("{sub_field_line}", sub_field_line)


def generate_user_prompt(text: str) -> str:
    """Wrap the user message around the paper text.

    This helper simply surrounds the document text with a short instruction to
    perform the extraction.  It can be extended in the future to include
    allowed property lists or other context if desired.
    """
    template = load_text_template("extraction_user_template.txt")
    return template.replace("{paper_text}", text)