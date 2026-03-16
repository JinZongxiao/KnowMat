"""
Node for detecting the materials science sub‑field and updating the prompt.

This node calls the ``subfield_extractor`` defined in
:mod:`knowmat2.extractors` to analyse the paper text and determine the
appropriate niche domain.  It returns both the detected sub‑field and an
updated extraction prompt tailored to that sub‑field.  The updated prompt
should be concatenated with any existing prompt modifications stored in
``state['updated_prompt']``.
"""

from knowmat.extractors import subfield_extractor, SubFieldDetection
from knowmat.prompt_loader import load_yaml_templates_required
from knowmat.states import KnowMatState

_SUBFIELD_PROMPT_TEMPLATE = load_yaml_templates_required(
    "subfield_detection.yaml", ("prompt_template",)
).get("prompt_template", "")


def detect_sub_field(state: KnowMatState) -> dict:
    """Detect the materials science sub‑field and return prompt updates.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state, must contain ``paper_text``.

    Returns
    -------
    dict
        Updates containing ``sub_field`` and ``updated_prompt``.  The
        returned ``updated_prompt`` includes any prior prompt stored on
        ``state['updated_prompt']`` concatenated with the new suggestion.
    """
    paper_text = state.get("paper_text", "")
    prompt = _SUBFIELD_PROMPT_TEMPLATE.format(paper_text=paper_text)
    # Invoke the extractor
    result = subfield_extractor.invoke(prompt)
    # TrustCall returns a dict with a 'responses' key containing the parsed objects
    response = result.get("responses", [None])[0]
    if not response:
        # In the unlikely event no response is returned, fall back to a sensible default
        sub_field = "experimental"
        updated = state.get("updated_prompt", "")
    else:
        # response is a SubFieldDetection instance
        # For compatibility, convert to dict if necessary
        if isinstance(response, SubFieldDetection):
            sub_field = response.sub_field
            updated = response.updated_prompt
        else:
            # Assume dict
            sub_field = response.get("sub_field")
            updated = response.get("updated_prompt", "")
    # Prepend any existing prompt updates
    prior_update = state.get("updated_prompt", "")
    if prior_update:
        new_prompt = prior_update.strip() + "\n\n" + updated.strip()
    else:
        new_prompt = updated.strip()
    return {"sub_field": sub_field, "updated_prompt": new_prompt}