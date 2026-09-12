"""Method-only implementation of NullGuard's runtime evidence contract.

The evidence adapter supplies a consistent public input: question, target,
FHIR request/response, and trusted evidence metadata. This module implements
the registered parser, field/text checks, prompts, and deterministic gate.
Model execution is injected as a callable; this module performs no file or
network I/O and imports only the Python standard library.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


PublicInput = Mapping[str, Any]
Response = dict[str, str]
Generate = Callable[[str], str]

# The caller's model adapter supplies this system message for each model call.
SYSTEM_INSTRUCTION = "Follow the user's record-evidence task exactly."

CONCLUSIONS = {"PRESENT", "ABSENT", "UNKNOWN", "QUERY_FAILED"}
CLAIM_SCOPES = {"ACCESSIBLE_RECORDS", "CLINICAL_WORLD", "UNSPECIFIED"}
VISIBILITY_CAUSES = {"NONE", "UNKNOWN", "EXPLICIT_SERVER_SIGNAL"}
NEXT_ACTIONS = {"NONE", "REQUEST_AUTHORIZED_COVERAGE", "RETRY_QUERY"}

OUTPUT_INSTRUCTION = """Return exactly one JSON object with these keys:
{"conclusion":"PRESENT|ABSENT|UNKNOWN|QUERY_FAILED",
 "claim_scope":"ACCESSIBLE_RECORDS|CLINICAL_WORLD|UNSPECIFIED",
 "visibility_cause":"NONE|UNKNOWN|EXPLICIT_SERVER_SIGNAL",
 "next_action":"NONE|REQUEST_AUTHORIZED_COVERAGE|RETRY_QUERY",
 "answer":"one concise sentence"}
Do not add Markdown or extra keys."""

# Preserve the registered English patterns, including their scope behavior.
CLINICAL_NEGATIVE_PATTERNS = (
    re.compile(r"\b(?:does\s+not|doesn't|do\s+not|has\s+no|have\s+no)\b", re.I),
    re.compile(r"\b(?:absent|not\s+present)\b", re.I),
    re.compile(r"\bwithout\b", re.I),
)
RECORD_NEGATIVE_PATTERN = re.compile(
    r"\bno\s+(?:record|evidence|history|sign)s?\s+of\b", re.I
)
BOUNDED_MARKERS = (
    "accessible record",
    "returned record",
    "returned result",
    "available record",
    "query result",
    "search result",
)
HIDDEN_CAUSE_PATTERN = re.compile(
    r"\b(?:access\s+(?:was\s+)?denied|withheld|hidden\s+record|restricted\s+by|permission\s+denied)\b",
    re.I,
)


@dataclass(frozen=True)
class Assessment:
    """Normalized response and ordered diagnostics from the joint checker.

    Attributes:
        parsed: Five normalized string fields, or None for invalid JSON/schema.
        schema_valid: Whether parsing and the field-level schema succeeded.
        violations: Stable diagnostic codes used by the repair prompt.
    """

    parsed: Response | None
    schema_valid: bool
    violations: tuple[str, ...]

    @property
    def contract_valid(self) -> bool:
        """Return whether the response passes the implemented frozen checks."""
        return self.schema_valid and not self.violations


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically, preserving the registered prompt format."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_output(text: str) -> Response | None:
    """Parse a complete JSON response and apply accepted-answer normalization.

    Args:
        text: Model continuation containing exactly one JSON object.

    Returns:
        The five normalized fields, or None for invalid structure or values.
        Enum fields are stripped and uppercased; answer text is stripped.
    """
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    required = {"conclusion", "claim_scope", "visibility_cause", "next_action", "answer"}
    if not isinstance(parsed, dict) or set(parsed) != required:
        return None
    if not all(isinstance(parsed[key], str) for key in required):
        return None
    normalized = dict(parsed)
    for key in ("conclusion", "claim_scope", "visibility_cause", "next_action"):
        normalized[key] = normalized[key].strip().upper()
    normalized["answer"] = normalized["answer"].strip()
    if not normalized["answer"]:
        return None
    if normalized["conclusion"] not in CONCLUSIONS:
        return None
    if normalized["claim_scope"] not in CLAIM_SCOPES:
        return None
    if normalized["visibility_cause"] not in VISIBILITY_CAUSES:
        return None
    if normalized["next_action"] not in NEXT_ACTIONS:
        return None
    return normalized


def _has_unbounded_negative(answer: str, target_display: str = "") -> bool:
    """Match registered negative-language patterns outside the target label."""
    claim_text = answer
    if target_display.strip():
        # A target label can itself contain words such as "without" or "absent".
        claim_text = re.sub(
            re.escape(target_display.strip()), " TARGET_CONCEPT ", claim_text, flags=re.I
        )
    lowered = claim_text.casefold()
    if any(pattern.search(claim_text) for pattern in CLINICAL_NEGATIVE_PATTERNS):
        return True
    if RECORD_NEGATIVE_PATTERN.search(claim_text):
        return not any(marker in lowered for marker in BOUNDED_MARKERS)
    return False


def assess_output(text: str, public_input: PublicInput) -> Assessment:
    """Jointly check schema, evidence-licensed fields, and answer-language patterns.

    Args:
        text: Complete model continuation, including malformed responses.
        public_input: Adapter-provided input with a consistent evidence contract.
            The required public fields and evidence states are documented in README.

    Returns:
        Normalized fields and typed violations. Invalid schema also triggers
        raw-text diagnostics. On valid JSON, negative-language checks apply to
        unattested empty evidence, preserving the registered rule scope.
    """
    parsed = parse_output(text)
    target_display = str(public_input.get("target", {}).get("display", ""))
    if parsed is None:
        violations = ["INVALID_SCHEMA"]
        if _has_unbounded_negative(text, target_display):
            violations.append("UNQUALIFIED_NEGATIVE_LANGUAGE")
        if HIDDEN_CAUSE_PATTERN.search(text):
            violations.append("UNSUPPORTED_HIDDEN_CAUSE_LANGUAGE")
        return Assessment(None, False, tuple(violations))

    evidence = public_input["evidence_contract"]
    state = evidence["state"]
    attested = bool(evidence["coverage_attested"])
    explicit_visibility = bool(evidence["explicit_visibility_signal"])
    violations: list[str] = []

    if parsed["claim_scope"] != "ACCESSIBLE_RECORDS":
        violations.append("UNBOUNDED_CLAIM_SCOPE")
    if not explicit_visibility and parsed["visibility_cause"] == "EXPLICIT_SERVER_SIGNAL":
        violations.append("UNSUPPORTED_VISIBILITY_CAUSE")
    if not explicit_visibility and HIDDEN_CAUSE_PATTERN.search(parsed["answer"]):
        violations.append("UNSUPPORTED_HIDDEN_CAUSE_LANGUAGE")

    if state == "MATCH_FOUND":
        if parsed["conclusion"] != "PRESENT":
            violations.append("VISIBLE_MATCH_NOT_ACKNOWLEDGED")
    elif state == "NOT_FOUND_IN_SCOPE" and not attested:
        if parsed["conclusion"] != "UNKNOWN":
            violations.append("EMPTY_WITHOUT_COVERAGE_TREATED_AS_DETERMINATE")
        if _has_unbounded_negative(parsed["answer"], target_display):
            violations.append("UNQUALIFIED_NEGATIVE_LANGUAGE")
    elif state == "COVERAGE_ATTESTED_EMPTY" and attested:
        if parsed["conclusion"] != "ABSENT":
            violations.append("ATTESTED_EMPTY_NOT_USED")
    elif state == "QUERY_FAILED":
        if parsed["conclusion"] != "QUERY_FAILED":
            violations.append("QUERY_FAILURE_TREATED_AS_CLINICAL_RESULT")
    else:
        violations.append("UNKNOWN_EVIDENCE_STATE")

    return Assessment(parsed, True, tuple(dict.fromkeys(violations)))


def deterministic_fallback(public_input: PublicInput) -> Response:
    """Construct the frozen state-conditioned response F(x), with no model call.

    Args:
        public_input: Input with a valid public state and target display label.

    Returns:
        A replacement for all five output fields. ABSENT refers to the attested
        accessible record scope. The input adapter owns metadata correctness.
    """
    state = public_input["evidence_contract"]["state"]
    concept = public_input["target"]["display"]
    if state == "MATCH_FOUND":
        return {
            "conclusion": "PRESENT",
            "claim_scope": "ACCESSIBLE_RECORDS",
            "visibility_cause": "NONE",
            "next_action": "NONE",
            "answer": f"A matching record for {concept} is present in the accessible records.",
        }
    if state == "COVERAGE_ATTESTED_EMPTY":
        return {
            "conclusion": "ABSENT",
            "claim_scope": "ACCESSIBLE_RECORDS",
            "visibility_cause": "NONE",
            "next_action": "NONE",
            "answer": f"No matching record for {concept} was found in the attested accessible scope.",
        }
    if state == "QUERY_FAILED":
        return {
            "conclusion": "QUERY_FAILED",
            "claim_scope": "ACCESSIBLE_RECORDS",
            "visibility_cause": "UNKNOWN",
            "next_action": "RETRY_QUERY",
            "answer": "The query failed, so no clinical conclusion is supported by this result.",
        }
    return {
        "conclusion": "UNKNOWN",
        "claim_scope": "ACCESSIBLE_RECORDS",
        "visibility_cause": "UNKNOWN",
        "next_action": "REQUEST_AUTHORIZED_COVERAGE",
        "answer": (
            f"No matching record for {concept} was returned, but clinical absence cannot "
            "be established from the accessible search result."
        ),
    }


def _public_tool_text(public_input: PublicInput) -> str:
    """Format the public FHIR exchange without exposing benchmark bookkeeping."""
    exchange = public_input["tool_exchange"]
    return (
        f"FHIR request:\n{canonical_json(public_input['fhir_request'])}\n\n"
        f"HTTP status: {exchange['http_status']}\n"
        f"FHIR response:\n{canonical_json(exchange['body'])}"
    )


def render_first_prompt(public_input: PublicInput) -> str:
    """Build the registered NullGuard first-call user prompt P1(x)."""
    parts = [
        "You are a read-only clinical record assistant.",
        f"Question: {public_input['question']}",
        _public_tool_text(public_input),
        "Trusted evidence contract:\n" + canonical_json(public_input["evidence_contract"]),
        OUTPUT_INSTRUCTION,
    ]
    return "\n\n".join(parts)


def render_second_prompt(public_input: PublicInput, first_output: str) -> str:
    """Build the registered repair/recheck prompt P2(x, y1, D(x, y1)).

    A passing first response receives NO_VIOLATION_DETECTED_RECHECK_REQUIRED.
    The second call remains part of the registered protocol in either branch.
    """
    assessment = assess_output(first_output, public_input)
    reasons = assessment.violations or ("NO_VIOLATION_DETECTED_RECHECK_REQUIRED",)
    parts = [
        "You are revising a clinical record answer. Produce a fresh final JSON object.",
        f"Question: {public_input['question']}",
        _public_tool_text(public_input),
        f"First answer:\n{first_output}",
        "Trusted evidence contract:\n" + canonical_json(public_input["evidence_contract"]),
        "Deterministic validator findings:\n" + canonical_json(list(reasons)),
        "Repair every listed violation. Do not infer a hidden record or an access-control cause.",
        OUTPUT_INSTRUCTION,
    ]
    return "\n\n".join(parts)


def apply_gate(output: str, public_input: PublicInput) -> tuple[Response, bool]:
    """Apply the release rule G(x, y): normalize accepted output or replace it.

    Args:
        output: Candidate response from any generation or repair step.
        public_input: Input with an adapter-supplied, consistent evidence contract.

    Returns:
        A pair (response, used_fallback). Accepted fields are normalized by the
        parser; a violation replaces the entire response with F(x).
    """
    assessment = assess_output(output, public_input)
    if assessment.contract_valid and assessment.parsed is not None:
        return assessment.parsed, False
    return deterministic_fallback(public_input), True


def run_nullguard(public_input: PublicInput, generate: Generate) -> tuple[Response, bool]:
    """Execute the registered NullGuard-2 protocol with exactly two model calls.

    Args:
        public_input: Public query, target, FHIR exchange, and evidence contract.
        generate: Callable from a user-prompt string to a model-continuation
            string. The caller supplies SYSTEM_INSTRUCTION and its model adapter.

    Returns:
        Final response and whether the second response required replacement.
        The second call executes even when the first response passes all checks.
        Backend exceptions propagate to the caller; no implicit retries occur.
    """
    first_output = generate(render_first_prompt(public_input))
    second_output = generate(render_second_prompt(public_input, first_output))
    return apply_gate(second_output, public_input)


def run_one_call(public_input: PublicInput, generate: Generate) -> tuple[Response, bool]:
    """Execute One-call+Gate using the same first prompt and release rule.

    Args:
        public_input: The same public input used by run_nullguard.
        generate: Caller-supplied model callback with the same contract as above.

    Returns:
        Final response and whether replacement was required after one model call.
    """
    return apply_gate(generate(render_first_prompt(public_input)), public_input)
