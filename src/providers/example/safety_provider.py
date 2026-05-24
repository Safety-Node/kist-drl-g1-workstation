"""
Safety Provider -- KIST DRL G1 Workstation
==========================================

drawio C4 Container:
    Name        : Safety Provider
    Technology  : Python
    Description : Validates Cortex LLM input/output for safety.

Edges:
    Cortex -> Safety Provider : Fused prompt + Action plan [text]
    Safety Provider -> Cortex : Validated prompt + Filtered actions [text]

Standards anchors (see safety_artifacts/00_master):
    - ISO 12100:2010 §6 (3-step risk reduction)
    - ISO 5469 (AI safety integrity)
    - ISO/TS 15066:2016 (collaborative robot speed/force limits)
    - ISO 13849-1 (safety functions PL classification)

TBD:
    - Define the prompt-side validators:
        * prompt-injection / jailbreak detection
        * disallowed semantic substitutions
          (e.g. forbidden "발사믹 -> 오리엔탈" generalizations)
        * Korean + English co-coverage
    - Define the action-side validators:
        * geometric envelope check (joint/cartesian)
        * tool-use guards (knife / tongs)
        * velocity & force ceilings (ISO/TS 15066 PFL)
        * KAPEX-G1 sync timing window
    - Define the verdict schema:
        {accept|reject|degrade}, redacted_text, rationale, hazard_ids
    - Emit decisions to decision_log + Prometheus metrics
    - Backstop: hard-coded deny list independent of LLM
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .singleton import singleton


class SafetyVerdict(str, Enum):
    """Outcome of a single safety check."""

    ACCEPT = "accept"
    DEGRADE = "degrade"   # accept but with reduced scope/velocity
    REJECT = "reject"


@dataclass
class SafetyDecision:
    """Structured Safety Provider output."""

    verdict: SafetyVerdict
    filtered_text: str
    rationale: str = ""
    hazard_ids: List[str] = field(default_factory=list)
    # TODO: add traceability fields (decision_id, trigger_rule, ts)


@singleton
class SafetyProvider:
    """
    Bidirectional safety filter between Cortex and the rest of the stack.

    Implements both the AI input-side filter (prompt) and the AI output-side
    filter (action plan). Acts as the workstation-side backstop -- it does
    NOT replace the G1 onboard ``safety_monitor`` which runs independently.
    """

    def __init__(
        self,
        rules_path: Optional[str] = None,
        strict_mode: bool = True,
    ):
        """
        Parameters
        ----------
        rules_path : str, optional
            Path to YAML/JSON rules. None = built-in defaults.
        strict_mode : bool
            If True, ambiguous decisions default to REJECT.
        """
        # TODO: load rules; pre-compile regex / classifiers
        self._rules_path = rules_path
        self._strict_mode = strict_mode
        logging.info(
            "SafetyProvider: skeleton initialized (strict=%s, rules=%s)",
            strict_mode, rules_path,
        )

    def validate_prompt(self, fused_prompt: str) -> SafetyDecision:
        """
        Run input-side checks on the fused prompt before it reaches the LLM.
        """
        # TODO: prompt injection detection
        # TODO: disallowed semantic substitution detection
        # TODO: PII redaction
        raise NotImplementedError("SafetyProvider.validate_prompt: TBD")

    def validate_action_plan(self, action_plan_json: dict) -> SafetyDecision:
        """
        Run output-side checks on the action plan returned by the LLM.
        """
        # TODO: envelope / velocity / force guards
        # TODO: tool-use whitelist
        # TODO: KAPEX-G1 cycle-time check
        raise NotImplementedError("SafetyProvider.validate_action_plan: TBD")
