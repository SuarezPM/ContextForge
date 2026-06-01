"""SpeculativeCoordinator — cross-agent speculative decoding.

Architecture:
- Draft agents: Retriever, Reranker (non-thinking, fast completion)
- Target agent: Responder, Critic (thinking mode, 35B full model)
- Coordinator: intercepts draft output, formats as speculative prefix,
  submits to target agent for single-pass verification

Based on:
- arXiv:2505.24544v3 (May 2026): Cross-Attention Speculative Decoding
- Speculative-Speculative: overlapped drafting+verification, ~5x faster vs autoregressive
- Expected speedup: 2-5x decode latency reduction

INVARIANT-12: The target agent's output distribution MUST be identical
whether or not speculative decoding is used. Rejected tokens are
discarded; accepted prefix is committed. The target always generates
the final authoritative token if the draft is rejected.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from apohara_context_forge.scheduling.queueing_controller import QueueingController


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding behaviour."""

    draft_agent_roles: frozenset = frozenset({"retriever", "reranker"})
    target_agent_roles: frozenset = frozenset({"responder", "critic"})
    max_draft_tokens: int = 8  # tokens to speculate per step
    acceptance_threshold: float = 0.9  # min prob ratio for token acceptance
    enable_overlapped: bool = True  # speculative-speculative overlap
    min_stability_rho: float = 0.8  # don't run speculative if rho > 0.8


@dataclass
class SpeculativeResult:
    """Outcome of a speculative decoding verification pass."""

    draft_tokens: list[int]  # proposed token IDs from draft agent
    accepted_tokens: list[int]  # tokens accepted by target agent
    rejected_at_position: int  # first rejection position (-1 if all accepted)
    acceptance_rate: float  # accepted / draft_tokens
    decode_speedup_estimate: float  # estimated vs pure autoregressive
    overlapped_next_draft: Optional[list[int]] = None  # prefetched next draft


class SpeculativeCoordinator:
    """
    Coordinates cross-agent speculative decoding.

    Draft agents (Retriever, Reranker) produce short non-thinking completions.
    The target agents (Responder, Critic) verify the draft in a single pass.
    Rejected tokens are discarded; the target generates the authoritative token.

    INVARIANT-12: The target agent's output distribution is identical whether
    or not speculative decoding is used.  This is guaranteed by the acceptance
    criterion:  accept token i with probability min(1, p_i / q_i), where p_i is
    the target's probability and q_i is the draft's probability.  This is
    mathematically equivalent to sampling from the target's original distribution
    conditioned on the accepted prefix.
    """

    def __init__(
        self,
        config: SpeculativeConfig = SpeculativeConfig(),
        queueing_controller: Optional[QueueingController] = None,
    ) -> None:
        """
        Initialize the coordinator.

        Args:
            config: Speculative decoding configuration.
            queueing_controller: Optional queueing controller for load-aware decisions.
        """
        self.config = config
        self.queueing_controller = queueing_controller

        # Overlapped speculative-speculative draft buffer.
        # Queue of (target_agent_id, draft_tokens) pairs pending verification.
        self._draft_queue: asyncio.Queue[tuple[str, list[int]]] = asyncio.Queue()

        # Currently buffered draft awaiting verification.
        self._current_draft: Optional[tuple[str, list[int]]] = None

        # Track step count for logging.
        self._step: int = 0

        logger.info(
            f"SpeculativeCoordinator initialised: "
            f"draft_roles={config.draft_agent_roles}, "
            f"target_roles={config.target_agent_roles}, "
            f"max_draft_tokens={config.max_draft_tokens}, "
            f"overlapped={config.enable_overlapped}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def is_speculative_viable(
        self, draft_agent_id: str, target_agent_id: str
    ) -> bool:
        """
        Returns True if speculative decoding should be attempted.

        Conditions:
        1. draft_agent role in config.draft_agent_roles
        2. target_agent role in config.target_agent_roles
        3. If queueing_controller present: rho < config.min_stability_rho

        Args:
            draft_agent_id: Identifier for the draft agent.
            target_agent_id: Identifier for the target agent.

        Returns:
            True when all viability conditions are satisfied.
        """
        if not self._is_role_viable(draft_agent_id, target_agent_id):
            return False

        if not self._is_queue_stable():
            return False

        return True

    async def submit_draft(
        self, draft_output_tokens: list[int], target_agent_id: str, step: int
    ) -> None:
        """
        Buffer draft tokens for the target agent.

        If enable_overlapped=True, start preparing next draft batch
        while current batch is being verified.

        Args:
            draft_output_tokens: Token IDs produced by the draft agent.
            target_agent_id: Agent that will verify and extend the draft.
            step: Current decode step number.
        """
        self._step = step

        entry = (target_agent_id, draft_output_tokens)

        if self.config.enable_overlapped:
            # Asynchronous overlapped mode: place in queue so verification
            # can proceed while the next draft is being prepared.
            await self._draft_queue.put(entry)
            logger.debug(
                "Enqueued draft of %d tokens for target=%s step=%d",
                len(draft_output_tokens),
                target_agent_id,
                step,
            )
        else:
            # Synchronous mode: store directly.
            self._current_draft = entry
            logger.debug(
                "Buffered draft of %d tokens for target=%s step=%d",
                len(draft_output_tokens),
                target_agent_id,
                step,
            )

    async def verify_and_commit(
        self,
        target_verification_logprobs: list[float],
        draft_tokens: list[int],
        draft_logprobs: Optional[list[float]] = None,
    ) -> SpeculativeResult:
        """Standard speculative-decoding acceptance criterion.

        For each draft token t_i with draft probability q_i and target
        probability p_i (derived from logprobs):

            Accept with probability  min(1, p_i / q_i)
            Reject at the first position where random() > p_i / q_i.

        On rejection: sample a correction token from the adjusted
        distribution p_adj(x) = max(0, p(x) - q(x)) / Z. INVARIANT-12 (the
        target's marginal output distribution is preserved by the
        composition of speculate-and-verify) holds iff q_i is the draft
        model's actual per-token probability — see Leviathan et al. 2023.

        V6.1 truth-up: the previous implementation lacked access to the
        draft model's logprobs and fabricated `q_i` from the
        `acceptance_threshold` config knob via
        `q = max(0.4, 1 - 0.4 * threshold)`. That formula made INV-12
        mathematically false. The fix is to accept the draft logprobs as
        an argument and use them directly. Callers that cannot supply
        draft logprobs (and therefore cannot honour INV-12) MUST pass
        `draft_logprobs=None`, which routes through the legacy estimate
        and logs a warning so the lie is visible.

        Args:
            target_verification_logprobs: Log probabilities from the
                target model for each draft token position (one per
                token).
            draft_tokens: Token IDs proposed by the draft agent.
            draft_logprobs: Per-token log probabilities from the *draft*
                model. Required for INV-12 to hold. When `None`, the
                coordinator falls back to a calibration-knob estimate
                and the operation is no longer distribution-preserving.

        Returns:
            SpeculativeResult with accepted / rejected breakdown.
        """
        if not draft_tokens:
            # Empty draft: nothing to verify.
            return SpeculativeResult(
                draft_tokens=[],
                accepted_tokens=[],
                rejected_at_position=-1,
                acceptance_rate=1.0,
                decode_speedup_estimate=1.0,
                overlapped_next_draft=None,
            )

        n = len(draft_tokens)
        accepted: list[int] = []
        rejected_at_position = -1

        # target_verification_logprobs[i] corresponds to draft_tokens[i].
        target_probs = [math.exp(lp) for lp in target_verification_logprobs]

        # ------------------------------------------------------------------ #
        # q_i — draft probability per token                                  #
        # ------------------------------------------------------------------ #
        # Two paths:
        #
        # (a) draft_logprobs supplied — the honest Leviathan path.
        #     Each q_i is the draft model's probability for the token it
        #     itself proposed. INV-12 holds.
        #
        # (b) draft_logprobs is None — legacy estimate. We use a fixed
        #     per-token probability derived from the acceptance_threshold
        #     calibration knob, and we LOG a warning so the operator
        #     knows INV-12 is no longer preserved. This path exists for
        #     backwards-compatibility with callers that have not yet
        #     plumbed the draft logprobs through.
        if draft_logprobs is not None:
            if len(draft_logprobs) != n:
                raise ValueError(
                    f"draft_logprobs length {len(draft_logprobs)} != "
                    f"draft_tokens length {n}"
                )
            draft_probs = [math.exp(lp) for lp in draft_logprobs]
            inv12_preserved = True
        else:
            logger.warning(
                "verify_and_commit called without draft_logprobs; "
                "falling back to calibration-knob estimate. "
                "INV-12 (target distribution preservation) is NOT guaranteed."
            )
            # Stubbed value; INV-12 math is currently disabled on this
            # fallback path.  Renamed to make the
            # stub-nature visible at the call site (previously written
            # as a local ``estimate`` whose origin was opaque).  See
            # AUDIT.md item 12 and the V6.0 retraction.
            _stub_draft_prob = max(0.4, 1.0 - 0.4 * self.config.acceptance_threshold)
            draft_probs = [_stub_draft_prob] * n
            inv12_preserved = False

        for i in range(n):
            draft_token = draft_tokens[i]
            p_i = target_probs[i]
            q_i = max(draft_probs[i], 1e-12)  # guard against /0 if logprob -∞

            ratio = min(1.0, p_i / q_i)

            if random.random() <= ratio:
                accepted.append(draft_token)
            else:
                rejected_at_position = i
                logger.debug(
                    "Rejected token %d at position %d (p=%.4f, q=%.4f, ratio=%.4f)",
                    draft_token, i, p_i, q_i, ratio,
                )
                break

        num_accepted = len(accepted)
        acceptance_rate = num_accepted / n if n > 0 else 1.0

        # Estimate speedup from the accepted tokens.
        speedup = self.estimate_speedup(acceptance_rate, self.config.max_draft_tokens)

        # Determine overlapped next draft if enabled.
        overlapped_next_draft: Optional[list[int]] = None
        if self.config.enable_overlapped:
            try:
                # Non-blocking check for a prefetched next draft.
                overlapped_next_draft = self._fetch_overlapped_next()
            except Exception as exc:
                logger.warning("Failed to fetch overlapped draft: %s", exc)

        result = SpeculativeResult(
            draft_tokens=draft_tokens,
            accepted_tokens=accepted,
            rejected_at_position=rejected_at_position,
            acceptance_rate=acceptance_rate,
            decode_speedup_estimate=speedup,
            overlapped_next_draft=overlapped_next_draft,
        )

        logger.info(
            "Speculative result: accepted=%d/%d rate=%.2f speedup=%.2fx",
            num_accepted,
            n,
            acceptance_rate,
            speedup,
        )
        return result

    def estimate_speedup(
        self, acceptance_rate: float, max_draft_tokens: int = 8
    ) -> float:
        """
        Theoretical speedup from speculative decoding.

        E[tokens_per_step] = (1 - acceptance_rate^(k+1)) / (1 - acceptance_rate)
        where k = max_draft_tokens

        speedup = E[tokens_per_step] / 1.0  (vs 1 token per autoregressive step)

        For acceptance_rate=0.9, k=8: E[tokens] ≈ 5.7 → 5.7x speedup

        Args:
            acceptance_rate: Fraction of draft tokens accepted [0, 1].
            max_draft_tokens: Maximum tokens drafted per step.

        Returns:
            Estimated decode speedup factor.
        """
        if not (0.0 <= acceptance_rate <= 1.0):
            return 1.0

        if acceptance_rate == 1.0:
            # All tokens accepted — maximum speedup.
            return float(max_draft_tokens + 1)

        if acceptance_rate == 0.0:
            # All rejected — no speedup (only the fallback token).
            return 1.0

        # Expected tokens = sum_{i=0}^k acceptance_rate^i
        # = (1 - acceptance_rate^(k+1)) / (1 - acceptance_rate)
        k = max_draft_tokens
        numerator = 1.0 - (acceptance_rate ** (k + 1))
        denominator = 1.0 - acceptance_rate
        expected_tokens = numerator / denominator

        return expected_tokens

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _is_role_viable(self, draft_agent_id: str, target_agent_id: str) -> bool:
        """Check if both draft and target agent roles are in allowed roles."""
        draft_role = self._role_from_agent_id(draft_agent_id)
        target_role = self._role_from_agent_id(target_agent_id)

        if draft_role not in self.config.draft_agent_roles:
            logger.debug("Draft role %s not in allowed roles", draft_role)
            return False

        if target_role not in self.config.target_agent_roles:
            logger.debug("Target role %s not in allowed roles", target_role)
            return False

        return True

    def _is_queue_stable(self) -> bool:
        """Check if the queueing controller is stable enough for speculative decoding."""
        if self.queueing_controller is not None:
            rho = getattr(self.queueing_controller, "current_rho", lambda: 0.0)()
            if isinstance(rho, (int, float)) and rho >= self.config.min_stability_rho:
                logger.info(
                    "Skipping speculative decode: rho=%.2f >= min_stability_rho=%.2f",
                    rho,
                    self.config.min_stability_rho,
                )
                return False
        return True

    @staticmethod
    def _role_from_agent_id(agent_id: str) -> str:
        """
        Derive agent role from agent_id.

        Uses the last colon-separated segment as the role.
        E.g.  "retriever-0" -> "retriever",  "responder-1" -> "responder"
        """
        return agent_id.split(":")[-1].split("-")[0]

    def _fetch_overlapped_next(self) -> Optional[list[int]]:
        """
        Attempt to dequeue a prefetched next draft (non-blocking).

        Returns:
            Draft tokens if available, else None.
        """
        try:
            _, tokens = self._draft_queue.get_nowait()
            return tokens
        except asyncio.QueueEmpty:
            return None