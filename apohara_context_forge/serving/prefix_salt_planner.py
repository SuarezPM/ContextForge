"""Prefix cache_salt planner for vLLM Automatic Prefix Caching (APC).

vLLM's APC keys every prefix KV block by ``hash(cache_salt, token_ids, ...)``.
Two requests share prefix blocks ONLY when their ``cache_salt`` is identical
AND their leading tokens are byte-identical. This module decides, per request,
which ``cache_salt`` string each agent should carry so that:

* Agents that legitimately share the same prefix anchor get the SAME salt and
  therefore SHARE KV blocks intra-instance (free, 100% native to vLLM).
* Judge-class agents whose JCR Safety Gate fired INV-15 (``use_dense=True``)
  get a UNIQUE salt, which forces vLLM to allocate fresh blocks for them —
  physical isolation via vLLM's own block-hash keying. This is the serving-side
  realisation of INV-15: a judge under high JCR risk never reuses another
  agent's KV blocks.

This module is PURE and DETERMINISTIC for the shared path: given the same
``anchor_hash`` + ``cla_group``, the shared salt is reproducible. The isolated
(dense) path is deliberately UNIQUE per request: it mixes in a caller-supplied
``request_id`` so two distinct judge requests never collide with each other or
with the shared group.

It does NOT import vLLM, lmcache, or torch, and it does NOT construct an
AnchorPool (which is async and pulls in a heavy embedding engine). The caller
supplies the already-computed ``anchor_hash`` (the ``base_kv_hash`` an
``AnchorPool`` assigns to an anchor) and ``cla_group`` identifier. The real
``JCRSafetyGate`` is imported and used directly — INV-15 is decided by the gate,
never re-implemented here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Union

from apohara_context_forge.safety.jcr_gate import JCRSafetyGate

# Stable namespace so salts from this planner can't accidentally collide with
# salts produced by some other subsystem that hashes raw integers.
_SALT_NAMESPACE = "apohara.apc.v1"

# Prefix on the isolated (dense) salt so it is human-greppable in vLLM logs and
# can never equal a shared salt (which is prefixed "shared:").
_ISOLATED_PREFIX = "iso"
_SHARED_PREFIX = "shared"


@dataclass(frozen=True)
class SaltPlan:
    """The cache_salt decision for a single agent request.

    Attributes:
        cache_salt: The string to pass to vLLM as the per-request salt.
        shared: True if this salt is reused across agents with the same anchor
            (KV blocks are shared); False if the salt is unique to isolate the
            request (INV-15 dense path).
        reason: Human-readable explanation, mirrors the JCR gate's reasoning
            when isolation was triggered.
    """

    cache_salt: str
    shared: bool
    reason: str


def _digest(*parts: str) -> str:
    """Stable short hex digest over the namespace + parts."""
    h = hashlib.sha256()
    h.update(_SALT_NAMESPACE.encode("utf-8"))
    for p in parts:
        h.update(b"\x1f")  # unit separator — avoids ambiguous concatenation
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


class PrefixSaltPlanner:
    """Maps (anchor_hash, cla_group, JCRDecision) -> per-request cache_salt.

    Wraps a real ``JCRSafetyGate``. The gate alone owns the INV-15 decision;
    the planner only translates that decision into a vLLM salt. Pure and
    deterministic on the shared path (no GPU, no I/O).
    """

    def __init__(self, jcr_gate: Optional[JCRSafetyGate] = None):
        self._gate = jcr_gate if jcr_gate is not None else JCRSafetyGate()

    @property
    def gate(self) -> JCRSafetyGate:
        """The underlying JCRSafetyGate (for telemetry / inspection)."""
        return self._gate

    def shared_salt(
        self,
        anchor_hash: Union[int, str],
        cla_group: Union[int, str],
    ) -> str:
        """Deterministic salt for the prefix-sharing path.

        Agents with the same ``anchor_hash`` and ``cla_group`` get the SAME
        salt and therefore share vLLM prefix-cache blocks. The salt does NOT
        depend on agent_id or request_id — that is the whole point.
        """
        return f"{_SHARED_PREFIX}:{_digest(str(anchor_hash), str(cla_group))}"

    def isolated_salt(self, anchor_hash: Union[int, str], request_id: str) -> str:
        """Unique salt for the INV-15 dense path.

        Mixes in ``request_id`` so the salt is unique per request. This forces
        vLLM to key the prefix blocks differently, giving the judge physically
        isolated KV blocks (no reuse from the shared group, no reuse between two
        distinct judge requests).
        """
        return f"{_ISOLATED_PREFIX}:{_digest(str(anchor_hash), request_id)}"

    def plan(
        self,
        agent_role: str,
        anchor_hash: Union[int, str],
        cla_group: Union[int, str],
        request_id: str,
        *,
        candidate_count: int = 1,
        reuse_rate: float = 0.0,
        layout_shuffled: bool = False,
    ) -> SaltPlan:
        """Decide the cache_salt for one agent request.

        Runs the REAL ``JCRSafetyGate.gate_decision(...)``. If the gate returns
        ``use_dense=True`` (INV-15 fired for a judge-class role), the request
        gets a UNIQUE isolated salt. Otherwise it gets the deterministic shared
        salt keyed by ``anchor_hash`` + ``cla_group``.

        Args:
            agent_role: Caller's role (drives the JCR gate; e.g. "critic").
            anchor_hash: The anchor's ``base_kv_hash`` from the AnchorPool
                (an int) or any stable string identifying the shared prefix.
            cla_group: Identifier of the CLA sharing group this request belongs
                to (so distinct CLA groups never collide on the shared path).
            request_id: Unique id for THIS request (used only on the isolated
                path; ignored on the shared path so sharing still works).
            candidate_count: Number of candidates the agent compares (JCR gate).
            reuse_rate: Block reuse rate the registry would apply (JCR gate).
            layout_shuffled: Whether candidate layout changed (JCR gate).

        Returns:
            A ``SaltPlan`` carrying the chosen salt and whether it is shared.
        """
        decision = self._gate.gate_decision(
            agent_role=agent_role,
            candidate_count=candidate_count,
            reuse_rate=reuse_rate,
            layout_shuffled=layout_shuffled,
        )

        if decision.use_dense:
            return SaltPlan(
                cache_salt=self.isolated_salt(anchor_hash, request_id),
                shared=False,
                reason=decision.reason,
            )

        return SaltPlan(
            cache_salt=self.shared_salt(anchor_hash, cla_group),
            shared=True,
            reason=decision.reason,
        )
