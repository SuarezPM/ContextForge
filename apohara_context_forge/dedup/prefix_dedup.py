"""PrefixDedup — lightweight token-count + shared-prefix helper.

The CompressionCoordinator needs two cheap operations on raw context
strings: count a context's tokens, and find the longest shared prefix
between two contexts. This is the production default for
``ContextRegistry.dedup``; tests inject their own fake with the same
contract (``count_prefix_tokens``, ``find_shared_prefix``).
"""
from typing import Optional

from apohara_context_forge.token_counter import TokenCounter


class PrefixDedup:
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._tc = token_counter or TokenCounter.get()

    def count_prefix_tokens(self, prefix: str) -> int:
        return self._tc.count(prefix)

    def find_shared_prefix(self, a: str, b: str) -> str:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        if i == n:
            return a[:i]
        # Back off to the last word boundary so we don't split a token.
        j = a.rfind(" ", 0, i)
        return a[:j] if j > 0 else a[:i]
