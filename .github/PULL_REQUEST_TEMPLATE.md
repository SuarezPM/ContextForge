## Summary

<!-- What does this PR do, in 1-2 sentences -->

## Motivation

<!-- Why is this change needed; link issue/discussion if applicable -->

## Honesty checklist

- [ ] All claims in code, docstrings, and documentation reflect what the code actually does at runtime
- [ ] Flags introduced (if any) reflect runtime state, not intent (V6.1 honesty discipline)
- [ ] `AUDIT.md` updated if a new module is introduced or a module's status changes
- [ ] No hardcoded metrics, fabricated benchmark numbers, or simulated values in production code
- [ ] `scripts/check_honesty.sh` runs locally without errors

## Testing

- [ ] All existing tests pass (`PYTHONPATH=. pytest tests/ -v`)
- [ ] New tests added for new code paths (or explicit justification if not)
- [ ] Tested on Python 3.11+ (3.14 is the development default)

## DCO

- [ ] All commits in this PR are signed off (`git commit -s`)
