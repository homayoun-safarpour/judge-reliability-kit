# Contributing

Public repo: https://github.com/homayoun-safarpour/judge-reliability-kit

## Local loop

These are the commands CI runs.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Pull requests

1. One change per pull request.
2. A named pytest lock if you add a README claim.
3. Open a PR against `main`. Do not leave work only on a long-lived branch.
