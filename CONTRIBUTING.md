# Contributing to HNEP

Thanks for your interest in making HNEP better.

HNEP is an alpha-stage library: APIs are still moving, and bug reports +
feature suggestions are especially welcome.

## Quick links
- 🐛 Bug report: open a GitHub issue with the `bug` label.
- 💡 Feature suggestion: open a GitHub issue with the `enhancement` label.
- 📖 Documentation fix: a PR is faster than an issue.
- 🤔 Methodology question: open a GitHub discussion rather than an issue.

## Setting up a dev environment

```bash
git clone https://github.com/Pratik25priyanshu20/HNEP.git
cd hnep
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

## Style + tests

- Format with **ruff** (`ruff format hnep tests`) — line length 100.
- Lint with **ruff** (`ruff check hnep tests`).
- All new functionality needs at least one test in `tests/`.
- For numerical results, prefer `pytest.approx` with explicit tolerances.

## Writing a new probe

A probe is a subclass of `hnep.Probe` that implements
`.run(model, dataset, verbose) -> ProbeResult`. The `ProbeResult` dataclass
is the uniform return format — see `hnep/examples/05_custom_probe.py` for
a worked example.

## Writing a new adapter

Adapters subclass `hnep.ModelInterface` and implement three methods:
`predict`, `extract_quantum_output`, `predict_with_quantum_override`.
Two optional hooks (`get_classical_embedding`, `get_quantum_input`) unlock
richer probes — implement them when you can.

## Submitting a PR

1. Fork the repo and create a feature branch.
2. Make the change. Add tests.
3. Run `pytest tests/`, `ruff check hnep tests`, and `ruff format --check
   hnep tests`.
4. Update the `CHANGELOG.md` under `Unreleased`.
5. Open a PR. Small PRs review faster than large ones.

## Code of conduct

Be excellent to each other. The discussion norms are: ask before
assuming, criticise ideas not people, and assume good faith.
