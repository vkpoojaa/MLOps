"""Guard rail: dataset-specific values must live in YAML, not in src/."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _forbidden_literals(settings) -> set[str]:
    cfg = settings.config
    names = set(cfg.data_contract.columns)
    names |= {cfg.target, *cfg.feature_columns, *cfg.excluded_columns}
    for spec in cfg.data_contract.columns.values():
        names |= {v for v in (spec.allowed_values or []) if isinstance(v, str)}
    paths = {Path(p).name for p in (cfg.paths.train_data, cfg.paths.incoming_data)}
    return names | set(cfg.paths.model_dump().values()) | paths


def test_src_contains_no_dataset_literals(settings):
    forbidden = _forbidden_literals(settings)
    offenders = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for literal in forbidden:
            if re.search(rf"""["']{re.escape(literal)}["']""", text):
                offenders.append(f"{py.relative_to(SRC)}: {literal!r}")
    assert not offenders, "hard-coded config values found:\n" + "\n".join(offenders)
