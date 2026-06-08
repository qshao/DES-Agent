from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXAMPLES_ROOT = Path("examples")
BASELINE_ROOT = Path(__file__).resolve().parent / "example_benchmark_baselines"
WARNING_MARKERS = (
    "DeprecationWarning: torch_geometric.distributed",
    "DeprecationWarning: `torch.jit.script` is deprecated.",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    example_dir: Path
    baseline_dir: Path

    @property
    def example_input(self) -> Path:
        return self.example_dir / "input.txt"

    @property
    def example_output(self) -> Path:
        return self.example_dir / "output.txt"

    @property
    def example_readme(self) -> Path:
        return self.example_dir / "README.md"

    @property
    def baseline_input(self) -> Path:
        return self.baseline_dir / "input.txt"

    @property
    def baseline_output(self) -> Path:
        return self.baseline_dir / "output.txt"

    @property
    def baseline_readme(self) -> Path:
        return self.baseline_dir / "README.md"


def normalize_benchmark_output(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] == "":
                continue
            lines.append("")
            continue
        if any(marker in line for marker in WARNING_MARKERS):
            continue
        lines.append(line)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def iter_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for example_dir in sorted(EXAMPLES_ROOT.iterdir()):
        if not example_dir.is_dir():
            continue
        if not (example_dir / "input.txt").exists():
            continue
        if not (example_dir / "output.txt").exists():
            continue
        if not (example_dir / "README.md").exists():
            continue
        baseline_dir = BASELINE_ROOT / example_dir.name
        if not baseline_dir.exists():
            continue
        cases.append(BenchmarkCase(example_dir.name, example_dir, baseline_dir))
    return tuple(cases)
