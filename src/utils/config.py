"""
src/utils/config.py

One resolved location for config.yaml, and one loader.

Every module used to carry its own copy of

    def _load_config(path: str = "config/config.yaml"):
        with open(path) as f:
            return yaml.safe_load(f)

which is eight copies of the same three lines, and -- more importantly --
a relative path. That only resolves if the process happens to have been
started from the repository root, so `streamlit run dashboard/app.py`
worked while `cd dashboard && streamlit run app.py` did not, and the
notebooks had to reach for absolute paths like /home/<user>/EquiRisk/...
to work at all.

Resolving against __file__ instead makes the default correct regardless
of the working directory, on any machine.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path=None) -> dict:
    """Load config.yaml. Pass `path` to point at an alternative file
    (a test fixture, a prod override); omit it for the repo default."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. Expected it at "
            f"{DEFAULT_CONFIG_PATH} relative to the repository root."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
