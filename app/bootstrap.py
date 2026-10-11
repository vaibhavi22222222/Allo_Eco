"""The ONE place that makes the data module and optimizer importable.

Why this file exists
--------------------
Vibhu's optimizer and the data module live in ``carbonwise-data/`` and import
each other as top-level packages (``optimization``, ``services``,
``storage``). Their own scripts and tests add ``carbonwise-data`` to
``sys.path``. The folder name contains a hyphen, so it cannot be imported as a
normal Python package.

Instead of scattering ``sys.path`` edits across the backend, every
application module that needs the engine does::

    from app.bootstrap import ENGINE_DIR   # noqa: F401  (side effect: path set up)

and then imports ``optimization`` / ``services`` / ``storage`` normally. The
path is added only once, and the existing tests keep working because they use
the same top-level names.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root (the folder that contains ``app/`` and ``carbonwise-data/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Folder holding the data module and Vibhu's optimizer.
ENGINE_DIR: Path = PROJECT_ROOT / "carbonwise-data"

#: Vibhu's planning configuration (energy prices, shared capacities, horizon).
DEFAULT_PLANNING_CONFIG: Path = ENGINE_DIR / "config" / "optimizer.yaml"

#: The team's synthetic data pack, used to seed an empty database.
DEFAULT_SEED_DATA_DIR: Path = ENGINE_DIR / "synthetic_data" / "main"

if not ENGINE_DIR.is_dir():  # fail loudly: nothing else can work without it
    raise ImportError(f"Engine folder not found: {ENGINE_DIR}")

if str(ENGINE_DIR) not in sys.path:
    # Append (not insert at 0) so the application's own ``app`` package and the
    # standard library always win over any same-named module in the engine.
    sys.path.append(str(ENGINE_DIR))
