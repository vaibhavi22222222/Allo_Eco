from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ingestion import DatasetIngestionError, ingest_all
from storage.repository import default_db_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and transactionally ingest CarbonWise CSV data into SQLite."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "synthetic_data" / "main",
        help="Directory containing all four Annex B CSV files.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=default_db_path(),
        help="Persistent SQLite database path (default can be set via CARBONWISE_DB_PATH).",
    )
    parser.add_argument(
        "--planning-start",
        help="Optional ISO date; if provided, orders due before this date are rejected.",
    )
    args = parser.parse_args()

    try:
        result = ingest_all(args.data_dir, args.db_path, args.planning_start)
    except DatasetIngestionError as exc:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "dataset_version_id": exc.dataset_version_id,
                    "validation_report": exc.report,
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "accepted": True,
                "dataset_version_id": result["dataset_version_id"],
                "row_counts": result["row_counts"],
                "validation_report": result["validation_report"],
                "database_path": str(args.db_path.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
