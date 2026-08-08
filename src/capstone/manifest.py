"""Model artifact manifests for inference"""

from pathlib import Path
from datetime import datetime, timezone

SCHEMA_VERSION = 1

def build_manifest(
        model_type: str,
        artifact_path: Path,
        feature_cols: list[str],
        hyperparameters: dict | None = None
) -> dict:
    manfiest = { 
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "artifact_path": Path(artifact_path).name,
        "feature_cols": feature_cols,     
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if hyperparameters is not None:
        manfiest["hyperparameters"] = hyperparameters
        
    return manfiest