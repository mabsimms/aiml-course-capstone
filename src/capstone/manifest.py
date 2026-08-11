"""Model artifact manifests for inference"""

from pathlib import Path
from datetime import datetime, timezone
from tokenizers import Tokenizer
import json
import logging

from capstone.tokenization.tokenization import save_tokenizer
from capstone.utils import get_git_info, get_machine_info

logger = logging.getLogger("capstone")

SCHEMA_VERSION = 1

def build_manifest(
        model_type: str,
        artifact_path: Path,
        feature_cols: list[str],
        hyperparameters: dict | None = None
) -> dict:
    manifest = { 
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "artifact_path": Path(artifact_path).name,
        "feature_cols": feature_cols,     
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if hyperparameters is not None:
        manifest["hyperparameters"] = hyperparameters
        
    return manifest

def save_training_artifacts(
        output: Path,
        model_type: str,
        feature_cols: list[str],
        tokenizer: Tokenizer,
        summary: dict,
        hyperparameters: dict | None = None
) -> None:
    if not "machine" in summary.keys():
        summary["machine"] = get_machine_info()
    if not "git" in summary.keys():
        summary["git"] = get_git_info()
    if not "timestamp" in summary.keys():
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved metrics summary to {metrics_path}")

    tokenizer_file = output.with_suffix(".tokenizer.json")
    save_tokenizer(tokenizer, tokenizer_file)
    logger.info(f"Saved tokenizer to {tokenizer_file}")

    manifest = build_manifest(model_type, output, feature_cols, hyperparameters=hyperparameters)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved model manifest to {manifest_path}")