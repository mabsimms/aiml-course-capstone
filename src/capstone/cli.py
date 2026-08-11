from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import pandas as pd
import typer
import json
import joblib
import logging

from enum import Enum

from capstone.dataset import prepare_experiment
from capstone.classic import build_classical_pipeline, classical_predict_proba, train_classical_model
from capstone.dnn import train_dnn, dnn_predict_proba
from capstone.dnn_tuner import tune_dnn
from capstone.utils import log_duration, get_machine_info, build_metrics_summary
from capstone.evaluate import evaluate_model
from capstone.gpu import configure_gpu
from capstone.manifest import build_manifest
from capstone.tokenization.tokenization import train_tokenizer

class LogLevel(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"

logger = logging.getLogger("capstone")

_LOG_LEVEL_MAP = { 
    LogLevel.info : logging.INFO,
    LogLevel.warning : logging.WARNING,
    LogLevel.error: logging.ERROR
}

_KERAS_VERBOSE_MAP = { 
    LogLevel.error: 0,
    LogLevel.warning: 2,
    LogLevel.info: 1
}

classic_app = typer.Typer()
dnn_app = typer.Typer()

app = typer.Typer()
app.add_typer(classic_app, name="classic")
app.add_typer(dnn_app, name="dnn")

def load_raw_data(
    file: Optional[Path], 
    directory: Optional[Path]
) -> dict[str, pd.DataFrame]:
    if (file is None) == (directory is None):
        raise typer.BadParameter("Specify either --file or --directory")
    if file is not None:
        return {file.stem : pd.read_csv(file)}

    return {
        csv_file.stem : pd.read_csv(csv_file) for csv_file in directory.glob("*.csv")
    }

def resolve_training_data(
        file: Optional[Path], 
        directory: Optional[Path]        
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = load_raw_data(file, directory)
    df_train, df_test, feature_cols = prepare_experiment(raw, stratify_by_source=True)
    return df_train, df_test, feature_cols

@classic_app.command("train")
def classic_train(
    config: Path = typer.Option(..., exists=True, help="JSON hyperparameter configuration"),
    file: Optional[Path] = typer.Option(None, help="Single raw source CSV"),
    directory: Optional[Path] = typer.Option(None, help="Directory of raw per-source CSVs"),
    output: Path = typer.Option(..., help="Target file for the trained model (.joblib)")
):
    df_train, df_test, feature_cols = resolve_training_data(file, directory)

    hyperparams = json.loads(config.read_text())
    if "features__tfidf__ngram_range" in hyperparams:
        hyperparams["features__tfidf__ngram_range"] = tuple(hyperparams["features__tfidf__ngram_range"])
    tokenizer = train_tokenizer(df_train["text"])

    pipeline = build_classical_pipeline(feature_cols, tokenizer)
    pipeline.set_params(**hyperparams)

    with log_duration("Training classic model") as timing:
        pipeline.fit(df_train, df_train["Label"])

    predict_proba_fn = classical_predict_proba(pipeline)
    metrics = evaluate_model(predict_proba_fn, df_test, "Label")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    logger.info(f"Saved trained model to {output}")

    summary = { 
        "operation": "classic train",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timing["seconds"],
        "machine": get_machine_info(),
        "input_data": { 
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "num_features": len(feature_cols)
        },
        "hyperparameters": hyperparams,
        "metrics": build_metrics_summary(metrics)
    }

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved metrics summary to {metrics_path}")

    tokenizer_file = output.with_suffix(".tokenizer.json")
    tokenizer.save(str(tokenizer_file))
    logger.info(f"Saved tokenizer to {tokenizer_file}")

    manifest = build_manifest("classic", output, feature_cols)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved model manifest to {manifest_path}")

@dnn_app.command("train")
def dnn_train(
    ctx: typer.Context,
    config: Path = typer.Option(..., exists=True, help="JSON hyperparameter configuration"),
    file: Optional[Path] = typer.Option(None, help="Single raw source CSV"),
    directory: Optional[Path] = typer.Option(None, help="Directory of raw per-source CSVs"),
    output: Path = typer.Option(..., help="Target file template for the trained model artifacts"),
    epochs: int = typer.Option(15, help="Maximum training runs (epochs)")
):
    df_train, df_test, feature_cols = resolve_training_data(file, directory)
    hyperparams = json.loads(config.read_text())
    
    verbose = _KERAS_VERBOSE_MAP[ctx.obj]
    with log_duration("DNN training") as timing:
        model, history, stop, tokenizer = train_dnn(df_train, feature_cols, hyperparams=hyperparams, epochs=epochs, verbose=verbose)

    predict_proba_fn = dnn_predict_proba(model, tokenizer, feature_cols)
    metrics = evaluate_model(predict_proba_fn, df_test, "Label")

    output.parent.mkdir(parents=True, exist_ok=True)
    weights_file = output.with_suffix(".weights.h5")
    model.save_weights(weights_file)
    logger.info(f"Saved model weights summary to {weights_file}")
  
    tokenizer_file = output.with_suffix(".tokenizer.json")
    tokenizer.save(str(tokenizer_file))
    logger.info(f"Saved tokenizer to {tokenizer_file}")
             
    manifest = build_manifest("dnn", output, feature_cols, hyperparameters=hyperparams)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved model manifest to {manifest_path}")

    summary = { 
        "operation": "dnn train",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timing["seconds"],
        "machine": get_machine_info(),
        "input_data": { 
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "num_features": len(feature_cols)
        },
        "hyperparameters": hyperparams,
        "training": { 
            "epochs_req" : epochs,
            "epochs_run" : len(history.epoch),                 
            "epochs_stopped" : stop['stopped_epoch'],
        },
        "metrics": build_metrics_summary(metrics)
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved metrics summary to {metrics_path}")
        
    logger.info(f"Saved trained model to {output}")
    
@classic_app.command("tune")
def classic_tune(
    ctx: typer.Context,
    config: Path = typer.Option(..., exists=True, help="JSON search space"),
    file: Optional[Path] = typer.Option(None, help="Single raw source CSV"),
    directory: Optional[Path] = typer.Option(None, help="Directory of raw per-source CSVs"),
    output: Path = typer.Option(..., help="Target file for the trained model (.joblib)"),
    cv : int = typer.Option(5, help="Number of cross validation folds"),
    n_jobs : int = typer.Option(-1, help="Parallel jobs for GridSearchCV")
):
    df_train, df_test, feature_cols = resolve_training_data(file, directory)
  
    search_space = json.loads(config.read_text())
    if "features__tfidf__ngram_range" in search_space:
        search_space["features__tfidf__ngram_range"] = [
            tuple(r) for r in search_space["features__tfidf__ngram_range"]
        ]

    tokenizer = train_tokenizer(df_train["text"])

    with log_duration("Tuning classic model") as timing:
        grid = train_classical_model(
            df_train=df_train,
            feature_cols=feature_cols,
            tokenizer=tokenizer,
            param_grid=search_space,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )

    pipeline = grid.best_estimator_
    predict_proba_fn = classical_predict_proba(pipeline)
    metrics = evaluate_model(predict_proba_fn, df_test, "Label")


    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    logger.info(f"Saved trained model to {output}")

    summary = { 
        "operation": "classic train",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timing["seconds"],
        "machine": get_machine_info(),
        "input_data": { 
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "num_features": len(feature_cols)
        },
        "search_space": search_space,
        "cv_folds": cv,
        "best_params": grid.best_params_,
        "best_cv_f1": grid.best_score_,
        "metrics": build_metrics_summary(metrics)
    }

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved metrics summary to {metrics_path}")

    tokenizer_file = output.with_suffix(".tokenizer.json")
    tokenizer.save(str(tokenizer_file))
    logger.info(f"Saved tokenizer to {tokenizer_file}")

    manifest = build_manifest("classic", output, feature_cols)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved model manifest to {manifest_path}")



  

@dnn_app.command("tune")
def dnn_tune(
    ctx: typer.Context,
    config: Path = typer.Option(..., exists=True, help="JSON search space"),
    file: Optional[Path] = typer.Option(None, help="Single raw source CSV"),
    directory: Optional[Path] = typer.Option(None, help="Directory of raw per-source CSVs"),
    output: Path = typer.Option(..., help="Target file for the trained model (.joblib)"),
    max_trials: int = typer.Option(15),
    epochs: int = typer.Option(15, help="Maximum training runs (epochs)"),
    tuner_dir : Path = typer.Option("artifacts/tuner"),
    project_name : str = typer.Option("dnn_search"),
    overwrite : bool = typer.Option(False, "--overwrite", help="Start a fresh search, discarding any existing trials in --tuner-dir")
):
    with log_duration("dnn tune") as timing:
        df_train, df_test, feature_cols = resolve_training_data(file, directory)
        search_space = json.loads(config.read_text())
        verbose = _KERAS_VERBOSE_MAP[ctx.obj]

        result = tune_dnn(
            df_train=df_train,
            feature_cols=feature_cols,
            search_space=search_space,
            fixed_hyperparameters={
                # Use CuDNN where available (5x speed increase over raw GPU)
                "use_cudnn": "auto",
                # Optimal tokenizer configuration on this data set (from parametric sweep)
                "max_tokens": 40_000,
                # Optimal tokenizer configuration on this data set (from parametric sweep)
                "output_sequence_length": 6_000
            },
            max_trials=max_trials,
            epochs=epochs,
            tuner_dir=tuner_dir,
            project_name=project_name,
            overwrite=overwrite,
            verbose=verbose
        )
                 
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result["hyperparameters"], indent=2))
        logger.info("Saved best hyperparameters to {output}")

        summary = { 
            "operation": "dnn train",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": timing["seconds"],
            "machine": get_machine_info(),
            "input_data": { 
                "train_rows": len(df_train),
                "test_rows": len(df_test),
                "num_features": len(feature_cols)
            },
            "search_space": search_space,
            "max_trials": max_trials,
            "epochs_per_trial": epochs,
            "trials_completed": result["trials_completed"],
            "best_val_loss": result["best_val_loss"],
            "best_hyperparameters": result["hyperparameters"]
        }
        summary_path = output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved metrics summary to {summary_path}")


@app.callback()
def main(
    ctx: typer.Context,
    verbose : LogLevel = typer.Option(LogLevel.warning, "--verbose", help="Logging level (info, warning or error)")
):
    logging.basicConfig(level=_LOG_LEVEL_MAP[verbose], format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    ctx.obj = verbose

    detected_gpus = configure_gpu()
    logger.info("GPU configuration: %s", detected_gpus)
    
if __name__ == "__main__":
    app()
