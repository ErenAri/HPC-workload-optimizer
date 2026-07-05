"""HPC Workload Optimizer CLI – entry point and sub-app assembly."""

from __future__ import annotations

import logging

import typer

from hpcopt.cli.ingest import ingest_app
from hpcopt.cli.model import data_app, model_app, serve_app
from hpcopt.cli.pipeline import analysis_app, credibility_app, features_app, profile_app
from hpcopt.cli.report import artifacts_app, recommend_app, report_app
from hpcopt.cli.simulate import simulate_app, stress_app
from hpcopt.cli.train import train_app
from hpcopt.cli.whatif import whatif_app

app = typer.Typer(help="HPC Workload Optimizer CLI")

app.add_typer(ingest_app, name="ingest")
app.add_typer(profile_app, name="profile")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(simulate_app, name="simulate")
app.add_typer(stress_app, name="stress")
app.add_typer(recommend_app, name="recommend")
app.add_typer(report_app, name="report")
app.add_typer(serve_app, name="serve")
app.add_typer(data_app, name="data")
app.add_typer(credibility_app, name="credibility")
app.add_typer(analysis_app, name="analysis")
app.add_typer(model_app, name="model")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(whatif_app, name="whatif")


def run() -> None:
    from hpcopt.utils.logging import setup_logging

    try:
        setup_logging(level="INFO", format_mode="structured")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app()
