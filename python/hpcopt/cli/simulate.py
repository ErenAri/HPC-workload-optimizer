"""CLI commands for simulation, replay, batsim, and stress testing."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import typer
import yaml

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.data.reference_suite import assert_reference_by_filename_and_hash
from hpcopt.models.runtime_quantile import (
    RuntimeQuantilePredictor,
    resolve_runtime_model_dir,
)
from hpcopt.simulate.batsim import (
    SUPPORTED_EDC_MODES,
    build_batsim_run_config,
    invoke_batsim_run,
    normalize_batsim_run_outputs,
)
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace
from hpcopt.simulate.fidelity import run_baseline_fidelity_gate, run_candidate_fidelity_report
from hpcopt.simulate.objective import evaluate_constraint_contract
from hpcopt.simulate.stress import generate_stress_scenario
from hpcopt.utils.io import ensure_dir, write_json

simulate_app = typer.Typer(help="Simulation commands")
stress_app = typer.Typer(help="Stress scenario commands")


# ──────────────────────── Simulate ────────────────────────


@simulate_app.command("run")
def simulate_run_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    policy: str = typer.Option(
        "FIFO_STRICT",
        help="FIFO_STRICT|EASY_BACKFILL_BASELINE|ML_BACKFILL_P50|ML_BACKFILL_P10",
    ),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    out: Path = typer.Option(Path("outputs/simulations"), help="Simulation artifact output directory"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    run_id: str | None = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(False, help="Fail on first invariant violation"),
    runtime_model_dir: Path | None = typer.Option(None, help="Runtime quantile model directory"),
    runtime_guard_k: float = typer.Option(0.5, min=0.0, max=2.0, help="ML backfill runtime guard coefficient"),
    strict_uncertainty_mode: bool = typer.Option(False, help="ML policy strict mode"),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
        help="Reference suite config for trace hash checks",
    ),
) -> None:
    if policy not in SUPPORTED_POLICIES:
        from hpcopt import plugins

        if not plugins.is_registered(policy):
            raise typer.BadParameter(
                f"Unsupported policy '{policy}'. Available: {list(plugins.all_policy_ids())}"
            )
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"sim_{policy.lower()}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)

    runtime_predictor = None
    resolved_model_dir = None
    if policy in ("ML_BACKFILL_P50", "ML_BACKFILL_P10"):
        resolved_model_dir = resolve_runtime_model_dir(runtime_model_dir)
        if resolved_model_dir is not None:
            runtime_predictor = RuntimeQuantilePredictor(resolved_model_dir)

    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=policy,
        capacity_cpus=capacity_cpus,
        run_id=resolved_run_id,
        strict_invariants=strict_invariants,
        runtime_predictor=runtime_predictor,
        runtime_guard_k=runtime_guard_k,
        strict_uncertainty_mode=strict_uncertainty_mode,
    )

    jobs_path = out / f"{resolved_run_id}_{policy.lower()}_jobs.parquet"
    queue_path = out / f"{resolved_run_id}_{policy.lower()}_queue.parquet"
    sim_report_path = report_out / f"{resolved_run_id}_{policy.lower()}_sim_report.json"
    invariant_path = report_out / f"{resolved_run_id}_{policy.lower()}_invariants.json"

    result.jobs_df.to_parquet(jobs_path, index=False)
    result.queue_series_df.to_parquet(queue_path, index=False)
    source_reference_match = None
    trace_metadata_path = trace.with_suffix(".metadata.json")
    if trace_metadata_path.exists():
        trace_meta = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        source_reference_match = assert_reference_by_filename_and_hash(
            filename=str(trace_meta.get("source_trace_filename", "")),
            sha256_observed=trace_meta.get("source_trace_sha256"),
            config_path=reference_suite_config,
        )
    write_json(
        sim_report_path,
        {
            "run_id": resolved_run_id,
            "policy_id": policy,
            "status": "ok",
            "metrics": result.metrics,
            "objective_metrics": result.objective_metrics,
            "fallback_accounting": result.fallback_accounting,
            "model_dir": str(resolved_model_dir) if resolved_model_dir is not None else None,
            "source_trace_reference_suite": source_reference_match,
            "jobs_artifact": str(jobs_path),
            "queue_artifact": str(queue_path),
        },
    )
    write_json(invariant_path, result.invariant_report)

    manifest = build_manifest(
        command="hpcopt simulate run",
        inputs=[trace, reference_suite_config],
        outputs=[jobs_path, queue_path, sim_report_path, invariant_path],
        params={
            "run_id": resolved_run_id,
            "policy_id": policy,
            "capacity_cpus": capacity_cpus,
            "strict_invariants": strict_invariants,
            "runtime_model_dir": str(resolved_model_dir) if resolved_model_dir else None,
            "runtime_guard_k": runtime_guard_k,
        },
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_{policy.lower()}_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Simulation report: {sim_report_path}")
    typer.echo(f"Manifest: {manifest_path}")


@simulate_app.command("fidelity-gate")
def simulate_fidelity_gate_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    config: Path = typer.Option(
        Path("configs/simulation/fidelity_gate.yaml"),
        exists=True,
        readable=True,
        help="Fidelity gate threshold config",
    ),
    out: Path = typer.Option(Path("outputs/reports"), help="Fidelity report output directory"),
    run_id: str | None = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(True, help="Fail if invariants fail"),
) -> None:
    ensure_dir(out)
    resolved_run_id = run_id or f"fidelity_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    report_path = out / f"{resolved_run_id}_fidelity_report.json"
    result = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=capacity_cpus,
        out_path=report_path,
        run_id=resolved_run_id,
        config_path=config,
        strict_invariants=strict_invariants,
    )
    manifest = build_manifest(
        command="hpcopt simulate fidelity-gate",
        inputs=[trace, config],
        outputs=[result.report_path],
        params={"run_id": resolved_run_id, "capacity_cpus": capacity_cpus},
        config_paths=[config],
        seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_fidelity_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Fidelity status: {result.status}")
    typer.echo(f"Fidelity report: {result.report_path}")


@simulate_app.command("batsim-config")
def simulate_batsim_config_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    policy: str = typer.Option("FIFO_STRICT"),
    out: Path = typer.Option(Path("outputs/simulations")),
    run_id: str | None = typer.Option(None),
    platform_path: Path | None = typer.Option(None),
    workload_path: Path | None = typer.Option(None),
    capacity_cpus: int | None = typer.Option(None, min=1),
    edc_mode: str = typer.Option("library_file"),
    edc_library_path: str | None = typer.Option(None),
    edc_socket_endpoint: str | None = typer.Option(None),
    edc_init_json: str = typer.Option("{}"),
    export_prefix: Path | None = typer.Option(None),
    use_wsl_defaults: bool = typer.Option(True),
    wsl_distro: str = typer.Option("Ubuntu"),
    report_out: Path = typer.Option(Path("outputs/reports")),
) -> None:
    if edc_mode not in SUPPORTED_EDC_MODES:
        raise typer.BadParameter(f"Unsupported edc_mode '{edc_mode}'")
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"batsim_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    config = build_batsim_run_config(
        run_id=resolved_run_id,
        trace_dataset=trace,
        policy_id=policy,
        out_dir=out,
        platform_path=platform_path,
        workload_path=workload_path,
        capacity_cpus=capacity_cpus,
        edc_mode=edc_mode,
        edc_library_path=edc_library_path,
        edc_socket_endpoint=edc_socket_endpoint,
        edc_init_json=edc_init_json,
        export_prefix=export_prefix,
        use_wsl_defaults=use_wsl_defaults,
        wsl_distro=wsl_distro,
    )
    typer.echo(f"Batsim config: {config.config_path}")


@simulate_app.command("batsim-run")
def simulate_batsim_run_cmd(
    config: Path = typer.Option(..., exists=True, readable=True),
    batsim_bin: str = typer.Option("batsim"),
    dry_run: bool = typer.Option(True),
    use_wsl: bool = typer.Option(False),
    wsl_distro: str = typer.Option("Ubuntu"),
    wsl_load_nix_profile: bool = typer.Option(True),
    normalize_to_sim_report: bool = typer.Option(True),
    simulation_out: Path = typer.Option(Path("outputs/simulations")),
    emit_fidelity_report: bool = typer.Option(True),
    fidelity_config: Path | None = typer.Option(Path("configs/simulation/fidelity_gate.yaml")),
    out: Path = typer.Option(Path("outputs/reports")),
) -> None:
    ensure_dir(out)
    ensure_dir(simulation_out)
    result = invoke_batsim_run(
        config_path=config,
        batsim_bin=batsim_bin,
        dry_run=dry_run,
        use_wsl=use_wsl,
        wsl_distro=wsl_distro,
        wsl_load_nix_profile=wsl_load_nix_profile,
    )
    report_payload: dict[str, object] = {
        "config_path": str(config),
        "status": result.status,
        "reason": result.reason,
        "returncode": result.returncode,
        "command": result.command,
    }
    if result.status == "ok" and not dry_run and normalize_to_sim_report:
        normalization = normalize_batsim_run_outputs(
            config_path=config,
            report_out_dir=out,
            simulation_out_dir=simulation_out,
        )
        report_payload["normalization"] = {
            "sim_report": str(normalization.sim_report_path),
            "jobs_artifact": str(normalization.jobs_artifact_path),
            "queue_artifact": str(normalization.queue_artifact_path),
            "invariant_report": str(normalization.invariant_report_path),
            "metrics": normalization.metrics,
        }
        typer.echo(f"Normalized sim report: {normalization.sim_report_path}")
        if emit_fidelity_report:
            config_payload = json.loads(config.read_text(encoding="utf-8"))
            source_trace = config_payload.get("workload", {}).get("source_trace")
            if source_trace and Path(source_trace).suffix.lower() == ".parquet" and Path(source_trace).exists():
                fidelity_result = run_candidate_fidelity_report(
                    trace_df=pd.read_parquet(source_trace),
                    simulated_jobs=pd.read_parquet(normalization.jobs_artifact_path),
                    simulated_queue=pd.read_parquet(normalization.queue_artifact_path),
                    capacity_cpus=int(config_payload.get("resources", {}).get("capacity_cpus", 1)),
                    out_path=out / f"{config.stem}_batsim_fidelity_report.json",
                    run_id=normalization.run_id,
                    policy_id=normalization.policy_id,
                    config_path=fidelity_config if fidelity_config and fidelity_config.exists() else None,
                )
                report_payload["fidelity"] = {
                    "status": fidelity_result.status,
                    "report": str(fidelity_result.report_path),
                }
                typer.echo(f"Batsim fidelity status: {fidelity_result.status}")
    report_path = out / f"{config.stem}_batsim_run_report.json"
    write_json(report_path, report_payload)
    typer.echo(f"Batsim run status: {result.status}")
    typer.echo(f"Run report: {report_path}")


@simulate_app.command("replay-baselines")
def simulate_replay_baselines_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    capacity_cpus: int = typer.Option(64, min=1),
    out: Path = typer.Option(Path("outputs/simulations")),
    report_out: Path = typer.Option(Path("outputs/reports")),
    run_id: str | None = typer.Option(None),
    strict_invariants: bool = typer.Option(True),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
    ),
) -> None:
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"baseline_replay_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    policies = ("FIFO_STRICT", "EASY_BACKFILL_BASELINE")
    combined: dict[str, dict[str, object]] = {}
    outputs: list[Path] = []

    for policy in policies:
        sim = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id=policy,
            capacity_cpus=capacity_cpus,
            run_id=f"{resolved_run_id}_{policy.lower()}",
            strict_invariants=strict_invariants,
        )
        jobs_path = out / f"{resolved_run_id}_{policy.lower()}_jobs.parquet"
        queue_path = out / f"{resolved_run_id}_{policy.lower()}_queue.parquet"
        inv_path = report_out / f"{resolved_run_id}_{policy.lower()}_invariants.json"
        sim.jobs_df.to_parquet(jobs_path, index=False)
        sim.queue_series_df.to_parquet(queue_path, index=False)
        write_json(inv_path, sim.invariant_report)
        outputs.extend([jobs_path, queue_path, inv_path])
        combined[policy] = {
            "metrics": sim.metrics,
            "objective_metrics": sim.objective_metrics,
            "fallback_accounting": sim.fallback_accounting,
        }

    summary_path = report_out / f"{resolved_run_id}_baseline_replay_report.json"
    write_json(
        summary_path,
        {
            "run_id": resolved_run_id,
            "trace": str(trace),
            "capacity_cpus": capacity_cpus,
            "policies": combined,
        },
    )
    manifest = build_manifest(
        command="hpcopt simulate replay-baselines",
        inputs=[trace, reference_suite_config],
        outputs=outputs + [summary_path],
        params={"run_id": resolved_run_id, "capacity_cpus": capacity_cpus},
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_baseline_replay_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Baseline replay report: {summary_path}")


# ──────────────────────── Stress ────────────────────────


@stress_app.command("gen")
def stress_gen_cmd(
    scenario: str = typer.Option(..., help="heavy_tail|low_congestion|user_skew|burst_shock"),
    out: Path = typer.Option(Path("data/curated")),
    n_jobs: int = typer.Option(5000, min=100),
    seed: int = typer.Option(42),
    alpha: float = typer.Option(1.2),
    target_util: float = typer.Option(0.35),
    top_user_share: float = typer.Option(0.65),
    burst_factor: int = typer.Option(4),
    burst_duration_sec: int = typer.Option(1800),
) -> None:
    params = {
        "alpha": alpha,
        "target_util": target_util,
        "top_user_share": top_user_share,
        "burst_factor": burst_factor,
        "burst_duration_sec": burst_duration_sec,
    }
    result = generate_stress_scenario(scenario=scenario, out_dir=out, n_jobs=n_jobs, seed=seed, params=params)
    typer.echo(f"Stress dataset: {result.dataset_path}")


@stress_app.command("run")
def stress_run_cmd(
    scenario: str = typer.Option(...),
    policy: Path = typer.Option(..., exists=True, readable=True),
    model: str = typer.Option(...),
    dataset: Path | None = typer.Option(None),
    capacity_cpus: int = typer.Option(64, min=1),
    baseline_policy: str = typer.Option("EASY_BACKFILL_BASELINE"),
    out: Path = typer.Option(Path("outputs/simulations")),
    report_out: Path = typer.Option(Path("outputs/reports")),
    run_id: str | None = typer.Option(None),
    strict_invariants: bool = typer.Option(True),
    runtime_model_dir: Path | None = typer.Option(None),
) -> None:
    if baseline_policy not in SUPPORTED_POLICIES:
        raise typer.BadParameter(f"Unsupported baseline policy '{baseline_policy}'")
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_dataset = dataset or (Path("data/curated") / f"stress_{scenario}.parquet")
    if not resolved_dataset.exists():
        raise typer.BadParameter(f"Stress dataset does not exist: {resolved_dataset}")

    cfg = yaml.safe_load(policy.read_text(encoding="utf-8"))
    policy_id = str(cfg.get("policy_id", "ML_BACKFILL_P50"))
    runtime_guard_k = float(cfg.get("runtime_guard_k", 0.5))
    resolved_run_id = (
        run_id or f"stress_{scenario}_{policy_id.lower()}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    trace_df = pd.read_parquet(resolved_dataset)

    runtime_predictor = None
    resolved_model_dir = None
    if policy_id in ("ML_BACKFILL_P50", "ML_BACKFILL_P10"):
        resolved_model_dir = resolve_runtime_model_dir(runtime_model_dir)
        if resolved_model_dir is not None:
            runtime_predictor = RuntimeQuantilePredictor(resolved_model_dir)

    baseline_sim = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=baseline_policy,
        capacity_cpus=capacity_cpus,
        run_id=f"{resolved_run_id}_{baseline_policy.lower()}",
        strict_invariants=strict_invariants,
    )
    candidate_sim = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=policy_id,
        capacity_cpus=capacity_cpus,
        run_id=f"{resolved_run_id}_{policy_id.lower()}",
        strict_invariants=strict_invariants,
        runtime_predictor=runtime_predictor,
        runtime_guard_k=runtime_guard_k,
    )
    fairness_cfg = cfg.get("fairness", {})
    constraints = evaluate_constraint_contract(
        candidate=candidate_sim.objective_metrics,
        baseline=baseline_sim.objective_metrics,
        starvation_rate_max=float(fairness_cfg.get("starvation_rate_max", 0.02)),
        fairness_dev_delta_max=float(fairness_cfg.get("fairness_dev_delta_max", 0.05)),
        jain_delta_max=float(fairness_cfg.get("jain_delta_max", 0.03)),
    )
    stress_status = "pass" if constraints["constraints_passed"] else "fail"

    degrade_signatures: dict[str, object] = {}
    for key in candidate_sim.objective_metrics:
        cval = candidate_sim.objective_metrics.get(key)
        bval = baseline_sim.objective_metrics.get(key)
        if isinstance(cval, (int, float)) and isinstance(bval, (int, float)) and bval != 0:
            delta = cval - bval
            ratio = delta / abs(bval)
            degrade_signatures[key] = {"candidate": cval, "baseline": bval, "delta": delta, "ratio": round(ratio, 6)}

    stress_report_path = report_out / f"{resolved_run_id}_stress_report.json"
    write_json(
        stress_report_path,
        {
            "run_id": resolved_run_id,
            "scenario": scenario,
            "status": stress_status,
            "constraints": constraints,
            "candidate_policy_id": policy_id,
            "baseline_policy_id": baseline_policy,
            "degrade_signatures": degrade_signatures,
        },
    )

    manifest = build_manifest(
        command="hpcopt stress run",
        inputs=[resolved_dataset, policy],
        outputs=[stress_report_path],
        params={
            "run_id": resolved_run_id,
            "scenario": scenario,
            "policy_id": policy_id,
            "baseline_policy": baseline_policy,
            "capacity_cpus": capacity_cpus,
        },
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_stress_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Stress status: {stress_status}")
    typer.echo(f"Stress report: {stress_report_path}")
    typer.echo(f"Stress manifest: {manifest_path}")
