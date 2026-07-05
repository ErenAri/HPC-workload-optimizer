use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

// ── CLI ────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "High-performance discrete-event HPC scheduling simulator"
)]
struct Args {
    /// Input file: JSON array of canonical job records.
    #[arg(long)]
    input: PathBuf,

    /// Scheduling policy: FIFO_STRICT | EASY_BACKFILL_BASELINE
    #[arg(long, default_value = "FIFO_STRICT")]
    policy: String,

    /// Total CPU capacity of the simulated cluster.
    #[arg(long, default_value_t = 64)]
    capacity_cpus: u32,

    /// Fail immediately on invariant violations.
    #[arg(long, default_value_t = false)]
    strict_invariants: bool,

    /// Output report path. Omit to print JSON to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Data structures ────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct Job {
    job_id: u64,
    submit_ts: i64,
    runtime_actual_sec: i64,
    requested_cpus: u32,
}

#[derive(Debug, Clone)]
struct RunningJob {
    job_id: u64,
    submit_ts: i64,
    start_ts: i64,
    end_ts: i64,
    requested_cpus: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CompletedJob {
    job_id: u64,
    submit_ts: i64,
    start_ts: i64,
    end_ts: i64,
    requested_cpus: u32,
    wait_sec: i64,
}

// ── Report structures ──────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Metrics {
    policy_id: String,
    capacity_cpus: u32,
    jobs_total: usize,
    jobs_completed: usize,
    mean_wait_sec: f64,
    p95_wait_sec: f64,
    p95_bsld: f64,
    utilization_mean: f64,
    makespan_sec: i64,
    invariant_violations: usize,
}

#[derive(Debug, Serialize)]
struct InvariantReport {
    strict: bool,
    total_violations: usize,
    examples: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SimulationReport {
    policy_id: String,
    run_id: String,
    metrics: Metrics,
    invariant_report: InvariantReport,
}

// ── Utility functions ──────────────────────────────────────────

/// Linear-interpolation percentile, matching numpy.quantile's default method
/// used by the Python reference engine (metric parity contract).
fn percentile_f64(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * p;
    let lower = pos.floor() as usize;
    let upper = (pos.ceil() as usize).min(sorted.len() - 1);
    let frac = pos - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * frac
}

fn percentile_i64(sorted: &[i64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let as_f64: Vec<f64> = sorted.iter().map(|v| *v as f64).collect();
    percentile_f64(&as_f64, p)
}

/// Bounded slowdown: (wait + runtime) / max(runtime, 60), matching the Python
/// reference engine (python/hpcopt/simulate/metrics.py).
fn bsld(wait_sec: i64, runtime_sec: i64) -> f64 {
    let thresh = runtime_sec.max(60);
    (wait_sec as f64 + runtime_sec as f64) / thresh as f64
}

fn check_invariants(
    clock_ts: i64,
    capacity: u32,
    free_cpus: u32,
    running: &[RunningJob],
    queued: &VecDeque<Job>,
) -> Vec<String> {
    let mut violations = Vec::new();
    if free_cpus > capacity {
        violations.push("free_cpus_exceeds_capacity".to_string());
    }
    let running_cpus: u32 = running.iter().map(|j| j.requested_cpus).sum();
    if running_cpus.saturating_add(free_cpus) != capacity {
        violations.push("cpu_conservation_broken".to_string());
    }
    for job in running {
        if job.start_ts < job.submit_ts {
            violations.push(format!("job_start_before_submit:{}", job.job_id));
        }
    }
    for job in queued {
        if job.submit_ts > clock_ts {
            violations.push(format!("queued_job_submit_in_future:{}", job.job_id));
        }
    }
    violations
}

// ── FIFO dispatch ──────────────────────────────────────────────

fn dispatch_fifo(queued: &mut VecDeque<Job>, free_cpus: &mut u32, clock_ts: i64) -> Vec<RunningJob> {
    let mut started = Vec::new();
    loop {
        let can_dispatch = queued.front().is_some_and(|h| h.requested_cpus > 0 && h.requested_cpus <= *free_cpus);
        if !can_dispatch {
            break;
        }
        let job = queued.pop_front().unwrap();
        let runtime = job.runtime_actual_sec.max(0);
        let end_ts = clock_ts.saturating_add(runtime);
        *free_cpus -= job.requested_cpus;
        started.push(RunningJob {
            job_id: job.job_id,
            submit_ts: job.submit_ts,
            start_ts: clock_ts,
            end_ts,
            requested_cpus: job.requested_cpus,
        });
    }
    started
}

// ── EASY BACKFILL dispatch ─────────────────────────────────────
//
// 1. Try to dispatch the head-of-queue (same as FIFO).
// 2. If head is blocked, compute its "shadow time" = earliest time
//    enough CPUs will free up to run it.
// 3. Backfill: scan remaining queued jobs; start any that fit in
//    the current free CPUs AND finish before the shadow time.

fn dispatch_easy_backfill(
    queued: &mut VecDeque<Job>,
    running: &[RunningJob],
    free_cpus: &mut u32,
    clock_ts: i64,
) -> Vec<RunningJob> {
    let mut started = Vec::new();

    // Step 1: Try head-of-queue dispatch (greedy, like FIFO)
    loop {
        let can = queued.front().is_some_and(|h| h.requested_cpus > 0 && h.requested_cpus <= *free_cpus);
        if !can {
            break;
        }
        let job = queued.pop_front().unwrap();
        let runtime = job.runtime_actual_sec.max(0);
        let end_ts = clock_ts.saturating_add(runtime);
        *free_cpus -= job.requested_cpus;
        started.push(RunningJob {
            job_id: job.job_id,
            submit_ts: job.submit_ts,
            start_ts: clock_ts,
            end_ts,
            requested_cpus: job.requested_cpus,
        });
    }

    // If queue is empty or head can already run, no backfill needed.
    let head = match queued.front() {
        Some(h) if h.requested_cpus > *free_cpus => h,
        _ => return started,
    };

    // Step 2: Compute shadow time for the head-of-queue job.
    // Sort running jobs by end_ts ascending; accumulate freed CPUs until head fits.
    let head_cpus = head.requested_cpus;
    let mut ends: Vec<(i64, u32)> = running.iter().map(|j| (j.end_ts, j.requested_cpus)).collect();
    // Also include jobs we just started
    for s in &started {
        ends.push((s.end_ts, s.requested_cpus));
    }
    ends.sort_by_key(|&(t, _)| t);

    let mut cumulative_free = *free_cpus;
    let mut shadow_time = i64::MAX;
    for (end_ts, cpus) in &ends {
        cumulative_free += cpus;
        if cumulative_free >= head_cpus {
            shadow_time = *end_ts;
            break;
        }
    }

    // Step 3: Backfill — scan queue (skip head) for jobs that fit and complete before shadow time.
    let mut backfill_indices = Vec::new();
    for (i, job) in queued.iter().enumerate().skip(1) {
        if job.requested_cpus > 0
            && job.requested_cpus <= *free_cpus
            && clock_ts.saturating_add(job.runtime_actual_sec.max(0)) <= shadow_time
        {
            backfill_indices.push(i);
            *free_cpus -= job.requested_cpus;
        }
        if *free_cpus == 0 {
            break;
        }
    }

    // Remove backfilled jobs from queue (reverse order to preserve indices).
    for &idx in backfill_indices.iter().rev() {
        let job = queued.remove(idx).unwrap();
        let runtime = job.runtime_actual_sec.max(0);
        let end_ts = clock_ts.saturating_add(runtime);
        started.push(RunningJob {
            job_id: job.job_id,
            submit_ts: job.submit_ts,
            start_ts: clock_ts,
            end_ts,
            requested_cpus: job.requested_cpus,
        });
    }

    started
}

// ── Main simulation loop ───────────────────────────────────────

fn simulate(
    mut jobs: Vec<Job>,
    policy: &str,
    capacity_cpus: u32,
    strict_invariants: bool,
) -> Result<SimulationReport> {
    jobs.sort_by(|a, b| a.submit_ts.cmp(&b.submit_ts).then_with(|| a.job_id.cmp(&b.job_id)));

    let total_jobs = jobs.len();
    let mut submit_idx: usize = 0;
    let mut queued: VecDeque<Job> = VecDeque::new();
    let mut running: Vec<RunningJob> = Vec::new();
    let mut free_cpus = capacity_cpus;
    let mut clock_ts = jobs.first().map(|j| j.submit_ts).unwrap_or(0);
    let min_submit_ts = clock_ts;
    let mut max_end_ts = min_submit_ts;
    let mut total_cpu_seconds: i128 = 0;
    let mut completed: Vec<CompletedJob> = Vec::with_capacity(total_jobs);
    let mut all_violations: Vec<String> = Vec::new();

    while completed.len() < total_jobs {
        let next_submit = jobs.get(submit_idx).map(|j| j.submit_ts).unwrap_or(i64::MAX);
        let next_complete = running.iter().map(|j| j.end_ts).min().unwrap_or(i64::MAX);

        if next_submit == i64::MAX && next_complete == i64::MAX {
            break;
        }
        clock_ts = next_submit.min(next_complete);

        // Complete finished jobs (deterministic: complete before submit at same timestamp).
        let mut i = 0;
        while i < running.len() {
            if running[i].end_ts == clock_ts {
                let rj = running.swap_remove(i);
                free_cpus += rj.requested_cpus;
                let wait = rj.start_ts.saturating_sub(rj.submit_ts).max(0);
                let runtime = rj.end_ts.saturating_sub(rj.start_ts).max(0);
                total_cpu_seconds += i128::from(rj.requested_cpus) * i128::from(runtime);
                max_end_ts = max_end_ts.max(rj.end_ts);
                completed.push(CompletedJob {
                    job_id: rj.job_id,
                    submit_ts: rj.submit_ts,
                    start_ts: rj.start_ts,
                    end_ts: rj.end_ts,
                    requested_cpus: rj.requested_cpus,
                    wait_sec: wait,
                });
            } else {
                i += 1;
            }
        }

        // Admit newly submitted jobs to queue.
        while let Some(job) = jobs.get(submit_idx) {
            if job.submit_ts != clock_ts {
                break;
            }
            queued.push_back(job.clone());
            submit_idx += 1;
        }

        // Policy dispatch.
        let newly_started = match policy {
            "EASY_BACKFILL_BASELINE" => dispatch_easy_backfill(&mut queued, &running, &mut free_cpus, clock_ts),
            _ => dispatch_fifo(&mut queued, &mut free_cpus, clock_ts),
        };
        running.extend(newly_started);

        // Invariant check.
        let violations = check_invariants(clock_ts, capacity_cpus, free_cpus, &running, &queued);
        if !violations.is_empty() {
            all_violations.extend(violations.iter().cloned());
            if strict_invariants {
                anyhow::bail!("strict invariants failed at ts={clock_ts}: {}", violations.join(","));
            }
        }
    }

    // Compute metrics.
    let mut waits: Vec<i64> = completed.iter().map(|j| j.wait_sec).collect();
    waits.sort_unstable();
    let mean_wait = if waits.is_empty() {
        0.0
    } else {
        waits.iter().sum::<i64>() as f64 / waits.len() as f64
    };
    let p95_wait = percentile_i64(&waits, 0.95);

    let mut bslds: Vec<f64> = completed
        .iter()
        .map(|j| bsld(j.wait_sec, j.end_ts.saturating_sub(j.start_ts).max(0)))
        .collect();
    bslds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_bsld = percentile_f64(&bslds, 0.95);

    let makespan = max_end_ts.saturating_sub(min_submit_ts).max(0);
    let utilization_mean = if makespan > 0 {
        let denom = capacity_cpus as f64 * makespan as f64;
        (total_cpu_seconds as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let violation_examples: Vec<String> = all_violations.iter().take(20).cloned().collect();

    Ok(SimulationReport {
        policy_id: policy.to_string(),
        run_id: format!("rust_sim_{}", policy.to_ascii_lowercase()),
        metrics: Metrics {
            policy_id: policy.to_string(),
            capacity_cpus,
            jobs_total: total_jobs,
            jobs_completed: completed.len(),
            mean_wait_sec: mean_wait,
            p95_wait_sec: p95_wait,
            p95_bsld,
            utilization_mean,
            makespan_sec: makespan,
            invariant_violations: all_violations.len(),
        },
        invariant_report: InvariantReport {
            strict: strict_invariants,
            total_violations: all_violations.len(),
            examples: violation_examples,
        },
    })
}

// ── Entry point ────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    if args.capacity_cpus == 0 {
        anyhow::bail!("capacity_cpus must be > 0");
    }

    let valid_policies = ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"];
    if !valid_policies.contains(&args.policy.as_str()) {
        anyhow::bail!(
            "unknown policy '{}'. Valid: {}",
            args.policy,
            valid_policies.join(", ")
        );
    }

    let input_raw = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let jobs: Vec<Job> = serde_json::from_str(&input_raw)
        .with_context(|| "failed to parse input JSON array of jobs")?;

    eprintln!(
        "sim-runner: {} jobs, policy={}, capacity={}",
        jobs.len(),
        args.policy,
        args.capacity_cpus
    );

    let report = simulate(jobs, &args.policy, args.capacity_cpus, args.strict_invariants)?;

    eprintln!(
        "sim-runner: done. p95_bsld={:.3}, util={:.3}, makespan={}s",
        report.metrics.p95_bsld,
        report.metrics.utilization_mean,
        report.metrics.makespan_sec
    );

    let output = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        fs::write(&path, &output)
            .with_context(|| format!("failed to write report {}", path.display()))?;
    } else {
        println!("{output}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Metric parity contract with python/hpcopt/simulate/metrics.py — these
    // formulas MUST stay identical across engines (see
    // docs/validation/batsim-agreement.md for the defect this guards against).

    #[test]
    fn bsld_matches_python_metric_contract() {
        // (wait + runtime) / max(runtime, 60)
        assert_eq!(bsld(0, 120), 1.0);
        assert_eq!(bsld(120, 120), 2.0);
        // Short jobs are floored at 60s in the denominator only.
        assert_eq!(bsld(570, 30), 10.0);
        // Zero-wait short job can be below 1.0 (no clamping, matching Python).
        assert_eq!(bsld(0, 30), 0.5);
    }

    #[test]
    fn percentile_uses_linear_interpolation_like_numpy() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        // numpy.quantile([1,2,3,4], 0.95) == 3.85
        assert!((percentile_f64(&values, 0.95) - 3.85).abs() < 1e-9);
        assert_eq!(percentile_f64(&values, 0.0), 1.0);
        assert_eq!(percentile_f64(&values, 1.0), 4.0);
        assert_eq!(percentile_f64(&[], 0.95), 0.0);
        assert!((percentile_i64(&[10, 20], 0.5) - 15.0).abs() < 1e-9);
    }
}
