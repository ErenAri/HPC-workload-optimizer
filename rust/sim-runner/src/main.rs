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

    /// Total GPU capacity. 0 (default) disables the GPU dimension:
    /// job GPU requests are ignored and scheduling is identical to the
    /// CPU-only engine.
    #[arg(long, default_value_t = 0)]
    capacity_gpus: u32,

    /// Total memory capacity (same unit as the trace's requested_mem,
    /// canonically KB from SWF). 0 (default) disables the dimension.
    #[arg(long, default_value_t = 0)]
    capacity_mem: u64,

    /// Facility power cap in watts. When set (and the trace carries
    /// power_mean_watts), the report includes time and excess energy above
    /// the cap. Measurement only unless --enforce-power-cap is also given.
    #[arg(long)]
    power_cap_watts: Option<f64>,

    /// Treat --power-cap-watts as a hard dispatch constraint: a job starts
    /// only if current draw + its mean watts stays at or under the cap.
    /// Jobs whose own draw exceeds the cap can never start (reported via
    /// jobs_completed < jobs_total).
    #[arg(long, default_value_t = false)]
    enforce_power_cap: bool,

    /// Fail immediately on invariant violations.
    #[arg(long, default_value_t = false)]
    strict_invariants: bool,

    /// Output report path. Omit to print JSON to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Resource vector ────────────────────────────────────────────
//
// One allocatable resource bundle. A dimension with capacity 0 is "not
// modeled": job requests in that dimension are normalized to 0 at load
// time, so every fit/conservation check degenerates to the historical
// CPU-only behavior (bit-for-bit — the published benchmarks depend on it).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Resources {
    cpus: u32,
    gpus: u32,
    mem: u64,
}

impl Resources {
    const ZERO: Resources = Resources { cpus: 0, gpus: 0, mem: 0 };

    fn fits_in(&self, free: &Resources) -> bool {
        self.cpus <= free.cpus && self.gpus <= free.gpus && self.mem <= free.mem
    }

    fn add(&mut self, other: &Resources) {
        self.cpus = self.cpus.saturating_add(other.cpus);
        self.gpus = self.gpus.saturating_add(other.gpus);
        self.mem = self.mem.saturating_add(other.mem);
    }

    fn sub(&mut self, other: &Resources) {
        self.cpus -= other.cpus;
        self.gpus -= other.gpus;
        self.mem -= other.mem;
    }

    fn exceeds(&self, capacity: &Resources) -> bool {
        self.cpus > capacity.cpus || self.gpus > capacity.gpus || self.mem > capacity.mem
    }
}

// ── Data structures ────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct JobRecord {
    job_id: u64,
    submit_ts: i64,
    runtime_actual_sec: i64,
    requested_cpus: u32,
    #[serde(default)]
    requested_gpus: u32,
    #[serde(default)]
    requested_mem: u64,
    /// Mean power draw while running (watts, summed over the job's nodes).
    /// 0 = no power data; power metrics are omitted from the report.
    #[serde(default)]
    power_mean_watts: f64,
}

#[derive(Debug, Clone)]
struct Job {
    job_id: u64,
    submit_ts: i64,
    runtime_actual_sec: i64,
    requested: Resources,
    power_mean_watts: f64,
}

#[derive(Debug, Clone)]
struct RunningJob {
    job_id: u64,
    submit_ts: i64,
    start_ts: i64,
    end_ts: i64,
    requested: Resources,
    power_mean_watts: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CompletedJob {
    job_id: u64,
    submit_ts: i64,
    start_ts: i64,
    end_ts: i64,
    requested: Resources,
    wait_sec: i64,
}

// ── Report structures ──────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Metrics {
    policy_id: String,
    capacity_cpus: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity_gpus: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity_mem: Option<u64>,
    jobs_total: usize,
    jobs_completed: usize,
    mean_wait_sec: f64,
    p95_wait_sec: f64,
    p95_bsld: f64,
    utilization_mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    utilization_gpu_mean: Option<f64>,
    makespan_sec: i64,
    invariant_violations: usize,
    /// Power metrics: present only when the trace carries power_mean_watts.
    /// Energy is schedule-invariant (a job draws the same joules whenever it
    /// runs); peak power and above-cap exposure are schedule-DEPENDENT and
    /// are what energy-aware policies actually trade against BSLD.
    #[serde(skip_serializing_if = "Option::is_none")]
    energy_joules_total: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_peak_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_mean_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_cap_watts: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_cap_enforced: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds_above_power_cap: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    joules_above_power_cap: Option<f64>,
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
    capacity: &Resources,
    free: &Resources,
    running: &[RunningJob],
    queued: &VecDeque<Job>,
) -> Vec<String> {
    let mut violations = Vec::new();
    if free.exceeds(capacity) {
        violations.push("free_resources_exceed_capacity".to_string());
    }
    let mut allocated = Resources::ZERO;
    for j in running {
        allocated.add(&j.requested);
    }
    let mut total = allocated;
    total.add(free);
    if total != *capacity {
        violations.push("resource_conservation_broken".to_string());
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

fn start_job(job: Job, clock_ts: i64, free: &mut Resources, headroom_watts: &mut f64) -> RunningJob {
    let runtime = job.runtime_actual_sec.max(0);
    free.sub(&job.requested);
    *headroom_watts -= job.power_mean_watts;
    RunningJob {
        job_id: job.job_id,
        submit_ts: job.submit_ts,
        start_ts: clock_ts,
        end_ts: clock_ts.saturating_add(runtime),
        requested: job.requested,
        power_mean_watts: job.power_mean_watts,
    }
}

/// Time-integrated cluster power profile. Draw changes only at job
/// start/complete events, so integrating piecewise-constant segments between
/// event timestamps is exact (given per-job mean power).
struct PowerProfile {
    current_watts: f64,
    peak_watts: f64,
    energy_joules: f64,
    cap_watts: Option<f64>,
    seconds_above_cap: i64,
    joules_above_cap: f64,
    last_ts: i64,
}

impl PowerProfile {
    fn new(start_ts: i64, cap_watts: Option<f64>) -> Self {
        PowerProfile {
            current_watts: 0.0,
            peak_watts: 0.0,
            energy_joules: 0.0,
            cap_watts,
            seconds_above_cap: 0,
            joules_above_cap: 0.0,
            last_ts: start_ts,
        }
    }

    /// Account the constant-draw segment [last_ts, now_ts) — call BEFORE
    /// applying this instant's start/complete events.
    fn advance_to(&mut self, now_ts: i64) {
        let dt = now_ts.saturating_sub(self.last_ts).max(0);
        if dt > 0 {
            self.energy_joules += self.current_watts * dt as f64;
            if let Some(cap) = self.cap_watts {
                if self.current_watts > cap {
                    self.seconds_above_cap += dt;
                    self.joules_above_cap += (self.current_watts - cap) * dt as f64;
                }
            }
        }
        self.last_ts = now_ts;
    }

    fn job_started(&mut self, watts: f64) {
        self.current_watts += watts;
        self.peak_watts = self.peak_watts.max(self.current_watts);
    }

    fn job_completed(&mut self, watts: f64) {
        self.current_watts = (self.current_watts - watts).max(0.0);
    }

    /// Zero out float drift when nothing is running, so an enforced cap
    /// never phantom-blocks a job that needs the full budget.
    fn machine_idle(&mut self) {
        self.current_watts = 0.0;
    }
}

// ── FIFO dispatch ──────────────────────────────────────────────

fn dispatch_fifo(
    queued: &mut VecDeque<Job>,
    free: &mut Resources,
    headroom_watts: &mut f64,
    clock_ts: i64,
) -> Vec<RunningJob> {
    let mut started = Vec::new();
    loop {
        let can_dispatch = queued.front().is_some_and(|h| {
            h.requested.cpus > 0 && h.requested.fits_in(free) && h.power_mean_watts <= *headroom_watts
        });
        if !can_dispatch {
            break;
        }
        let job = queued.pop_front().unwrap();
        started.push(start_job(job, clock_ts, free, headroom_watts));
    }
    started
}

// ── EASY BACKFILL dispatch ─────────────────────────────────────
//
// 1. Try to dispatch the head-of-queue (same as FIFO).
// 2. If head is blocked, compute its "shadow time" = earliest time
//    enough resources will free up to run it.
// 3. Backfill: scan remaining queued jobs; start any that fit in
//    the current free resources AND finish before the shadow time.

fn dispatch_easy_backfill(
    queued: &mut VecDeque<Job>,
    running: &[RunningJob],
    free: &mut Resources,
    headroom_watts: &mut f64,
    clock_ts: i64,
) -> Vec<RunningJob> {
    let mut started = Vec::new();

    // Step 1: Try head-of-queue dispatch (greedy, like FIFO)
    loop {
        let can = queued.front().is_some_and(|h| {
            h.requested.cpus > 0 && h.requested.fits_in(free) && h.power_mean_watts <= *headroom_watts
        });
        if !can {
            break;
        }
        let job = queued.pop_front().unwrap();
        started.push(start_job(job, clock_ts, free, headroom_watts));
    }

    // If queue is empty or head can already run, no backfill needed.
    let head = match queued.front() {
        Some(h) if !h.requested.fits_in(free) || h.power_mean_watts > *headroom_watts => h,
        _ => return started,
    };

    // Step 2: Compute shadow time for the head-of-queue job.
    // Sort running jobs by end_ts ascending; accumulate freed resources
    // (every dimension, and power headroom under an enforced cap) until
    // head fits.
    let head_req = head.requested;
    let head_watts = head.power_mean_watts;
    let mut ends: Vec<(i64, Resources, f64)> = running
        .iter()
        .map(|j| (j.end_ts, j.requested, j.power_mean_watts))
        .collect();
    // Also include jobs we just started
    for s in &started {
        ends.push((s.end_ts, s.requested, s.power_mean_watts));
    }
    ends.sort_by(|a, b| a.0.cmp(&b.0));

    let mut cumulative_free = *free;
    let mut cumulative_headroom = *headroom_watts;
    let mut shadow_time = i64::MAX;
    for (end_ts, res, watts) in &ends {
        cumulative_free.add(res);
        cumulative_headroom += watts;
        if head_req.fits_in(&cumulative_free) && head_watts <= cumulative_headroom {
            shadow_time = *end_ts;
            break;
        }
    }

    // Step 3: Backfill — scan queue (skip head) for jobs that fit and complete before shadow time.
    let mut backfill_indices = Vec::new();
    for (i, job) in queued.iter().enumerate().skip(1) {
        if job.requested.cpus > 0
            && job.requested.fits_in(free)
            && job.power_mean_watts <= *headroom_watts
            && clock_ts.saturating_add(job.runtime_actual_sec.max(0)) <= shadow_time
        {
            backfill_indices.push(i);
            free.sub(&job.requested);
            *headroom_watts -= job.power_mean_watts;
        }
        if free.cpus == 0 {
            break;
        }
    }

    // Restore reserved resources; start_job re-subtracts them below.
    for &idx in &backfill_indices {
        free.add(&queued[idx].requested);
        *headroom_watts += queued[idx].power_mean_watts;
    }

    // Remove backfilled jobs from queue (reverse order to preserve indices).
    for &idx in backfill_indices.iter().rev() {
        let job = queued.remove(idx).unwrap();
        started.push(start_job(job, clock_ts, free, headroom_watts));
    }

    started
}

// ── Main simulation loop ───────────────────────────────────────

fn simulate(
    records: Vec<JobRecord>,
    policy: &str,
    capacity: Resources,
    power_cap_watts: Option<f64>,
    enforce_power_cap: bool,
    strict_invariants: bool,
) -> Result<SimulationReport> {
    if enforce_power_cap && power_cap_watts.is_none() {
        anyhow::bail!("--enforce-power-cap requires --power-cap-watts");
    }
    // Normalize: a dimension with capacity 0 is not modeled — zero the
    // request so all fit/conservation checks degrade to CPU-only behavior.
    let mut jobs: Vec<Job> = records
        .into_iter()
        .map(|r| Job {
            job_id: r.job_id,
            submit_ts: r.submit_ts,
            runtime_actual_sec: r.runtime_actual_sec,
            requested: Resources {
                cpus: r.requested_cpus,
                gpus: if capacity.gpus == 0 { 0 } else { r.requested_gpus },
                mem: if capacity.mem == 0 { 0 } else { r.requested_mem },
            },
            power_mean_watts: r.power_mean_watts.max(0.0),
        })
        .collect();
    let has_power_data = jobs.iter().any(|j| j.power_mean_watts > 0.0);
    jobs.sort_by(|a, b| a.submit_ts.cmp(&b.submit_ts).then_with(|| a.job_id.cmp(&b.job_id)));

    let total_jobs = jobs.len();
    let mut submit_idx: usize = 0;
    let mut queued: VecDeque<Job> = VecDeque::new();
    let mut running: Vec<RunningJob> = Vec::new();
    let mut free = capacity;
    let mut clock_ts = jobs.first().map(|j| j.submit_ts).unwrap_or(0);
    let min_submit_ts = clock_ts;
    let mut max_end_ts = min_submit_ts;
    let mut total_cpu_seconds: i128 = 0;
    let mut total_gpu_seconds: i128 = 0;
    let mut completed: Vec<CompletedJob> = Vec::with_capacity(total_jobs);
    let mut all_violations: Vec<String> = Vec::new();
    let mut power = PowerProfile::new(clock_ts, power_cap_watts);

    while completed.len() < total_jobs {
        let next_submit = jobs.get(submit_idx).map(|j| j.submit_ts).unwrap_or(i64::MAX);
        let next_complete = running.iter().map(|j| j.end_ts).min().unwrap_or(i64::MAX);

        if next_submit == i64::MAX && next_complete == i64::MAX {
            break;
        }
        clock_ts = next_submit.min(next_complete);
        power.advance_to(clock_ts);

        // Complete finished jobs (deterministic: complete before submit at same timestamp).
        let mut i = 0;
        while i < running.len() {
            if running[i].end_ts == clock_ts {
                let rj = running.swap_remove(i);
                free.add(&rj.requested);
                power.job_completed(rj.power_mean_watts);
                let wait = rj.start_ts.saturating_sub(rj.submit_ts).max(0);
                let runtime = rj.end_ts.saturating_sub(rj.start_ts).max(0);
                total_cpu_seconds += i128::from(rj.requested.cpus) * i128::from(runtime);
                total_gpu_seconds += i128::from(rj.requested.gpus) * i128::from(runtime);
                max_end_ts = max_end_ts.max(rj.end_ts);
                completed.push(CompletedJob {
                    job_id: rj.job_id,
                    submit_ts: rj.submit_ts,
                    start_ts: rj.start_ts,
                    end_ts: rj.end_ts,
                    requested: rj.requested,
                    wait_sec: wait,
                });
            } else {
                i += 1;
            }
        }
        if running.is_empty() {
            power.machine_idle();
        }

        // Admit newly submitted jobs to queue.
        while let Some(job) = jobs.get(submit_idx) {
            if job.submit_ts != clock_ts {
                break;
            }
            queued.push_back(job.clone());
            submit_idx += 1;
        }

        // Policy dispatch. Headroom is recomputed from the live draw each
        // event, so float drift cannot accumulate across the run.
        let mut headroom_watts = match power_cap_watts {
            Some(cap) if enforce_power_cap => (cap - power.current_watts).max(0.0),
            _ => f64::INFINITY,
        };
        let newly_started = match policy {
            "EASY_BACKFILL_BASELINE" => {
                dispatch_easy_backfill(&mut queued, &running, &mut free, &mut headroom_watts, clock_ts)
            }
            _ => dispatch_fifo(&mut queued, &mut free, &mut headroom_watts, clock_ts),
        };
        for job in &newly_started {
            power.job_started(job.power_mean_watts);
        }
        running.extend(newly_started);

        // Invariant check.
        let violations = check_invariants(clock_ts, &capacity, &free, &running, &queued);
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
        let denom = capacity.cpus as f64 * makespan as f64;
        (total_cpu_seconds as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let utilization_gpu_mean = if capacity.gpus > 0 && makespan > 0 {
        let denom = capacity.gpus as f64 * makespan as f64;
        Some((total_gpu_seconds as f64 / denom).clamp(0.0, 1.0))
    } else {
        None
    };

    let violation_examples: Vec<String> = all_violations.iter().take(20).cloned().collect();

    let (energy_joules_total, power_peak_watts, power_mean_watts) = if has_power_data {
        let mean = if makespan > 0 {
            power.energy_joules / makespan as f64
        } else {
            0.0
        };
        (Some(power.energy_joules), Some(power.peak_watts), Some(mean))
    } else {
        (None, None, None)
    };
    let cap_metrics = power_cap_watts.filter(|_| has_power_data);

    Ok(SimulationReport {
        policy_id: policy.to_string(),
        run_id: format!("rust_sim_{}", policy.to_ascii_lowercase()),
        metrics: Metrics {
            policy_id: policy.to_string(),
            capacity_cpus: capacity.cpus,
            capacity_gpus: (capacity.gpus > 0).then_some(capacity.gpus),
            capacity_mem: (capacity.mem > 0).then_some(capacity.mem),
            jobs_total: total_jobs,
            jobs_completed: completed.len(),
            mean_wait_sec: mean_wait,
            p95_wait_sec: p95_wait,
            p95_bsld,
            utilization_mean,
            utilization_gpu_mean,
            makespan_sec: makespan,
            invariant_violations: all_violations.len(),
            energy_joules_total,
            power_peak_watts,
            power_mean_watts,
            power_cap_watts: cap_metrics,
            power_cap_enforced: cap_metrics.map(|_| enforce_power_cap),
            seconds_above_power_cap: cap_metrics.map(|_| power.seconds_above_cap),
            joules_above_power_cap: cap_metrics.map(|_| power.joules_above_cap),
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
    let jobs: Vec<JobRecord> = serde_json::from_str(&input_raw)
        .with_context(|| "failed to parse input JSON array of jobs")?;

    let capacity = Resources {
        cpus: args.capacity_cpus,
        gpus: args.capacity_gpus,
        mem: args.capacity_mem,
    };

    eprintln!(
        "sim-runner: {} jobs, policy={}, capacity=cpus:{} gpus:{} mem:{}",
        jobs.len(),
        args.policy,
        capacity.cpus,
        capacity.gpus,
        capacity.mem
    );

    let report = simulate(
        jobs,
        &args.policy,
        capacity,
        args.power_cap_watts,
        args.enforce_power_cap,
        args.strict_invariants,
    )?;

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

    // ── Multi-resource tests ───────────────────────────────────

    fn rec(job_id: u64, submit_ts: i64, runtime: i64, cpus: u32, gpus: u32, mem: u64) -> JobRecord {
        JobRecord {
            job_id,
            submit_ts,
            runtime_actual_sec: runtime,
            requested_cpus: cpus,
            requested_gpus: gpus,
            requested_mem: mem,
            power_mean_watts: 0.0,
        }
    }

    fn rec_power(job_id: u64, submit_ts: i64, runtime: i64, cpus: u32, watts: f64) -> JobRecord {
        JobRecord {
            power_mean_watts: watts,
            ..rec(job_id, submit_ts, runtime, cpus, 0, 0)
        }
    }

    fn cap(cpus: u32, gpus: u32, mem: u64) -> Resources {
        Resources { cpus, gpus, mem }
    }

    #[test]
    fn resources_fit_requires_every_dimension() {
        let free = cap(8, 2, 1000);
        assert!(cap(8, 2, 1000).fits_in(&free));
        assert!(!cap(9, 0, 0).fits_in(&free));
        assert!(!cap(1, 3, 0).fits_in(&free));
        assert!(!cap(1, 0, 1001).fits_in(&free));
        assert!(Resources::ZERO.fits_in(&free));
    }

    #[test]
    fn scalar_mode_ignores_gpu_and_mem_requests() {
        // capacity_gpus == 0 → GPU requests must not affect scheduling.
        let jobs_with_gpus = vec![rec(1, 0, 100, 4, 999, 0), rec(2, 0, 100, 4, 999, 0)];
        let jobs_plain = vec![rec(1, 0, 100, 4, 0, 0), rec(2, 0, 100, 4, 0, 0)];
        let a = simulate(jobs_with_gpus, "FIFO_STRICT", cap(8, 0, 0), None, false, true).unwrap();
        let b = simulate(jobs_plain, "FIFO_STRICT", cap(8, 0, 0), None, false, true).unwrap();
        assert_eq!(a.metrics.mean_wait_sec, b.metrics.mean_wait_sec);
        assert_eq!(a.metrics.makespan_sec, b.metrics.makespan_sec);
        assert_eq!(a.metrics.p95_bsld, b.metrics.p95_bsld);
        // Both jobs run concurrently: zero wait.
        assert_eq!(a.metrics.mean_wait_sec, 0.0);
        assert!(a.metrics.capacity_gpus.is_none());
        assert!(a.metrics.utilization_gpu_mean.is_none());
    }

    #[test]
    fn gpu_scarcity_serializes_jobs_with_free_cpus() {
        // Plenty of CPUs, one GPU: two 1-GPU jobs must run back to back.
        let jobs = vec![rec(1, 0, 100, 1, 1, 0), rec(2, 0, 100, 1, 1, 0)];
        let report = simulate(jobs, "FIFO_STRICT", cap(64, 1, 0), None, false, true).unwrap();
        assert_eq!(report.metrics.jobs_completed, 2);
        // Second job waits the full 100s runtime of the first.
        assert_eq!(report.metrics.mean_wait_sec, 50.0);
        assert_eq!(report.metrics.makespan_sec, 200);
        assert_eq!(report.metrics.utilization_gpu_mean, Some(1.0));
        assert_eq!(report.invariant_report.total_violations, 0);
    }

    #[test]
    fn mem_scarcity_serializes_jobs_with_free_cpus() {
        let jobs = vec![rec(1, 0, 100, 1, 0, 800), rec(2, 0, 100, 1, 0, 800)];
        let report = simulate(jobs, "FIFO_STRICT", cap(64, 0, 1000), None, false, true).unwrap();
        assert_eq!(report.metrics.mean_wait_sec, 50.0);
        assert_eq!(report.metrics.makespan_sec, 200);
        assert_eq!(report.metrics.capacity_mem, Some(1000));
        assert_eq!(report.invariant_report.total_violations, 0);
    }

    #[test]
    fn easy_backfill_respects_gpu_shadow_time() {
        // t=0: job1 takes both GPUs for 100s.
        // job2 (head, blocked) needs 2 GPUs → shadow time = 100.
        // job3 needs 1 CPU, 0 GPUs, runs 50s ≤ shadow → backfills at t=0.
        // job4 needs 1 GPU and would finish at 150 > shadow → must NOT backfill
        //   (GPUs are free? no — job1 holds both, so it can't start anyway).
        // job5 needs 1 CPU, 0 GPUs, runs 200s > shadow → must NOT backfill.
        let jobs = vec![
            rec(1, 0, 100, 2, 2, 0),
            rec(2, 1, 100, 2, 2, 0),
            rec(3, 1, 50, 1, 0, 0),
            rec(5, 1, 200, 1, 0, 0),
        ];
        let report = simulate(jobs, "EASY_BACKFILL_BASELINE", cap(8, 2, 0), None, false, true).unwrap();
        assert_eq!(report.metrics.jobs_completed, 4);
        assert_eq!(report.invariant_report.total_violations, 0);
        // job3 backfilled at t=1 (0 wait); job2 starts at t=100 (99 wait);
        // job5 starts when job2 finishes? No — job5 only needs CPUs, it can
        // start at t=100 alongside job2 (8 cpus ≥ 2+1). Waits: j1=0, j2=99,
        // j3=0, j5=99 → mean = 49.5.
        assert_eq!(report.metrics.mean_wait_sec, 49.5);
    }

    #[test]
    fn easy_backfill_gpu_job_cannot_backfill_past_gpu_reservation() {
        // GPUs: 2 total. job1 holds 1 GPU until t=100. Head job2 needs 2 GPUs
        // → shadow = 100 with exactly the freed GPU + the 1 free GPU reserved.
        // job3 (1 GPU, 30s) fits in current free (1 GPU) and ends ≤ shadow →
        // legitimately backfills (EASY only protects the head's start time).
        let jobs = vec![
            rec(1, 0, 100, 1, 1, 0),
            rec(2, 1, 100, 1, 2, 0),
            rec(3, 1, 30, 1, 1, 0),
        ];
        let report = simulate(jobs, "EASY_BACKFILL_BASELINE", cap(8, 2, 0), None, false, true).unwrap();
        assert_eq!(report.invariant_report.total_violations, 0);
        // j3 backfills at t=1, ends t=31 ≤ 100; head starts at t=100.
        // Waits: j1=0, j2=99, j3=0 → mean = 33.
        assert_eq!(report.metrics.mean_wait_sec, 33.0);
    }

    // ── Power / energy tests ───────────────────────────────────

    #[test]
    fn power_metrics_absent_without_power_data() {
        let jobs = vec![rec(1, 0, 100, 4, 0, 0)];
        let report = simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), Some(500.0), false, true).unwrap();
        assert!(report.metrics.energy_joules_total.is_none());
        assert!(report.metrics.power_peak_watts.is_none());
        assert!(report.metrics.power_cap_watts.is_none());
        assert!(report.metrics.seconds_above_power_cap.is_none());
    }

    #[test]
    fn energy_is_schedule_invariant_but_peak_power_is_not() {
        // Two 400 W jobs, 100 s each. Concurrent (capacity 8): peak 800 W.
        // Serialized (capacity 4): peak 400 W. Energy identical: 80 kJ.
        let jobs = || vec![rec_power(1, 0, 100, 4, 400.0), rec_power(2, 0, 100, 4, 400.0)];
        let wide = simulate(jobs(), "FIFO_STRICT", cap(8, 0, 0), None, false, true).unwrap();
        let narrow = simulate(jobs(), "FIFO_STRICT", cap(4, 0, 0), None, false, true).unwrap();

        assert_eq!(wide.metrics.energy_joules_total, Some(80_000.0));
        assert_eq!(narrow.metrics.energy_joules_total, Some(80_000.0));
        assert_eq!(wide.metrics.power_peak_watts, Some(800.0));
        assert_eq!(narrow.metrics.power_peak_watts, Some(400.0));
        // Time-weighted mean power = energy / makespan.
        assert_eq!(wide.metrics.power_mean_watts, Some(800.0)); // 80 kJ / 100 s
        assert_eq!(narrow.metrics.power_mean_watts, Some(400.0)); // 80 kJ / 200 s
    }

    #[test]
    fn power_cap_exposure_accounts_time_and_excess_joules() {
        // Concurrent draw 800 W for 100 s against a 600 W cap:
        // 100 s above cap, 200 W excess * 100 s = 20 kJ above cap.
        let jobs = vec![rec_power(1, 0, 100, 4, 400.0), rec_power(2, 0, 100, 4, 400.0)];
        let report = simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), Some(600.0), false, true).unwrap();
        assert_eq!(report.metrics.power_cap_watts, Some(600.0));
        assert_eq!(report.metrics.seconds_above_power_cap, Some(100));
        assert_eq!(report.metrics.joules_above_power_cap, Some(20_000.0));

        // Serialized under the same cap: never above it.
        let jobs = vec![rec_power(1, 0, 100, 4, 400.0), rec_power(2, 0, 100, 4, 400.0)];
        let report = simulate(jobs, "FIFO_STRICT", cap(4, 0, 0), Some(600.0), false, true).unwrap();
        assert_eq!(report.metrics.seconds_above_power_cap, Some(0));
        assert_eq!(report.metrics.joules_above_power_cap, Some(0.0));
    }

    #[test]
    fn power_integrates_idle_gaps_as_zero_draw() {
        // Job 1 runs 0-100 (500 W); job 2 submits at 300, runs to 400 (500 W).
        // Energy 100 kJ over a 400 s makespan -> mean 250 W, peak 500 W.
        let jobs = vec![rec_power(1, 0, 100, 4, 500.0), rec_power(2, 300, 100, 4, 500.0)];
        let report = simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), None, false, true).unwrap();
        assert_eq!(report.metrics.energy_joules_total, Some(100_000.0));
        assert_eq!(report.metrics.power_peak_watts, Some(500.0));
        assert_eq!(report.metrics.power_mean_watts, Some(250.0));
    }

    // ── Cap-enforcing dispatch tests ───────────────────────────

    #[test]
    fn enforced_cap_serializes_jobs_and_eliminates_exposure() {
        // Two 400 W jobs under a 600 W enforced cap: they must run back to
        // back even though CPUs allow concurrency. Zero time above cap; the
        // BSLD price is the second job's 100 s wait.
        let jobs = vec![rec_power(1, 0, 100, 4, 400.0), rec_power(2, 0, 100, 4, 400.0)];
        let report =
            simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), Some(600.0), true, true).unwrap();
        assert_eq!(report.metrics.jobs_completed, 2);
        assert_eq!(report.metrics.power_cap_enforced, Some(true));
        assert_eq!(report.metrics.seconds_above_power_cap, Some(0));
        assert_eq!(report.metrics.joules_above_power_cap, Some(0.0));
        assert_eq!(report.metrics.power_peak_watts, Some(400.0));
        assert_eq!(report.metrics.mean_wait_sec, 50.0);
        assert_eq!(report.metrics.makespan_sec, 200);
        // Energy unchanged by enforcement.
        assert_eq!(report.metrics.energy_joules_total, Some(80_000.0));
    }

    #[test]
    fn easy_backfill_respects_power_shadow_time() {
        // Cap 1000 W enforced. Job1: 900 W, 100 s (running). Head job2 needs
        // 900 W -> blocked by power alone (CPUs are free), shadow = 100.
        // Job3: 50 W, 50 s -> fits headroom (100 W) and ends before shadow:
        // backfills. Job4: 50 W, 200 s -> would outlive the shadow: must not.
        let jobs = vec![
            rec_power(1, 0, 100, 1, 900.0),
            rec_power(2, 1, 100, 1, 900.0),
            rec_power(3, 1, 50, 1, 50.0),
            rec_power(4, 1, 200, 1, 50.0),
        ];
        let report =
            simulate(jobs, "EASY_BACKFILL_BASELINE", cap(64, 0, 0), Some(1000.0), true, true)
                .unwrap();
        assert_eq!(report.metrics.jobs_completed, 4);
        assert_eq!(report.metrics.seconds_above_power_cap, Some(0));
        // Waits: j1=0, j2=99 (starts at 100), j3=0 (backfilled at t=1),
        // j4 starts at t=200 when j2 finishes... j4 only needs 50 W: at
        // t=100 draw is 900 (j2) -> headroom 100 -> j4 starts at 100.
        // Waits: 0 + 99 + 0 + 99 = 198 -> mean 49.5.
        assert_eq!(report.metrics.mean_wait_sec, 49.5);
    }

    #[test]
    fn measurement_only_cap_does_not_change_dispatch() {
        let jobs = || vec![rec_power(1, 0, 100, 4, 400.0), rec_power(2, 0, 100, 4, 400.0)];
        let unmetered = simulate(jobs(), "FIFO_STRICT", cap(8, 0, 0), None, false, true).unwrap();
        let metered =
            simulate(jobs(), "FIFO_STRICT", cap(8, 0, 0), Some(600.0), false, true).unwrap();
        assert_eq!(unmetered.metrics.mean_wait_sec, metered.metrics.mean_wait_sec);
        assert_eq!(unmetered.metrics.makespan_sec, metered.metrics.makespan_sec);
        assert_eq!(metered.metrics.power_cap_enforced, Some(false));
        assert_eq!(metered.metrics.seconds_above_power_cap, Some(100));
    }

    #[test]
    fn job_drawing_more_than_the_cap_never_starts() {
        let jobs = vec![rec_power(1, 0, 100, 4, 900.0), rec_power(2, 0, 100, 4, 100.0)];
        let report =
            simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), Some(600.0), true, true).unwrap();
        // FIFO head blocks forever; the sim terminates and reports honestly.
        assert_eq!(report.metrics.jobs_completed, 0);
        assert_eq!(report.metrics.jobs_total, 2);
    }

    #[test]
    fn enforce_without_cap_is_an_error() {
        let jobs = vec![rec(1, 0, 100, 4, 0, 0)];
        assert!(simulate(jobs, "FIFO_STRICT", cap(8, 0, 0), None, true, true).is_err());
    }

    #[test]
    fn conservation_invariant_covers_all_dimensions() {
        let jobs = vec![rec(1, 0, 10, 2, 1, 500)];
        let report = simulate(jobs, "FIFO_STRICT", cap(4, 2, 1000), None, false, true).unwrap();
        assert_eq!(report.invariant_report.total_violations, 0);
    }
}
