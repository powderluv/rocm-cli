use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, AutomationEventRecord, AutomationRuntimeState, DEFAULT_LOCAL_HOST,
    ManagedServiceRecord, RocmCliConfig, WatcherMode, WatcherRuntimeSnapshot,
    append_automation_event, builtin_watchers, daemon_binary_path, engine_binary_path,
    unix_time_millis,
};
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;

#[derive(Parser, Debug)]
#[command(name = "rocmd", about = "rocm-cli local supervisor")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Run {
        #[arg(long)]
        automations_enabled: bool,
    },
    Supervise {
        service_id: String,
        #[arg(long)]
        engine: String,
        #[arg(long)]
        model_ref: String,
        #[arg(long)]
        canonical_model_id: String,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
        host: String,
        #[arg(long)]
        port: u16,
        #[arg(long, default_value = "gpu_preferred")]
        device_policy: String,
    },
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::discover()?;

    match cli.command.unwrap_or(Command::Status) {
        Command::Run {
            automations_enabled,
        } => run_daemon(&paths, automations_enabled).await?,
        Command::Supervise {
            service_id,
            engine,
            model_ref,
            canonical_model_id,
            runtime_id,
            env_id,
            host,
            port,
            device_policy,
        } => supervise_service(
            &paths,
            service_id,
            engine,
            model_ref,
            canonical_model_id,
            runtime_id,
            env_id,
            host,
            port,
            device_policy,
        )?,
        Command::Status => {
            print_status(&paths)?;
        }
    }

    Ok(())
}

async fn run_daemon(paths: &AppPaths, automations_enabled: bool) -> Result<()> {
    let config = RocmCliConfig::load(paths)?;
    let mut state = build_runtime_state(&config, automations_enabled);

    println!("rocmd run");
    println!("  automations enabled: {automations_enabled}");
    println!(
        "  lifecycle: {}",
        if automations_enabled {
            "persistent"
        } else {
            "on-demand"
        }
    );
    println!("  config: {}", paths.config_path().display());
    println!("  state: {}", paths.automation_state_path().display());
    let enabled_count = state
        .active_watchers
        .iter()
        .filter(|watcher| watcher.enabled)
        .count();
    println!("  enabled watchers: {enabled_count}");

    if !automations_enabled {
        println!(
            "  note: rerun with --automations-enabled to keep rocmd alive for watcher execution"
        );
        return Ok(());
    }

    paths.ensure()?;
    state.write(paths)?;
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let mut ticker = time::interval(WATCHER_TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let config = RocmCliConfig::load(paths)?;
                reconcile_watcher_snapshots(&config, &mut state);
                evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            _ = &mut shutdown => {
                break;
            }
        }
    }

    state.running = false;
    state.last_tick_unix_ms = unix_time_millis();
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_stop",
        "rocmd automation supervisor stopped",
        None,
    )?;
    state.write(paths)?;
    Ok(())
}

fn print_status(paths: &AppPaths) -> Result<()> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  policy: on-demand by default, persistent only with background features");
    println!(
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match AutomationRuntimeState::load(paths)? {
        Some(state) => {
            println!(
                "  automations runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
            for watcher in state
                .active_watchers
                .into_iter()
                .filter(|watcher| watcher.enabled)
            {
                println!(
                    "  watcher {} mode={} last_event={}",
                    watcher.id,
                    watcher.mode.as_str(),
                    watcher.last_event.as_deref().unwrap_or("<none>")
                );
            }
        }
        None => println!("  automations runtime: inactive"),
    }
    println!(
        "  automation events: {}",
        paths.automation_events_path().display()
    );

    let records = load_managed_services(paths)?;
    if records.is_empty() {
        println!("  services: none");
        return Ok(());
    }

    for record in records {
        println!(
            "  service {} engine={} status={} endpoint={}",
            record.service_id, record.engine, record.status, record.endpoint_url
        );
    }

    Ok(())
}

fn supervise_service(
    paths: &AppPaths,
    service_id: String,
    engine: String,
    model_ref: String,
    canonical_model_id: String,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    device_policy: String,
) -> Result<()> {
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(&engine))?;
    fs::create_dir_all(paths.engine_state_dir(&engine))?;
    fs::create_dir_all(paths.services_dir())?;

    let _ = daemon_binary_path();

    let mut record = ManagedServiceRecord::new(
        paths,
        service_id,
        engine.clone(),
        model_ref,
        canonical_model_id.clone(),
        host.clone(),
        port,
        "managed",
        std::process::id(),
        runtime_id.clone(),
        env_id.clone(),
        Some(device_policy.clone()),
    );
    record.write()?;

    let log_file = fs::File::create(&record.log_path)
        .with_context(|| format!("failed to create {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    let engine_binary = engine_binary_path(&engine)?;
    let mut child = ProcessCommand::new(engine_binary)
        .arg("serve-http")
        .arg(&record.service_id)
        .arg(&canonical_model_id)
        .arg("--host")
        .arg(&record.host)
        .arg("--port")
        .arg(record.port.to_string())
        .arg("--device-policy")
        .arg(&device_policy)
        .args(optional_arg("--runtime-id", runtime_id.as_deref()))
        .args(optional_arg("--env-id", env_id.as_deref()))
        .arg("--state-path")
        .arg(&record.engine_state_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .with_context(|| format!("failed to spawn engine supervisor child for {}", engine))?;

    record.engine_pid = Some(child.id());
    record.status = "running".to_owned();
    record.write()?;

    if wait_for_port(&record.host, record.port, Duration::from_secs(5)) {
        record.status = "ready".to_owned();
        record.write()?;
    }

    let exit_status = child.wait().context("failed waiting for engine child")?;
    record.status = if exit_status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;

    if exit_status.success() {
        Ok(())
    } else {
        std::process::exit(exit_status.code().unwrap_or(1));
    }
}

fn build_runtime_state(
    config: &RocmCliConfig,
    automations_enabled: bool,
) -> AutomationRuntimeState {
    let now = unix_time_millis();
    let active_watchers = builtin_watchers()
        .iter()
        .map(|watcher| WatcherRuntimeSnapshot {
            id: watcher.id.to_owned(),
            enabled: config.watcher_enabled(watcher),
            mode: config.effective_watcher_mode(watcher),
            summary: watcher.summary.to_owned(),
            last_event: None,
            last_event_unix_ms: None,
        })
        .collect();
    AutomationRuntimeState {
        running: automations_enabled,
        automations_enabled,
        daemon_pid: std::process::id(),
        started_at_unix_ms: now,
        last_tick_unix_ms: now,
        active_watchers,
    }
}

fn reconcile_watcher_snapshots(config: &RocmCliConfig, state: &mut AutomationRuntimeState) {
    for watcher in builtin_watchers() {
        match state.watcher_mut(watcher.id) {
            Some(snapshot) => {
                snapshot.enabled = config.watcher_enabled(watcher);
                snapshot.mode = config.effective_watcher_mode(watcher);
                snapshot.summary = watcher.summary.to_owned();
            }
            None => state.active_watchers.push(WatcherRuntimeSnapshot {
                id: watcher.id.to_owned(),
                enabled: config.watcher_enabled(watcher),
                mode: config.effective_watcher_mode(watcher),
                summary: watcher.summary.to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }),
        }
    }
}

fn evaluate_watchers(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    for watcher in builtin_watchers() {
        if !config.watcher_enabled(watcher) {
            continue;
        }
        match watcher.id {
            "therock-update" => {
                evaluate_therock_update(paths, config.effective_watcher_mode(watcher), state)?
            }
            "server-recover" => {
                evaluate_server_recover(paths, config.effective_watcher_mode(watcher), state)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn evaluate_therock_update(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let last_event_unix_ms = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "therock-update")
        .and_then(|watcher| watcher.last_event_unix_ms);
    let now = unix_time_millis();
    if let Some(last_event) = last_event_unix_ms
        && now.saturating_sub(last_event) < THEROCK_UPDATE_INTERVAL_MS
    {
        return Ok(());
    }

    let action = match mode {
        WatcherMode::Observe => "observe_schedule",
        WatcherMode::Propose => "queue_update_proposal",
        WatcherMode::Contained => "queue_update_proposal",
    };
    let message = if mode == WatcherMode::Contained {
        "scheduled TheRock update check reminder emitted; contained mode falls back to proposal until SDK updates are fully wired"
    } else {
        "scheduled TheRock update check reminder emitted; run `rocm update` to inspect the selected channel"
    };
    record_event(
        paths,
        state,
        "therock-update",
        "info",
        action,
        message,
        None,
    )
}

fn evaluate_server_recover(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let last_event_unix_ms = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "server-recover")
        .and_then(|watcher| watcher.last_event_unix_ms);
    let now = unix_time_millis();
    if let Some(last_event) = last_event_unix_ms
        && now.saturating_sub(last_event) < SERVER_RECOVER_BACKOFF_MS
    {
        return Ok(());
    }

    let Some(mut record) = load_managed_services(paths)?
        .into_iter()
        .find(|record| record.mode == "managed" && record.status == "failed")
    else {
        return Ok(());
    };

    match mode {
        WatcherMode::Observe => record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "observe_failure",
            &format!(
                "observed failed managed service {}; restart not attempted in observe mode",
                record.service_id
            ),
            Some(record.service_id.clone()),
        ),
        WatcherMode::Propose => record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "queue_restart_proposal",
            &format!(
                "managed service {} failed; queueing restart proposal",
                record.service_id
            ),
            Some(record.service_id.clone()),
        ),
        WatcherMode::Contained => {
            if let Some(last_restart) = record.last_restart_unix_ms
                && now.saturating_sub(last_restart) < SERVER_RECOVER_BACKOFF_MS
            {
                return Ok(());
            }
            restart_managed_service(paths, &mut record)?;
            record_event(
                paths,
                state,
                "server-recover",
                "info",
                "restart_managed_service",
                &format!(
                    "restarted failed managed service {} on {}:{}",
                    record.service_id, record.host, record.port
                ),
                Some(record.service_id.clone()),
            )
        }
    }
}

fn restart_managed_service(_paths: &AppPaths, record: &mut ManagedServiceRecord) -> Result<()> {
    let rocmd_binary =
        std::env::current_exe().context("failed to resolve current rocmd executable path")?;
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    record.status = "recovering".to_owned();
    record.restart_count = record.restart_count.saturating_add(1);
    record.last_restart_unix_ms = Some(unix_time_millis());
    record.supervisor_pid = std::process::id();
    record.write()?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .arg("supervise")
        .arg(&record.service_id)
        .arg("--engine")
        .arg(&record.engine)
        .arg("--model-ref")
        .arg(&record.model_ref)
        .arg("--canonical-model-id")
        .arg(&record.canonical_model_id)
        .arg("--host")
        .arg(&record.host)
        .arg("--port")
        .arg(record.port.to_string())
        .arg("--device-policy")
        .arg(record.device_policy.as_deref().unwrap_or("gpu_preferred"))
        .args(optional_arg("--runtime-id", record.runtime_id.as_deref()))
        .args(optional_arg("--env-id", record.env_id.as_deref()))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    record.supervisor_pid = child.id();
    record.write()?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        record.status = "failed".to_owned();
        record.write()?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

fn record_event(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
    watcher_id: &str,
    level: &str,
    action: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    let now = unix_time_millis();
    if let Some(snapshot) = state.watcher_mut(watcher_id) {
        snapshot.last_event = Some(message.to_owned());
        snapshot.last_event_unix_ms = Some(now);
    }
    append_automation_event(
        paths,
        &AutomationEventRecord {
            at_unix_ms: now,
            watcher_id: watcher_id.to_owned(),
            level: level.to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            service_id,
        },
    )
}

fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) {
            records.push(record);
        }
    }

    records.sort_by(|left, right| right.created_at_unix_ms.cmp(&left.created_at_unix_ms));
    Ok(records)
}

#[cfg(unix)]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    let mut command = ProcessCommand::new("setsid");
    command.arg(rocmd_binary);
    command
}

#[cfg(not(unix))]
fn detached_rocmd_command(rocmd_binary: &std::path::Path) -> ProcessCommand {
    ProcessCommand::new(rocmd_binary)
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

fn wait_for_port(host: &str, port: u16, timeout: Duration) -> bool {
    let address: SocketAddr = match format!("{host}:{port}").parse() {
        Ok(value) => value,
        Err(_) => return false,
    };

    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}
