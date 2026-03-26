use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_HOST, ManagedServiceRecord, daemon_binary_path, engine_binary_path,
};
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::discover()?;

    match cli.command.unwrap_or(Command::Status) {
        Command::Run {
            automations_enabled,
        } => {
            println!("rocmd run scaffold");
            println!("  automations enabled: {automations_enabled}");
            println!(
                "  lifecycle: {}",
                if automations_enabled {
                    "persistent"
                } else {
                    "on-demand"
                }
            );
            println!("  data dir: {}", paths.data_dir.display());
        }
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

fn print_status(paths: &AppPaths) -> Result<()> {
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  policy: on-demand by default, persistent only with background features");

    let services_dir = paths.services_dir();
    if !services_dir.exists() {
        println!("  services: none");
        return Ok(());
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
        let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(record) = serde_json::from_slice::<ManagedServiceRecord>(&data) {
            records.push(record);
        }
    }

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
