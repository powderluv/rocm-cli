use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_HOST, DoctorSummary, ManagedServiceRecord, RocmCliConfig,
    daemon_binary_path, default_engine_for_platform, engine_binary_path, generate_service_id,
    interactive_terminal,
};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope, InstallRequest,
    InstallResponse, ResolveModelRequest, ResolveModelResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

const DEFAULT_RUNTIME_ID: &str = "therock-release";

#[derive(Parser, Debug)]
#[command(name = "rocm", about = "TheRock-oriented local AI control plane")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Doctor,
    Chat {
        #[arg(long)]
        provider: Option<Provider>,
    },
    Install {
        #[command(subcommand)]
        target: InstallTarget,
    },
    Update,
    Engines {
        #[command(subcommand)]
        command: EnginesCommand,
    },
    Serve {
        model: String,
        #[arg(long)]
        engine: Option<String>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
        host: String,
        #[arg(long, default_value_t = rocm_core::DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long, conflicts_with = "managed")]
        foreground: bool,
        #[arg(long)]
        managed: bool,
    },
    Automations {
        #[command(subcommand)]
        command: Option<AutomationsCommand>,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Logs,
    Daemon,
}

#[derive(Subcommand, Debug)]
enum InstallTarget {
    Sdk {
        #[arg(long, default_value = "release")]
        channel: String,
        #[arg(long, default_value = "pip")]
        format: InstallFormat,
        #[arg(long)]
        prefix: Option<std::path::PathBuf>,
    },
    Driver {
        #[arg(long)]
        dkms: bool,
    },
}

#[derive(Subcommand, Debug)]
enum EnginesCommand {
    List,
    Install {
        engine: String,
        #[arg(long, default_value = DEFAULT_RUNTIME_ID)]
        runtime_id: String,
        #[arg(long)]
        python_version: Option<String>,
        #[arg(long)]
        reinstall: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AutomationsCommand {
    List,
    Enable { watcher: String },
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    Show,
    SetEngine {
        engine: String,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long)]
        clear: bool,
    },
    SetDefaultEngine {
        engine: String,
    },
    ClearDefaultEngine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InstallFormat {
    Pip,
    Tarball,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Provider {
    Local,
    Anthropic,
    Openai,
}

fn main() -> Result<()> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() {
        return launch_default();
    }

    if treat_as_natural_language(&raw_args) {
        return run_freeform(raw_args.join(" "));
    }

    dispatch(Cli::parse())
}

fn launch_default() -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    println!("rocm interactive mode scaffold");
    println!(
        "  terminal: {}",
        if interactive_terminal() {
            "interactive"
        } else {
            "non-interactive"
        }
    );
    println!("  default engine: {selected_default_engine}");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  config file: {}", paths.config_path().display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  cache dir: {}", paths.cache_dir.display());
    println!("  note: TUI is not implemented yet in this scaffold.");
    Ok(())
}

fn run_freeform(request: String) -> Result<()> {
    println!("natural-language request captured");
    println!("  request: {request}");
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    println!(
        "  default engine: {}",
        config
            .default_engine
            .as_deref()
            .unwrap_or(default_engine_for_platform())
    );
    println!(
        "  note: planner execution is not wired yet; this scaffold only captures the request."
    );
    Ok(())
}

fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Command::Doctor) => doctor(),
        Some(Command::Chat { provider }) => {
            println!("chat scaffold");
            println!(
                "  provider: {}",
                provider.map(provider_name).unwrap_or("local")
            );
            println!("  note: provider-backed chat will be layered on top of the TUI.");
            Ok(())
        }
        Some(Command::Install { target }) => install(target),
        Some(Command::Update) => {
            println!("update scaffold");
            println!("  policy: check every run, prompt before mutating state.");
            Ok(())
        }
        Some(Command::Engines { command }) => engines(command),
        Some(Command::Serve {
            model,
            engine,
            device,
            runtime_id,
            env_id,
            host,
            port,
            foreground,
            managed,
        }) => serve(
            model, engine, device, runtime_id, env_id, host, port, foreground, managed,
        ),
        Some(Command::Automations { command }) => automations(command),
        Some(Command::Config { command }) => config(command),
        Some(Command::Logs) => {
            let paths = AppPaths::discover()?;
            println!("logs dir: {}", paths.data_dir.join("logs").display());
            Ok(())
        }
        Some(Command::Daemon) => {
            println!("rocmd lifecycle");
            println!("  default: on-demand");
            println!(
                "  persistent: only when automations or managed background services are enabled"
            );
            Ok(())
        }
        None => launch_default(),
    }
}

fn doctor() -> Result<()> {
    let summary = DoctorSummary::gather()?;
    print!("{}", summary.render_text());
    Ok(())
}

fn install(target: InstallTarget) -> Result<()> {
    let paths = AppPaths::discover()?;
    match target {
        InstallTarget::Sdk {
            channel,
            format,
            prefix,
        } => {
            println!("sdk install plan");
            println!("  channel: {channel}");
            println!(
                "  format: {}",
                match format {
                    InstallFormat::Pip => "pip",
                    InstallFormat::Tarball => "tarball",
                }
            );
            println!(
                "  target: {}",
                prefix
                    .as_ref()
                    .map(|value| value.display().to_string())
                    .unwrap_or_else(|| paths
                        .data_dir
                        .join("runtimes/therock")
                        .display()
                        .to_string())
            );
            println!("  note: installer implementation is not wired yet.");
        }
        InstallTarget::Driver { dkms } => {
            println!("driver install policy");
            println!("  dkms requested: {dkms}");
            if cfg!(target_os = "windows") {
                println!("  windows: driver validation only in early phases");
            } else {
                println!("  linux: wrap official AMD DKMS flows, never silent upgrades");
            }
        }
    }
    Ok(())
}

fn engines(command: EnginesCommand) -> Result<()> {
    match command {
        EnginesCommand::List => {
            let default_engine = default_engine_for_platform();
            println!("engine inventory");
            for (name, note) in [
                ("pytorch", "default Windows local serving engine"),
                ("llama.cpp", "CPU and quantized fallback engine"),
                ("vllm", "default Linux ROCm GPU serving engine"),
                ("sglang", "deferred advanced serving engine"),
                ("atom", "deferred AMD-optimized serving engine"),
            ] {
                let marker = if name == default_engine { "*" } else { " " };
                println!("{marker} {name:10} {note}");
            }
            println!(
                "  protocol: {}",
                rocm_engine_protocol::ENGINE_PROTOCOL_VERSION
            );
            Ok(())
        }
        EnginesCommand::Install {
            engine,
            runtime_id,
            python_version,
            reinstall,
        } => {
            let response = engine_request::<_, InstallResponse>(
                &engine,
                EngineMethod::Install,
                &InstallRequest {
                    runtime_id: runtime_id.clone(),
                    python_version: python_version.clone(),
                    reinstall,
                },
            )?;
            println!("engine install");
            println!("  engine: {engine}");
            println!("  runtime_id: {runtime_id}");
            println!(
                "  python_version: {}",
                python_version.as_deref().unwrap_or("default")
            );
            println!("  reinstall: {reinstall}");
            println!("  env_id: {}", response.env_id);
            println!("  env_path: {}", response.env_path);
            println!("  python_executable: {}", response.python_executable);
            println!("  lock_hash: {}", response.lock_hash);
            println!(
                "  packages: {}",
                summarize_packages(&response.installed_packages)
            );
            println!(
                "  capabilities: cpu={} rocm_gpu={} openai_compatible={}",
                response.capabilities.cpu,
                response.capabilities.rocm_gpu,
                response.capabilities.openai_compatible
            );
            for warning in response.warnings {
                println!("  warning: {warning}");
            }
            let paths = AppPaths::discover()?;
            let mut config = RocmCliConfig::load(&paths)?;
            let engine_config = config.engine_config_mut(&engine);
            engine_config.last_installed_runtime_id = Some(runtime_id);
            engine_config.last_installed_env_id = Some(response.env_id.clone());
            let mut seeded_preference = false;
            if engine_config.preferred_runtime_id.is_none()
                && engine_config.preferred_env_id.is_none()
            {
                engine_config.preferred_env_id = Some(response.env_id.clone());
                seeded_preference = true;
            }
            config.save(&paths)?;
            println!("  config: recorded last installed env for {engine}");
            if seeded_preference {
                println!("  config: seeded preferred env from first successful install");
            }
            Ok(())
        }
    }
}

fn serve(
    model: String,
    engine: Option<String>,
    device: Option<String>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    host: String,
    port: u16,
    foreground: bool,
    managed: bool,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    let selected_engine = engine
        .or_else(|| config.default_engine.clone())
        .unwrap_or_else(|| default_engine_for_platform().to_owned());
    let resolved_selection = resolve_engine_selection(
        &config,
        &selected_engine,
        runtime_id.as_deref(),
        env_id.as_deref(),
    );
    let device_policy = parse_device_policy(device.as_deref())?;
    let resolve = engine_request::<_, ResolveModelResponse>(
        &selected_engine,
        EngineMethod::ResolveModel,
        &ResolveModelRequest {
            model_ref: model.clone(),
            runtime_id: resolved_selection.runtime_id.clone(),
            device_policy: Some(device_policy.clone()),
            recipe_override: None,
        },
    )?;
    let service_id = generate_service_id(&selected_engine, &resolve.canonical_model_id);

    println!("serve plan");
    println!("  requested model: {model}");
    println!("  resolved model: {}", resolve.canonical_model_id);
    println!("  engine: {selected_engine}");
    println!("  host: {host}");
    println!("  port: {port}");
    if let Some(runtime_id) = resolved_selection.runtime_id.as_deref() {
        println!("  runtime_id: {runtime_id}");
    }
    if let Some(env_id) = resolved_selection.env_id.as_deref() {
        println!("  env_id: {env_id}");
    }
    if let Some(source) = resolved_selection.source.as_deref() {
        println!("  selection_source: {source}");
    }
    println!(
        "  device_policy: {}",
        device_policy_name(&resolve.device_policy)
    );

    if managed {
        return start_managed_service(
            &selected_engine,
            &service_id,
            &model,
            &resolve,
            &host,
            port,
            &resolve.device_policy,
            resolved_selection.runtime_id.as_deref(),
            resolved_selection.env_id.as_deref(),
        );
    }

    if !foreground {
        println!("  mode: foreground (default)");
        println!("  note: use --managed to hand supervision to rocmd.");
    }
    run_foreground_service(
        &selected_engine,
        &service_id,
        &resolve.canonical_model_id,
        &host,
        port,
        &resolve.device_policy,
        resolved_selection.runtime_id.as_deref(),
        resolved_selection.env_id.as_deref(),
    )
}

fn start_managed_service(
    engine: &str,
    service_id: &str,
    requested_model: &str,
    resolve: &ResolveModelResponse,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.services_dir())?;

    let rocmd_binary = daemon_binary_path().with_context(
        || "managed mode requires the rocmd binary; use --foreground if only the CLI is installed",
    )?;
    let log_path = paths.service_log_path(service_id);
    let log_file = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone managed service log file handle")?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .arg("supervise")
        .arg(service_id)
        .arg("--engine")
        .arg(engine)
        .arg("--model-ref")
        .arg(requested_model)
        .arg("--canonical-model-id")
        .arg(&resolve.canonical_model_id)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(device_policy))
        .args(optional_arg("--runtime-id", runtime_id))
        .args(optional_arg("--env-id", env_id))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to launch rocmd supervisor process")?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check rocmd startup state")?
    {
        bail!(
            "rocmd exited immediately with status {status}; inspect {}",
            log_path.display()
        );
    }

    let endpoint_url = format!("http://{host}:{port}/v1");
    let readiness = wait_for_port(host, port, Duration::from_secs(5));
    let manifest_path = paths.service_manifest_path(service_id);
    println!("managed service launched");
    println!("  service_id: {service_id}");
    println!("  supervisor_pid: {}", child.id());
    println!("  endpoint: {endpoint_url}");
    println!("  log_path: {}", log_path.display());
    println!("  manifest_path: {}", manifest_path.display());
    println!(
        "  readiness: {}",
        if readiness { "ready" } else { "starting" }
    );
    Ok(())
}

fn run_foreground_service(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.engine_state_dir(engine))?;

    let mut record = ManagedServiceRecord::new(
        &paths,
        service_id,
        engine,
        canonical_model_id,
        canonical_model_id,
        host,
        port,
        "foreground",
        std::process::id(),
    );
    record.write()?;

    let engine_binary = engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;

    println!("foreground service starting");
    println!("  service_id: {service_id}");
    println!("  endpoint: http://{host}:{port}/v1");
    println!("  stop: Ctrl-C");

    let mut child = ProcessCommand::new(engine_binary)
        .arg("serve-http")
        .arg(service_id)
        .arg(canonical_model_id)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(device_policy))
        .args(optional_arg("--runtime-id", runtime_id))
        .args(optional_arg("--env-id", env_id))
        .arg("--state-path")
        .arg(&record.engine_state_path)
        .spawn()
        .context("failed to start foreground engine service")?;

    record.engine_pid = Some(child.id());
    record.status = "running".to_owned();
    record.write()?;
    if wait_for_port(host, port, Duration::from_secs(5)) {
        record.status = "ready".to_owned();
        record.write()?;
    }

    let status = child
        .wait()
        .context("failed waiting for foreground engine")?;
    record.status = if status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;

    if status.success() {
        Ok(())
    } else {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn automations(command: Option<AutomationsCommand>) -> Result<()> {
    match command.unwrap_or(AutomationsCommand::List) {
        AutomationsCommand::List => {
            println!("automation watchers");
            println!("  therock-update    observe");
            println!("  server-recover    contained");
            println!("  note: native watcher execution is not implemented yet.");
        }
        AutomationsCommand::Enable { watcher } => {
            println!("enable watcher");
            println!("  watcher: {watcher}");
            println!("  note: watcher persistence is not implemented yet.");
        }
    }
    Ok(())
}

fn config(command: ConfigCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;

    match command {
        ConfigCommand::Show => {
            println!("rocm config");
            println!("  file: {}", paths.config_path().display());
            println!(
                "  default_engine: {}",
                config
                    .default_engine
                    .as_deref()
                    .unwrap_or("<platform default>")
            );
            if config.engines.is_empty() {
                println!("  engines: none");
                return Ok(());
            }
            for (engine, entry) in &config.engines {
                println!("  engine: {engine}");
                println!(
                    "    preferred_runtime_id: {}",
                    entry.preferred_runtime_id.as_deref().unwrap_or("<unset>")
                );
                println!(
                    "    preferred_env_id: {}",
                    entry.preferred_env_id.as_deref().unwrap_or("<unset>")
                );
                println!(
                    "    last_installed_runtime_id: {}",
                    entry
                        .last_installed_runtime_id
                        .as_deref()
                        .unwrap_or("<unset>")
                );
                println!(
                    "    last_installed_env_id: {}",
                    entry.last_installed_env_id.as_deref().unwrap_or("<unset>")
                );
            }
        }
        ConfigCommand::SetEngine {
            engine,
            runtime_id,
            env_id,
            clear,
        } => {
            let entry = config.engine_config_mut(&engine);
            if clear {
                entry.preferred_runtime_id = None;
                entry.preferred_env_id = None;
            } else if let Some(runtime_id) = runtime_id {
                entry.preferred_runtime_id = Some(runtime_id);
                entry.preferred_env_id = None;
            } else if let Some(env_id) = env_id {
                entry.preferred_env_id = Some(env_id);
                entry.preferred_runtime_id = None;
            } else {
                bail!("set-engine requires --runtime-id, --env-id, or --clear");
            }
            config.save(&paths)?;
            println!("updated engine config for {engine}");
        }
        ConfigCommand::SetDefaultEngine { engine } => {
            config.default_engine = Some(engine.clone());
            config.save(&paths)?;
            println!("default engine set to {engine}");
        }
        ConfigCommand::ClearDefaultEngine => {
            config.default_engine = None;
            config.save(&paths)?;
            println!("default engine cleared");
        }
    }

    Ok(())
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Local => "local",
        Provider::Anthropic => "anthropic",
        Provider::Openai => "openai",
    }
}

fn engine_request<T, R>(engine: &str, method: EngineMethod, request: &T) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let engine_binary = engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request)
            .context("failed to encode engine request payload")?,
    };

    let mut child = ProcessCommand::new(engine_binary)
        .arg("stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn engine stdio process")?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .context("engine stdio child did not expose stdin")?;
        serde_json::to_writer(&mut *stdin, &envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
    }

    let output = child
        .wait_with_output()
        .context("failed waiting for engine stdio response")?;
    if !output.status.success() && output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {}", output.status);
        }
        bail!(
            "engine stdio process exited with status {}: {}",
            output.status,
            stderr
        );
    }
    let envelope: EngineResponseEnvelope =
        serde_json::from_slice(&output.stdout).with_context(|| {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if stderr.is_empty() {
                "failed to parse engine response envelope".to_owned()
            } else {
                format!("failed to parse engine response envelope; stderr: {stderr}")
            }
        })?;
    if !envelope.ok {
        let error = envelope
            .error
            .map(|value| format!("{}: {}", value.code, value.message))
            .unwrap_or_else(|| "unknown engine error".to_owned());
        bail!("{error}");
    }

    let data = envelope
        .data
        .context("engine response envelope did not contain data")?;
    serde_json::from_value(data).context("failed to decode engine response payload")
}

fn parse_device_policy(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("auto") {
        "auto" | "gpu_preferred" | "gpu" => Ok(DevicePolicy::GpuPreferred),
        "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unsupported device policy: {other}"),
    }
}

fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn summarize_packages(packages: &[String]) -> String {
    let interesting = ["torch", "transformers", "accelerate", "fastapi"];
    let selected = interesting
        .iter()
        .filter_map(|name| {
            packages
                .iter()
                .find(|entry| entry.starts_with(&format!("{name}==")))
        })
        .cloned()
        .collect::<Vec<_>>();

    if selected.is_empty() {
        format!("{} captured in lockfile", packages.len())
    } else {
        format!(
            "{} captured in lockfile; {}",
            packages.len(),
            selected.join(", ")
        )
    }
}

#[derive(Debug, Clone)]
struct EngineSelection {
    runtime_id: Option<String>,
    env_id: Option<String>,
    source: Option<String>,
}

fn resolve_engine_selection(
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> EngineSelection {
    if let Some(env_id) = env_id {
        return EngineSelection {
            runtime_id: None,
            env_id: Some(env_id.to_owned()),
            source: Some("cli_env_id".to_owned()),
        };
    }
    if let Some(runtime_id) = runtime_id {
        return EngineSelection {
            runtime_id: Some(runtime_id.to_owned()),
            env_id: None,
            source: Some("cli_runtime_id".to_owned()),
        };
    }

    let Some(entry) = config.engine_config(engine) else {
        return EngineSelection {
            runtime_id: None,
            env_id: None,
            source: None,
        };
    };

    if let Some(env_id) = entry.preferred_env_id.as_ref() {
        return EngineSelection {
            runtime_id: None,
            env_id: Some(env_id.clone()),
            source: Some("config_preferred_env_id".to_owned()),
        };
    }
    if let Some(runtime_id) = entry.preferred_runtime_id.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_id.clone()),
            env_id: None,
            source: Some("config_preferred_runtime_id".to_owned()),
        };
    }
    if let Some(env_id) = entry.last_installed_env_id.as_ref() {
        return EngineSelection {
            runtime_id: None,
            env_id: Some(env_id.clone()),
            source: Some("config_last_installed_env_id".to_owned()),
        };
    }
    if let Some(runtime_id) = entry.last_installed_runtime_id.as_ref() {
        return EngineSelection {
            runtime_id: Some(runtime_id.clone()),
            env_id: None,
            source: Some("config_last_installed_runtime_id".to_owned()),
        };
    }

    EngineSelection {
        runtime_id: None,
        env_id: None,
        source: None,
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

fn treat_as_natural_language(args: &[String]) -> bool {
    const STRUCTURED: &[&str] = &[
        "doctor",
        "chat",
        "install",
        "update",
        "engines",
        "serve",
        "automations",
        "config",
        "logs",
        "daemon",
        "help",
        "--help",
        "-h",
        "version",
        "--version",
        "-V",
    ];

    !args.is_empty() && !args[0].starts_with('-') && !STRUCTURED.contains(&args[0].as_str())
}
