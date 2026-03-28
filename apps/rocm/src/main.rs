mod therock;
mod tui;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, AutomationRuntimeState, DEFAULT_LOCAL_HOST, DoctorSummary, ManagedServiceRecord,
    RocmCliConfig, WatcherMode, builtin_watcher, builtin_watchers, daemon_binary_path,
    default_engine_for_platform, engine_binary_path, generate_service_id, interactive_terminal,
    load_recent_automation_events,
};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope, InstallRequest,
    InstallResponse, ResolveModelRequest, ResolveModelResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;

const DEFAULT_RUNTIME_ID: &str = "therock-release";

#[derive(Parser, Debug)]
#[command(name = "rocm", about = "ROCm AI Command Center CLI", version)]
struct Cli {
    #[arg(long, global = true, hide = true)]
    experimental_codex_tui: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Doctor,
    Version,
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
    Uninstall {
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        keep_binaries: bool,
        #[arg(long)]
        keep_config: bool,
        #[arg(long)]
        keep_data: bool,
        #[arg(long)]
        keep_cache: bool,
        #[arg(long)]
        force_dev_binaries: bool,
    },
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
        #[arg(long)]
        dry_run: bool,
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
    Shell {
        engine: String,
        #[arg(long, conflicts_with = "env_id")]
        runtime_id: Option<String>,
        #[arg(long, conflicts_with = "runtime_id")]
        env_id: Option<String>,
        #[arg(long)]
        shell: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum AutomationsCommand {
    List,
    Enable {
        watcher: String,
        #[arg(long)]
        mode: Option<WatcherModeArg>,
    },
    Disable {
        watcher: String,
    },
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WatcherModeArg {
    Observe,
    Propose,
    Contained,
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
    if interactive_terminal() {
        return tui::run(None);
    }

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    print!("{}", render_launch_summary(&paths, &config));
    Ok(())
}

fn run_freeform(request: String) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    print!("{}", render_freeform_plan(&request, &paths, &config));
    Ok(())
}

fn dispatch(cli: Cli) -> Result<()> {
    if cli.experimental_codex_tui {
        return match cli.command {
            None | Some(Command::Chat { .. }) => launch_experimental_codex_tui(),
            Some(_) => bail!(
                "`--experimental-codex-tui` is only supported for interactive chat launch"
            ),
        };
    }

    match cli.command {
        Some(Command::Doctor) => doctor(),
        Some(Command::Version) => {
            print!("rocm {}\n", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Chat { provider }) => {
            if interactive_terminal() {
                return tui::run(provider.map(provider_name).map(str::to_owned));
            }
            print!(
                "{}",
                render_chat_text(provider.map(provider_name).unwrap_or("local"))
            );
            Ok(())
        }
        Some(Command::Install { target }) => install(target),
        Some(Command::Update) => {
            let paths = AppPaths::discover()?;
            print!("{}", render_update_text(&paths)?);
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
            print!("{}", render_logs_text(&paths));
            Ok(())
        }
        Some(Command::Daemon) => {
            let paths = AppPaths::discover()?;
            let config = RocmCliConfig::load(&paths)?;
            print!("{}", render_daemon_text(&paths, &config));
            Ok(())
        }
        Some(Command::Uninstall {
            yes,
            dry_run,
            keep_binaries,
            keep_config,
            keep_data,
            keep_cache,
            force_dev_binaries,
        }) => uninstall(UninstallOptions {
            yes,
            dry_run,
            keep_binaries,
            keep_config,
            keep_data,
            keep_cache,
            force_dev_binaries,
        }),
        None => launch_default(),
    }
}

fn launch_experimental_codex_tui() -> Result<()> {
    if !interactive_terminal() {
        bail!("`rocm --experimental-codex-tui` requires an interactive terminal");
    }

    let workspace = vendored_codex_workspace()?;
    let manifest_path = workspace.join("Cargo.toml");
    let binary_path = vendored_codex_binary(&workspace);
    let mut command = if let Some(binary_path) = &binary_path {
        let mut process = ProcessCommand::new(binary_path);
        process.arg("chat");
        process
    } else {
        let mut process = ProcessCommand::new("cargo");
        process.args([
            "run",
            "--manifest-path",
            manifest_path
                .to_str()
                .context("vendored Codex manifest path was not valid UTF-8")?,
            "-p",
            "codex-cli",
            "--bin",
            "codex",
            "--",
            "chat",
        ]);
        process
    };

    println!("experimental Codex TUI launch");
    println!("  source: {}", workspace.display());
    if let Some(binary_path) = &binary_path {
        println!("  mode: prebuilt binary");
        println!("  binary: {}", binary_path.display());
    } else {
        println!("  mode: cargo run");
        println!("  manifest: {}", manifest_path.display());
        println!("  note: this will compile the vendored Codex workspace on first launch");
    }
    println!(
        "  provider policy: vendored Codex auth flow remains intact; ChatGPT sign-in stays available as the no-key default path"
    );

    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to launch vendored Codex TUI")?;

    if status.success() {
        Ok(())
    } else {
        bail!("vendored Codex TUI exited with status {status}");
    }
}

fn vendored_codex_workspace() -> Result<PathBuf> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/openai-codex/codex-rs");
    if workspace.join("Cargo.toml").is_file() {
        Ok(workspace)
    } else {
        bail!(
            "vendored Codex workspace not found at {}",
            workspace.display()
        );
    }
}

fn vendored_codex_binary(workspace: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        vec![
            workspace.join("target").join("release").join("codex.exe"),
            workspace.join("target").join("debug").join("codex.exe"),
        ]
    } else {
        vec![
            workspace.join("target").join("release").join("codex"),
            workspace.join("target").join("debug").join("codex"),
        ]
    };

    candidates.into_iter().find(|path| path.is_file())
}

fn doctor() -> Result<()> {
    print!("{}", render_doctor_text()?);
    Ok(())
}

fn install(target: InstallTarget) -> Result<()> {
    let paths = AppPaths::discover()?;
    match target {
        InstallTarget::Sdk {
            channel,
            format,
            prefix,
            dry_run,
        } => {
            let format_name = match format {
                InstallFormat::Pip => "pip",
                InstallFormat::Tarball => "tarball",
            };
            print!(
                "{}",
                therock::install_sdk(&paths, &channel, format_name, prefix, dry_run)?
            );
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
            print!("{}", render_engine_inventory_text());
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
        EnginesCommand::Shell {
            engine,
            runtime_id,
            env_id,
            shell,
        } => engine_shell(
            &engine,
            runtime_id.as_deref(),
            env_id.as_deref(),
            shell.as_deref(),
        ),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ManagedEngineEnvManifest {
    env_id: String,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ResolvedEngineEnv {
    env_id: String,
    runtime_id: String,
    python_executable: String,
    env_path: PathBuf,
    source: String,
}

fn engine_shell(
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    shell_override: Option<&str>,
) -> Result<()> {
    if !interactive_terminal() {
        bail!("`rocm engines shell` requires an interactive terminal");
    }

    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    let resolved = resolve_engine_env(&paths, &config, engine, runtime_id, env_id)?;
    let shell_program = shell_override
        .map(str::to_owned)
        .or_else(default_shell_program)
        .context("unable to determine an interactive shell; set --shell or SHELL")?;
    let venv_bin = managed_env_bin_dir(&resolved.env_path);
    let shell_hint = activation_hint(&resolved.env_path);

    println!("engine shell");
    println!("  engine: {engine}");
    println!("  source: {}", resolved.source);
    println!("  env_id: {}", resolved.env_id);
    println!("  runtime_id: {}", resolved.runtime_id);
    println!("  env_path: {}", resolved.env_path.display());
    println!("  python: {}", resolved.python_executable);
    println!("  shell: {shell_program}");
    println!("  activate_hint: {shell_hint}");
    println!("  exit_hint: use `exit` or Ctrl-D to leave the managed env shell");

    let path_with_env = prepend_path(&venv_bin, std::env::var_os("PATH"))
        .context("failed to compose PATH for managed engine env shell")?;
    let mut command = ProcessCommand::new(&shell_program);
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("VIRTUAL_ENV", &resolved.env_path)
        .env("PATH", path_with_env)
        .env("ROCM_CLI_ENGINE", engine)
        .env("ROCM_CLI_ENV_ID", &resolved.env_id)
        .env("ROCM_CLI_RUNTIME_ID", &resolved.runtime_id)
        .env("ROCM_CLI_PYTHON", &resolved.python_executable);

    if !cfg!(windows) {
        let prompt = format!("(rocm:{engine}) ");
        command.env("VIRTUAL_ENV_PROMPT", &prompt);
        command.env("PS1", format!("{prompt}${{PS1:-}}"));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to launch shell `{shell_program}`"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("managed engine shell exited with status {status}");
    }
}

fn resolve_engine_env(
    paths: &AppPaths,
    config: &RocmCliConfig,
    engine: &str,
    runtime_id: Option<&str>,
    env_id: Option<&str>,
) -> Result<ResolvedEngineEnv> {
    let selection = resolve_engine_selection(config, engine, runtime_id, env_id);
    if let Some(env_id) = selection.env_id.as_deref() {
        let manifest = load_engine_env_manifest(paths, engine, env_id)?;
        return Ok(ResolvedEngineEnv {
            env_id: manifest.env_id,
            runtime_id: manifest.runtime_id,
            python_executable: manifest.python_executable,
            env_path: manifest.env_path,
            source: selection
                .source
                .unwrap_or_else(|| "manifest_env_id".to_owned()),
        });
    }

    let runtime_id = selection
        .runtime_id
        .unwrap_or_else(|| DEFAULT_RUNTIME_ID.to_owned());
    let response = engine_request::<_, InstallResponse>(
        engine,
        EngineMethod::Install,
        &InstallRequest {
            runtime_id: runtime_id.clone(),
            python_version: None,
            reinstall: false,
        },
    )?;
    Ok(ResolvedEngineEnv {
        env_id: response.env_id,
        runtime_id,
        python_executable: response.python_executable,
        env_path: PathBuf::from(response.env_path),
        source: selection
            .source
            .unwrap_or_else(|| "auto_install".to_owned()),
    })
}

fn load_engine_env_manifest(
    paths: &AppPaths,
    engine: &str,
    env_id: &str,
) -> Result<ManagedEngineEnvManifest> {
    let path = paths
        .engine_manifests_dir(engine)
        .join(format!("{env_id}.json"));
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn managed_env_bin_dir(env_path: &Path) -> PathBuf {
    if cfg!(windows) {
        env_path.join("Scripts")
    } else {
        env_path.join("bin")
    }
}

fn prepend_path(prefix: &Path, current_path: Option<OsString>) -> Result<OsString> {
    let mut parts = vec![prefix.to_path_buf()];
    if let Some(current_path) = current_path.as_ref() {
        parts.extend(std::env::split_paths(current_path));
    }
    std::env::join_paths(parts).context("failed to join PATH entries")
}

fn default_shell_program() -> Option<String> {
    if cfg!(windows) {
        std::env::var("COMSPEC")
            .ok()
            .filter(|value| !value.trim().is_empty())
    } else {
        std::env::var("SHELL")
            .ok()
            .filter(|value| !value.trim().is_empty())
    }
}

fn activation_hint(env_path: &Path) -> String {
    if cfg!(windows) {
        format!(
            "{}",
            env_path.join("Scripts").join("activate.bat").display()
        )
    } else {
        format!("source {}", env_path.join("bin").join("activate").display())
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
        runtime_id.map(str::to_owned),
        env_id.map(str::to_owned),
        Some(device_policy_name(device_policy).to_owned()),
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
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(AutomationsCommand::List) {
        AutomationsCommand::List => {
            print!("{}", render_automations_text(&paths, &config)?);
        }
        AutomationsCommand::Enable { watcher, mode } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = true;
            if let Some(mode) = mode {
                entry.mode = Some(mode.into());
            }
            config.automations.daemon_enabled = true;
            config.save(&paths)?;
            println!("automation watcher enabled");
            println!("  watcher: {}", spec.id);
            println!("  mode: {}", config.effective_watcher_mode(spec).as_str());
            println!("  trigger: {}", spec.trigger);
            println!("  config: {}", paths.config_path().display());
            println!(
                "  next step: run `rocmd run --automations-enabled` to start the persistent watcher loop"
            );
        }
        AutomationsCommand::Disable { watcher } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = false;
            if !config
                .automations
                .watchers
                .values()
                .any(|watcher| watcher.enabled)
            {
                config.automations.daemon_enabled = false;
            }
            config.save(&paths)?;
            println!("automation watcher disabled");
            println!("  watcher: {}", spec.id);
            println!("  config: {}", paths.config_path().display());
        }
    }
    Ok(())
}

fn config(command: ConfigCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;

    match command {
        ConfigCommand::Show => {
            print!("{}", render_config_text(&paths, &config));
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

pub(crate) fn render_launch_summary(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    let mut output = String::new();
    let _ = writeln!(output, "rocm interactive shell");
    let _ = writeln!(output, "  terminal: non-interactive");
    let _ = writeln!(output, "  default engine: {selected_default_engine}");
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  config file: {}", paths.config_path().display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(output, "  cache dir: {}", paths.cache_dir.display());
    let _ = writeln!(
        output,
        "  note: launch from an interactive terminal to enter the TUI."
    );
    output
}

pub(crate) fn render_chat_text(provider: &str) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "chat shell");
    let _ = writeln!(output, "  provider: {provider}");
    let _ = writeln!(
        output,
        "  note: launch from an interactive terminal to enter the TUI."
    );
    output
}

pub(crate) fn render_doctor_text() -> Result<String> {
    Ok(DoctorSummary::gather()?.render_text())
}

pub(crate) fn render_engine_inventory_text() -> String {
    let default_engine = default_engine_for_platform();
    let mut output = String::new();
    let _ = writeln!(output, "engine inventory");
    for (name, note) in engine_inventory() {
        let marker = if *name == default_engine { "*" } else { " " };
        let _ = writeln!(output, "{marker} {name:10} {note}");
    }
    let _ = writeln!(
        output,
        "  protocol: {}",
        rocm_engine_protocol::ENGINE_PROTOCOL_VERSION
    );
    output
}

pub(crate) fn render_config_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "rocm config");
    let _ = writeln!(output, "  file: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  default_engine: {}",
        config
            .default_engine
            .as_deref()
            .unwrap_or("<platform default>")
    );
    if config.engines.is_empty() {
        let _ = writeln!(output, "  engines: none");
        return output;
    }
    for (engine, entry) in &config.engines {
        let _ = writeln!(output, "  engine: {engine}");
        let _ = writeln!(
            output,
            "    preferred_runtime_id: {}",
            entry.preferred_runtime_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    preferred_env_id: {}",
            entry.preferred_env_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_runtime_id: {}",
            entry
                .last_installed_runtime_id
                .as_deref()
                .unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_env_id: {}",
            entry.last_installed_env_id.as_deref().unwrap_or("<unset>")
        );
    }
    output
}

pub(crate) fn render_services_text(paths: &AppPaths) -> Result<String> {
    let records = load_managed_services(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "managed services");
    if records.is_empty() {
        let _ = writeln!(output, "  services: none");
        return Ok(output);
    }

    for record in records {
        let _ = writeln!(
            output,
            "  service {} engine={} status={} endpoint={}",
            record.service_id, record.engine, record.status, record.endpoint_url
        );
    }

    Ok(output)
}

pub(crate) fn render_logs_text(paths: &AppPaths) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "logs");
    let _ = writeln!(output, "  dir: {}", paths.data_dir.join("logs").display());
    output
}

pub(crate) fn render_update_text(paths: &AppPaths) -> Result<String> {
    therock::render_update_report(paths)
}

pub(crate) fn render_automations_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let recent_events = load_recent_automation_events(paths, 5).unwrap_or_default();
    let mut output = String::new();
    let _ = writeln!(output, "automation watchers");
    let _ = writeln!(output, "  config: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  daemon desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match runtime_state.as_ref() {
        Some(state) => {
            let _ = writeln!(
                output,
                "  daemon runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
        }
        None => {
            let _ = writeln!(output, "  daemon runtime: inactive");
        }
    }
    for watcher in builtin_watchers() {
        let runtime_snapshot = runtime_state.as_ref().and_then(|state| {
            state
                .active_watchers
                .iter()
                .find(|item| item.id == watcher.id)
        });
        let _ = writeln!(
            output,
            "  watcher {} enabled={} mode={} trigger={}",
            watcher.id,
            config.watcher_enabled(watcher),
            config.effective_watcher_mode(watcher).as_str(),
            watcher.trigger
        );
        let _ = writeln!(output, "    summary: {}", watcher.summary);
        let _ = writeln!(output, "    actions: {}", watcher.actions.join(", "));
        if let Some(snapshot) = runtime_snapshot {
            let _ = writeln!(
                output,
                "    runtime: last_event={} last_event_unix_ms={}",
                snapshot.last_event.as_deref().unwrap_or("<none>"),
                snapshot
                    .last_event_unix_ms
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "<never>".to_owned())
            );
        }
    }
    if !recent_events.is_empty() {
        let _ = writeln!(output, "  recent events:");
        for event in recent_events {
            let _ = writeln!(
                output,
                "    {} {} {} {}",
                event.at_unix_ms, event.watcher_id, event.action, event.message
            );
        }
    }
    Ok(output)
}

pub(crate) fn render_daemon_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let runtime_state = AutomationRuntimeState::load(paths).unwrap_or(None);
    let mut output = String::new();
    let _ = writeln!(output, "rocmd lifecycle");
    let _ = writeln!(output, "  default: on-demand");
    let _ = writeln!(
        output,
        "  persistent: only when automations or managed background services are enabled"
    );
    let _ = writeln!(
        output,
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    let _ = writeln!(
        output,
        "  runtime state: {}",
        paths.automation_state_path().display()
    );
    match runtime_state {
        Some(state) => {
            let _ = writeln!(
                output,
                "  daemon status: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
        }
        None => {
            let _ = writeln!(output, "  daemon status: inactive");
        }
    }
    output
}

pub(crate) fn render_sidebar_text(
    paths: &AppPaths,
    config: &RocmCliConfig,
    provider: &str,
) -> String {
    let records = load_managed_services(paths).unwrap_or_default();
    let default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    let mut output = String::new();
    let _ = writeln!(output, "ROCm AI Command Center CLI");
    let _ = writeln!(output, "os: {}", std::env::consts::OS);
    let _ = writeln!(output, "arch: {}", std::env::consts::ARCH);
    let _ = writeln!(output, "provider: {provider}");
    let _ = writeln!(output, "default engine: {default_engine}");
    let _ = writeln!(output, "interactive: {}", interactive_terminal());
    let _ = writeln!(output, "services: {}", records.len());
    let enabled_watchers = builtin_watchers()
        .iter()
        .filter(|watcher| config.watcher_enabled(watcher))
        .count();
    let _ = writeln!(output, "watchers: {}", enabled_watchers);
    let _ = writeln!(
        output,
        "daemon: {}",
        if config.automation_daemon_enabled() {
            "desired"
        } else {
            "off"
        }
    );
    let _ = writeln!(output, "config: {}", paths.config_path().display());
    let _ = writeln!(output, "data: {}", paths.data_dir.display());
    let _ = writeln!(output, "cache: {}", paths.cache_dir.display());
    if !config.engines.is_empty() {
        let _ = writeln!(output);
        let _ = writeln!(output, "engine prefs:");
        for (engine, entry) in &config.engines {
            let selected = entry
                .preferred_env_id
                .as_deref()
                .or(entry.preferred_runtime_id.as_deref())
                .or(entry.last_installed_env_id.as_deref())
                .or(entry.last_installed_runtime_id.as_deref())
                .unwrap_or("<unset>");
            let _ = writeln!(output, "  {engine}: {selected}");
        }
    }
    output
}

pub(crate) fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
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

pub(crate) fn tui_help_text() -> String {
    let mut output = String::new();
    let _ = writeln!(output, "slash commands");
    let _ = writeln!(output, "  /help          show this help");
    let _ = writeln!(output, "  /doctor        inspect host and default paths");
    let _ = writeln!(output, "  /engines       show bundled engine inventory");
    let _ = writeln!(output, "  /config        show persisted config");
    let _ = writeln!(
        output,
        "  /automations   show watcher config and daemon state"
    );
    let _ = writeln!(output, "  /services      show managed service manifests");
    let _ = writeln!(output, "  /logs          show log directories");
    let _ = writeln!(
        output,
        "  /gpu           show the latest amd-smi telemetry snapshot"
    );
    let _ = writeln!(output, "  /update        show update policy");
    let _ = writeln!(output, "  /daemon        show rocmd lifecycle");
    let _ = writeln!(output, "  /provider X    switch provider for this session");
    let _ = writeln!(
        output,
        "  /uninstall     show the default uninstall dry-run"
    );
    let _ = writeln!(output, "  /clear         clear the transcript");
    let _ = writeln!(output, "  /quit          exit the TUI");
    let _ = writeln!(output);
    let _ = writeln!(output, "keyboard");
    let _ = writeln!(output, "  Up/Down        recall input history");
    let _ = writeln!(output, "  PgUp/PgDn      scroll the transcript");
    let _ = writeln!(output, "  Home/End       jump to top or bottom");
    let _ = writeln!(output);
    let _ = writeln!(output, "natural language");
    let _ = writeln!(output, "  serve Qwen3.5 with vllm");
    let _ = writeln!(output, "  install the latest therock release");
    let _ = writeln!(output, "  uninstall rocm-cli");
    let _ = writeln!(output);
    let _ = writeln!(output, "plain commands");
    let _ = writeln!(output, "  config set-default-engine pytorch");
    let _ = writeln!(
        output,
        "  config set-engine pytorch --runtime-id therock-release"
    );
    let _ = writeln!(output, "  engines install pytorch --reinstall");
    let _ = writeln!(output, "  automations enable server-recover");
    output
}

pub(crate) fn render_freeform_plan(
    request: &str,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> String {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or(default_engine_for_platform());
    let engine = infer_engine_from_request(&lower).unwrap_or(default_engine);
    let mut output = String::new();
    let _ = writeln!(output, "request plan");
    let _ = writeln!(output, "  request: {trimmed}");

    if lower.contains("serve") {
        let model = infer_model_from_request(trimmed).unwrap_or("<model>");
        let _ = writeln!(output, "  intent: serve");
        let _ = writeln!(output, "  selected engine: {engine}");
        let _ = writeln!(output, "  selected model: {model}");
        let _ = writeln!(output, "  plan:");
        let _ = writeln!(output, "    1. resolve the model recipe and engine runtime");
        let _ = writeln!(
            output,
            "    2. verify the preferred env or install one if needed"
        );
        let _ = writeln!(output, "    3. start a local OpenAI-compatible endpoint");
        let _ = writeln!(
            output,
            "  suggested command: rocm serve {model} --engine {engine}"
        );
        return output;
    }

    if lower.contains("driver") {
        let _ = writeln!(output, "  intent: install driver");
        let _ = writeln!(
            output,
            "  suggested command: rocm install driver{}",
            if lower.contains("dkms") {
                " --dkms"
            } else {
                ""
            }
        );
        let _ = writeln!(
            output,
            "  note: driver changes are always explicit and never silent."
        );
        return output;
    }

    if lower.contains("install") || lower.contains("therock") || lower.contains("sdk") {
        let channel = if lower.contains("nightly") {
            "nightly"
        } else {
            "release"
        };
        let _ = writeln!(output, "  intent: install sdk");
        let _ = writeln!(output, "  selected channel: {channel}");
        let _ = writeln!(
            output,
            "  suggested command: rocm install sdk --channel {channel}"
        );
        return output;
    }

    if lower.contains("update") {
        let _ = writeln!(output, "  intent: update check");
        let _ = writeln!(output, "  suggested command: rocm update");
        let _ = writeln!(
            output,
            "  note: update checks should compare against the selected release channel."
        );
        return output;
    }

    if lower.contains("uninstall") || lower.contains("remove rocm") {
        let _ = writeln!(output, "  intent: uninstall");
        let _ = writeln!(output, "  suggested command: rocm uninstall --dry-run");
        let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
        return output;
    }

    let _ = writeln!(output, "  intent: ask or inspect");
    let _ = writeln!(output, "  selected engine: {engine}");
    let _ = writeln!(
        output,
        "  note: enter the TUI with `rocm` to inspect config, services, and plans interactively."
    );
    output
}

#[derive(Debug, Clone, Default)]
struct UninstallOptions {
    yes: bool,
    dry_run: bool,
    keep_binaries: bool,
    keep_config: bool,
    keep_data: bool,
    keep_cache: bool,
    force_dev_binaries: bool,
}

#[derive(Debug, Clone)]
struct UninstallPlanEntry {
    kind: &'static str,
    path: PathBuf,
}

#[derive(Debug, Clone, Default)]
struct UninstallPlan {
    actions: Vec<UninstallPlanEntry>,
    skipped: Vec<String>,
    warnings: Vec<String>,
}

fn uninstall(options: UninstallOptions) -> Result<()> {
    let paths = AppPaths::discover()?;
    let plan = build_uninstall_plan(&paths, &options)?;
    print!("{}", render_uninstall_plan(&plan, &options));

    if plan.actions.is_empty() || options.dry_run {
        return Ok(());
    }

    if !options.yes {
        if !interactive_terminal() {
            bail!("uninstall requires --yes outside an interactive terminal");
        }
        if !confirm_uninstall()? {
            println!("uninstall cancelled");
            return Ok(());
        }
    }

    for entry in &plan.actions {
        remove_path(&entry.path)
            .with_context(|| format!("failed to remove {}", entry.path.display()))?;
        println!("removed {} {}", entry.kind, entry.path.display());
    }
    println!("uninstall complete");
    Ok(())
}

fn build_uninstall_plan(paths: &AppPaths, options: &UninstallOptions) -> Result<UninstallPlan> {
    let mut plan = UninstallPlan::default();

    if !options.keep_binaries {
        let current_exe =
            std::env::current_exe().context("failed to discover current rocm executable")?;
        if is_dev_binary_layout(&current_exe) && !options.force_dev_binaries {
            plan.skipped.push(format!(
                "binary removal skipped because {} looks like a cargo target build; pass --force-dev-binaries to remove sibling debug/release binaries",
                current_exe.display()
            ));
        } else {
            for path in collect_installed_binary_candidates(&current_exe)? {
                if cfg!(windows) && path == current_exe {
                    plan.skipped.push(format!(
                        "skipping running executable on Windows: {}",
                        path.display()
                    ));
                    continue;
                }
                plan.actions.push(UninstallPlanEntry {
                    kind: "binary",
                    path,
                });
            }
        }
    } else {
        plan.skipped
            .push("binary removal disabled by --keep-binaries".to_owned());
    }

    for (keep, kind, path) in [
        (options.keep_config, "config", paths.config_dir.clone()),
        (options.keep_data, "data", paths.data_dir.clone()),
        (options.keep_cache, "cache", paths.cache_dir.clone()),
    ] {
        if keep {
            plan.skipped
                .push(format!("{kind} removal disabled by command line flag"));
            continue;
        }
        if path.exists() {
            plan.actions.push(UninstallPlanEntry { kind, path });
        } else {
            plan.skipped
                .push(format!("{kind} path not present: {}", path.display()));
        }
    }

    let managed_services = load_managed_services(paths).unwrap_or_default();
    if !managed_services.is_empty() {
        plan.warnings.push(format!(
            "{} managed service record(s) exist under {}; background processes are not stopped automatically in this pass",
            managed_services.len(),
            paths.services_dir().display()
        ));
    }

    plan.actions
        .sort_by(|left, right| left.path.cmp(&right.path));
    plan.actions.dedup_by(|left, right| left.path == right.path);
    Ok(plan)
}

pub(crate) fn render_uninstall_dry_run(paths: &AppPaths) -> Result<String> {
    let options = UninstallOptions {
        dry_run: true,
        ..UninstallOptions::default()
    };
    let plan = build_uninstall_plan(paths, &options)?;
    Ok(render_uninstall_plan(&plan, &options))
}

fn render_uninstall_plan(plan: &UninstallPlan, options: &UninstallOptions) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "uninstall plan");
    let _ = writeln!(
        output,
        "  mode: {}",
        if options.dry_run { "dry-run" } else { "apply" }
    );
    if plan.actions.is_empty() {
        let _ = writeln!(output, "  actions: none");
    } else {
        let _ = writeln!(output, "  actions:");
        for entry in &plan.actions {
            let _ = writeln!(
                output,
                "    remove {:7} {}",
                entry.kind,
                entry.path.display()
            );
        }
    }
    for warning in &plan.warnings {
        let _ = writeln!(output, "  warning: {warning}");
    }
    for skipped in &plan.skipped {
        let _ = writeln!(output, "  skipped: {skipped}");
    }
    if options.dry_run {
        let _ = writeln!(
            output,
            "  next step: rerun with `rocm uninstall --yes` to apply"
        );
    }
    output
}

fn confirm_uninstall() -> Result<bool> {
    print!("Proceed with uninstall? [y/N]: ");
    io::stdout()
        .flush()
        .context("failed to flush uninstall prompt")?;
    let mut response = String::new();
    io::stdin()
        .read_line(&mut response)
        .context("failed to read uninstall confirmation")?;
    let normalized = response.trim().to_ascii_lowercase();
    Ok(matches!(normalized.as_str(), "y" | "yes"))
}

fn collect_installed_binary_candidates(current_exe: &Path) -> Result<Vec<PathBuf>> {
    let binary_dir = current_exe
        .parent()
        .context("current executable has no parent directory")?;
    let mut binaries = Vec::new();
    for entry in fs::read_dir(binary_dir)
        .with_context(|| format!("failed to read {}", binary_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !path.is_file() {
            continue;
        }
        if is_rocm_install_entry_name(name) {
            binaries.push(path);
        }
    }
    binaries.sort();
    Ok(binaries)
}

fn is_rocm_install_entry_name(name: &str) -> bool {
    if name == ".rocm-cli-manifest" {
        return true;
    }
    let normalized = name.strip_suffix(".exe").unwrap_or(name);
    normalized == "rocm" || normalized == "rocmd" || normalized.starts_with("rocm-engine-")
}

fn is_dev_binary_layout(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(parent_name) = parent.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    if parent_name != "debug" && parent_name != "release" {
        return false;
    }
    parent
        .parent()
        .and_then(|value| value.file_name())
        .and_then(|value| value.to_str())
        == Some("target")
}

fn remove_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else if metadata.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else {
        bail!("unsupported filesystem entry for {}", path.display());
    }
    Ok(())
}

fn engine_inventory() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "pytorch",
            "default local serving engine when vllm is not installed",
        ),
        ("llama.cpp", "CPU and quantized fallback engine"),
        (
            "vllm",
            "preferred Linux ROCm GPU serving engine when installed",
        ),
        ("sglang", "deferred advanced serving engine"),
        ("atom", "deferred AMD-optimized serving engine"),
    ]
}

fn infer_engine_from_request<'a>(lower: &'a str) -> Option<&'a str> {
    for engine in ["pytorch", "llama.cpp", "vllm", "sglang", "atom"] {
        if lower.contains(engine) {
            return Some(engine);
        }
    }
    if lower.contains("llama cpp") {
        return Some("llama.cpp");
    }
    None
}

fn infer_model_from_request(request: &str) -> Option<&str> {
    let trimmed = request.trim();
    let lower = trimmed.to_ascii_lowercase();
    let serve_index = lower.find("serve")?;
    let after = trimmed.get(serve_index + "serve".len()..)?.trim();
    if after.is_empty() {
        return None;
    }
    let end = after
        .find(" with ")
        .or_else(|| after.find(" using "))
        .unwrap_or(after.len());
    let model = after[..end].trim();
    (!model.is_empty()).then_some(model)
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Local => "local",
        Provider::Anthropic => "anthropic",
        Provider::Openai => "openai",
    }
}

impl From<WatcherModeArg> for WatcherMode {
    fn from(value: WatcherModeArg) -> Self {
        match value {
            WatcherModeArg::Observe => WatcherMode::Observe,
            WatcherModeArg::Propose => WatcherMode::Propose,
            WatcherModeArg::Contained => WatcherMode::Contained,
        }
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
        "uninstall",
        "help",
        "--help",
        "-h",
        "version",
        "--version",
        "-V",
    ];

    !args.is_empty() && !args[0].starts_with('-') && !STRUCTURED.contains(&args[0].as_str())
}
