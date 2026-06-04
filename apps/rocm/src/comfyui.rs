use crate::{format_structured_tool_call, runtime_usability_status, therock};
use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use rocm_core::{AppPaths, RocmCliConfig, format_http_base_url, unix_time_millis};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Read, Write as IoWrite};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime};

const APP_ID: &str = "comfyui";
const APP_NAME: &str = "ComfyUI";
const COMFYUI_SOURCE_ARCHIVE_URL: &str =
    "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.tar.gz";
const COMFYUI_SOURCE_ARCHIVE_NAME: &str = "ComfyUI-master.tar.gz";
const COMFYUI_DEFAULT_HOST: &str = "127.0.0.1";
const COMFYUI_DEFAULT_PORT: u16 = 8188;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ComfyUiInstallOptions {
    pub runtime_id: Option<String>,
    pub reinstall: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ComfyUiStartOptions {
    pub host: String,
    pub port: u16,
    pub no_open_browser: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComfyUiManifest {
    app_id: String,
    runtime_key: String,
    runtime_id: String,
    runtime_version: String,
    runtime_root: PathBuf,
    python_executable: PathBuf,
    source_url: String,
    source_path: PathBuf,
    requirements_path: PathBuf,
    pip_cache_dir: PathBuf,
    log_path: PathBuf,
    torch_version: Option<String>,
    torch_cuda_available: bool,
    installed_at_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComfyUiState {
    app_id: String,
    url: String,
    host: String,
    port: u16,
    pid: u32,
    source_path: PathBuf,
    python_executable: PathBuf,
    log_path: PathBuf,
    started_at_unix_ms: u128,
}

#[derive(Debug, Clone, Deserialize)]
struct ComfyUiProbe {
    torch_version: Option<String>,
    torch_cuda_available: bool,
    device_count: i64,
    devices: Vec<String>,
}

#[derive(Debug, Clone)]
struct SelectedRuntime {
    manifest: therock::InstalledRuntimeManifest,
    python: PathBuf,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ComfyUiRunState {
    Running,
    Starting,
    Stopped,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ComfyUiRunReport {
    state: ComfyUiRunState,
    process_running: bool,
    endpoint_reachable: bool,
}

pub(crate) fn default_host() -> &'static str {
    COMFYUI_DEFAULT_HOST
}

pub(crate) fn default_port() -> u16 {
    COMFYUI_DEFAULT_PORT
}

pub(crate) fn render_status(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output)?;

    match load_manifest(paths)? {
        Some(manifest) => {
            writeln!(output, "  installed: yes")?;
            writeln!(
                output,
                "  ROCm install: {} ({})",
                manifest.runtime_key, manifest.runtime_id
            )?;
            writeln!(output, "  folder: {}", manifest.source_path.display())?;
            writeln!(output, "  python: {}", manifest.python_executable.display())?;
            writeln!(
                output,
                "  torch: {}",
                manifest.torch_version.as_deref().unwrap_or("unknown")
            )?;
            writeln!(
                output,
                "  AMD GPU check: {}",
                if manifest.torch_cuda_available {
                    "ready"
                } else {
                    "failed"
                }
            )?;
            writeln!(
                output,
                "  last install log: {}",
                manifest.log_path.display()
            )?;
        }
        None => {
            writeln!(output, "  installed: no")?;
            writeln!(output, "  next step: rocm comfyui install")?;
        }
    }

    match load_state(paths)? {
        Some(state) => {
            let run_report = evaluate_running_state(&state);
            writeln!(output)?;
            writeln!(output, "Running")?;
            writeln!(
                output,
                "  status: {}",
                comfyui_run_state_cli_label(run_report.state)
            )?;
            writeln!(output, "  url: {}", state.url)?;
            writeln!(output, "  pid: {}", state.pid)?;
            writeln!(output, "  log: {}", state.log_path.display())?;
            match run_report.state {
                ComfyUiRunState::Running => {}
                ComfyUiRunState::Starting => {
                    writeln!(
                        output,
                        "  note: process exists, but the local URL is not ready yet"
                    )?;
                }
                ComfyUiRunState::Stopped => {
                    writeln!(output, "  next step: rocm comfyui start")?;
                }
            }
        }
        None => {
            writeln!(output)?;
            writeln!(output, "Running")?;
            writeln!(output, "  status: not started by rocm-cli")?;
        }
    }

    let runtimes = therock::load_runtime_manifests(paths)?;
    if runtimes.is_empty() {
        writeln!(output)?;
        writeln!(output, "ROCm")?;
        writeln!(output, "  Install ROCm first from Set Up ROCm.")?;
    } else if let Some(active) = config.active_runtime_key.as_deref() {
        writeln!(output)?;
        writeln!(output, "Default ROCm install")?;
        writeln!(output, "  {active}")?;
    }

    Ok(output)
}

pub(crate) fn render_tui_status(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output)?;
    match load_manifest(paths)? {
        Some(manifest) => {
            writeln!(output, "Installed")?;
            writeln!(output, "  status: ready")?;
            writeln!(
                output,
                "  ROCm install: {}",
                therock::runtime_version_display(&manifest.runtime_version)
            )?;
            writeln!(
                output,
                "  AMD GPU check: {}",
                if manifest.torch_cuda_available {
                    "ready"
                } else {
                    "needs attention"
                }
            )?;
        }
        None => {
            writeln!(output, "Not installed yet")?;
            writeln!(output, "  Choose Install ComfyUI below.")?;
        }
    }

    writeln!(output)?;
    match load_state(paths)? {
        Some(state) => {
            let run_report = evaluate_running_state(&state);
            writeln!(output, "Running")?;
            match run_report.state {
                ComfyUiRunState::Running => {
                    writeln!(output, "  status: running")?;
                    writeln!(output, "  URL: {}", state.url)?;
                }
                ComfyUiRunState::Starting => {
                    writeln!(output, "  status: starting")?;
                    writeln!(output, "  URL: {}", state.url)?;
                    writeln!(output, "  Waiting for the browser page to answer.")?;
                }
                ComfyUiRunState::Stopped => {
                    writeln!(output, "  status: stopped")?;
                    writeln!(output, "  Choose Start ComfyUI below to run it again.")?;
                }
            }
        }
        None => {
            writeln!(output, "Running")?;
            writeln!(output, "  status: not started")?;
        }
    }

    if config.active_runtime_key.is_none() && load_manifest(paths)?.is_none() {
        writeln!(output)?;
        writeln!(output, "ROCm")?;
        writeln!(output, "  Install ROCm first from Set Up ROCm.")?;
    }
    writeln!(output)?;
    writeln!(output, "Next actions")?;
    writeln!(
        output,
        "  Use the rows on the left to install, start, or open logs."
    )?;
    Ok(output)
}

pub(crate) fn render_logs(paths: &AppPaths, line_limit: usize) -> Result<String> {
    render_logs_with_options(paths, line_limit, true)
}

pub(crate) fn render_tui_logs(
    paths: &AppPaths,
    line_limit: usize,
    show_file_locations: bool,
) -> Result<String> {
    render_logs_with_options(paths, line_limit, show_file_locations)
}

fn render_logs_with_options(
    paths: &AppPaths,
    line_limit: usize,
    show_file_locations: bool,
) -> Result<String> {
    let limit = line_limit.clamp(1, 400);
    let logs = recent_log_paths(paths)?;
    let mut output = String::new();
    writeln!(output, "{APP_NAME} logs")?;
    writeln!(output)?;
    if logs.is_empty() {
        writeln!(output, "No ComfyUI logs yet.")?;
        writeln!(output)?;
        writeln!(output, "Start with: rocm comfyui install")?;
        return Ok(output);
    }

    for (index, path) in logs.iter().take(4).enumerate() {
        if index > 0 {
            writeln!(output)?;
        }
        writeln!(output, "{}", comfyui_log_title(path))?;
        if show_file_locations {
            writeln!(output, "  saved file: {}", path.display())?;
        }
        let lines = read_tail_lines(path, limit)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if lines.is_empty() {
            writeln!(output, "  The saved log is empty.")?;
        } else {
            writeln!(output, "  latest output:")?;
            for line in lines {
                writeln!(output, "    {line}")?;
            }
        }
    }
    Ok(output)
}

pub(crate) fn install(
    paths: &AppPaths,
    config: &RocmCliConfig,
    options: ComfyUiInstallOptions,
) -> Result<String> {
    paths.ensure()?;
    let runtime = select_runtime(paths, config, options.runtime_id.as_deref())?;
    let app_root = app_root(paths);
    let source_path = source_path(paths);
    let pip_cache = app_root.join("pip-cache");
    let log_path = install_log_path(paths);
    let requirements_path = source_path.join("requirements.txt");

    let mut output = String::new();
    writeln!(output, "{APP_NAME} install")?;
    writeln!(output, "  ROCm install: {}", runtime.manifest.runtime_key)?;
    writeln!(
        output,
        "  ROCm version: {}",
        therock::runtime_version_display(&runtime.manifest.version)
    )?;
    writeln!(output, "  folder: {}", source_path.display())?;
    writeln!(output, "  python: {}", runtime.python.display())?;
    writeln!(output, "  pip cache: {}", pip_cache.display())?;
    writeln!(
        output,
        "  package policy: keep the TheRock ROCm torch packages already installed in this Python environment"
    )?;

    if options.dry_run {
        writeln!(output, "  mode: dry-run")?;
        writeln!(
            output,
            "  install command: {}",
            format_structured_tool_call(
                "rocm",
                &[
                    "comfyui".to_owned(),
                    "install".to_owned(),
                    "--runtime-id".to_owned(),
                    runtime.manifest.runtime_key.clone(),
                ],
            )
        )?;
        return Ok(output);
    }

    fs::create_dir_all(
        log_path
            .parent()
            .context("ComfyUI install log path has no parent directory")?,
    )?;
    let mut log = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    writeln!(log, "{APP_NAME} install")?;
    writeln!(log, "runtime_key={}", runtime.manifest.runtime_key)?;
    writeln!(log, "python={}", runtime.python.display())?;

    if options.reinstall && source_path.exists() {
        writeln!(log, "Removing existing ComfyUI folder.")?;
        fs::remove_dir_all(&source_path)
            .with_context(|| format!("failed to remove {}", source_path.display()))?;
    }
    if !source_path.exists() {
        println!("Downloading ComfyUI source...");
        let _ = io::stdout().flush();
        download_and_extract_source(paths, &source_path, &mut log)?;
    } else {
        println!("Using existing ComfyUI source folder...");
        let _ = io::stdout().flush();
        writeln!(
            log,
            "Using existing ComfyUI folder at {}.",
            source_path.display()
        )?;
    }

    let packages = filtered_requirement_specs(&requirements_path)?;
    writeln!(
        log,
        "Installing {} ComfyUI dependency specs.",
        packages.len()
    )?;
    fs::create_dir_all(&pip_cache)
        .with_context(|| format!("failed to create {}", pip_cache.display()))?;
    if !packages.is_empty() {
        println!("Installing ComfyUI dependencies...");
        let _ = io::stdout().flush();
        run_logged_command(
            &runtime.python,
            pip_install_args(&pip_cache, &packages),
            &mut log,
            "install ComfyUI dependencies",
        )?;
    }

    println!("Checking AMD GPU access for ComfyUI...");
    let _ = io::stdout().flush();
    let probe = probe_comfyui(&runtime.python, &source_path)?;
    if !probe.torch_cuda_available {
        bail!("ComfyUI install finished, but the AMD GPU check failed. No CPU mode was used.");
    }
    let manifest = ComfyUiManifest {
        app_id: APP_ID.to_owned(),
        runtime_key: runtime.manifest.runtime_key.clone(),
        runtime_id: runtime.manifest.runtime_id.clone(),
        runtime_version: runtime.manifest.version.clone(),
        runtime_root: runtime.manifest.install_root.clone(),
        python_executable: runtime.python.clone(),
        source_url: COMFYUI_SOURCE_ARCHIVE_URL.to_owned(),
        source_path: source_path.clone(),
        requirements_path: requirements_path.clone(),
        pip_cache_dir: pip_cache.clone(),
        log_path: log_path.clone(),
        torch_version: probe.torch_version.clone(),
        torch_cuda_available: probe.torch_cuda_available,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_manifest(paths, &manifest)?;

    writeln!(output, "  installed: yes")?;
    writeln!(
        output,
        "  AMD GPU check: ready ({} device{})",
        probe.device_count,
        if probe.device_count == 1 { "" } else { "s" }
    )?;
    if !probe.devices.is_empty() {
        writeln!(output, "  GPU: {}", probe.devices.join(", "))?;
    }
    writeln!(output, "  log: {}", log_path.display())?;
    writeln!(output, "  next step: rocm comfyui start")?;
    Ok(output)
}

pub(crate) fn start(paths: &AppPaths, options: ComfyUiStartOptions) -> Result<String> {
    let manifest = load_manifest(paths)?.context("ComfyUI is not installed yet")?;
    if !manifest.source_path.join("main.py").is_file() {
        bail!(
            "ComfyUI main.py is missing from {}; reinstall ComfyUI",
            manifest.source_path.display()
        );
    }
    let probe = probe_comfyui(&manifest.python_executable, &manifest.source_path)?;
    if !probe.torch_cuda_available {
        bail!("ComfyUI cannot start because the AMD GPU check failed. No CPU mode was used.");
    }
    let url = format_http_base_url(&options.host, options.port);
    let log_path = start_log_path(paths);
    fs::create_dir_all(
        log_path
            .parent()
            .context("ComfyUI start log path has no parent directory")?,
    )?;
    let log = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let stdout = log
        .try_clone()
        .context("failed to clone ComfyUI log file")?;
    let stderr = log
        .try_clone()
        .context("failed to clone ComfyUI log file")?;
    let mut command = Command::new(&manifest.python_executable);
    command
        .current_dir(&manifest.source_path)
        .arg("main.py")
        .arg("--listen")
        .arg(&options.host)
        .arg("--port")
        .arg(options.port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    let child = command.spawn().with_context(|| {
        format!(
            "failed to start ComfyUI with {}",
            manifest.python_executable.display()
        )
    })?;
    let pid = child.id();
    save_state(
        paths,
        &ComfyUiState {
            app_id: APP_ID.to_owned(),
            url: url.clone(),
            host: options.host.clone(),
            port: options.port,
            pid,
            source_path: manifest.source_path.clone(),
            python_executable: manifest.python_executable.clone(),
            log_path: log_path.clone(),
            started_at_unix_ms: unix_time_millis(),
        },
    )?;
    let browser_status = if options.no_open_browser {
        "not opened (--no-open-browser)".to_owned()
    } else {
        match open_browser(&url) {
            Ok(()) => "opened".to_owned(),
            Err(error) => format!("not opened ({error})"),
        }
    };

    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output, "  status: starting")?;
    writeln!(
        output,
        "  AMD GPU check: ready ({} device{})",
        probe.device_count,
        if probe.device_count == 1 { "" } else { "s" }
    )?;
    if !probe.devices.is_empty() {
        writeln!(output, "  GPU: {}", probe.devices.join(", "))?;
    }
    writeln!(output, "  url: {url}")?;
    writeln!(output, "  browser: {browser_status}")?;
    writeln!(output, "  pid: {pid}")?;
    writeln!(output, "  log: {}", log_path.display())?;
    writeln!(
        output,
        "  note: ComfyUI keeps running in the background; if the browser did not open, use the URL above"
    )?;
    Ok(output)
}

fn app_root(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("apps").join(APP_ID)
}

fn source_path(paths: &AppPaths) -> PathBuf {
    app_root(paths).join("source")
}

fn manifest_path(paths: &AppPaths) -> PathBuf {
    app_root(paths).join("manifests").join("current.json")
}

fn state_path(paths: &AppPaths) -> PathBuf {
    app_root(paths).join("state").join("running.json")
}

fn install_log_path(paths: &AppPaths) -> PathBuf {
    app_root(paths)
        .join("logs")
        .join(format!("install-{}.log", unix_time_millis()))
}

fn start_log_path(paths: &AppPaths) -> PathBuf {
    app_root(paths)
        .join("logs")
        .join(format!("start-{}.log", unix_time_millis()))
}

fn recent_log_paths(paths: &AppPaths) -> Result<Vec<PathBuf>> {
    let mut paths_seen = Vec::<PathBuf>::new();
    if let Some(manifest) = load_manifest(paths)? {
        push_log_path(&mut paths_seen, manifest.log_path);
    }
    if let Some(state) = load_state(paths)? {
        push_log_path(&mut paths_seen, state.log_path);
    }
    let logs_root = app_root(paths).join("logs");
    if logs_root.is_dir() {
        for entry in fs::read_dir(&logs_root)
            .with_context(|| format!("failed to read {}", logs_root.display()))?
        {
            let path = entry?.path();
            push_log_path(&mut paths_seen, path);
        }
    }
    paths_seen.retain(|path| path.is_file());
    paths_seen.sort_by(|left, right| {
        log_modified_time(right)
            .cmp(&log_modified_time(left))
            .then_with(|| right.cmp(left))
    });
    Ok(paths_seen)
}

fn push_log_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|seen| seen == &path) {
        paths.push(path);
    }
}

fn log_modified_time(path: &Path) -> SystemTime {
    path.metadata()
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn comfyui_log_title(path: &Path) -> &'static str {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return "Saved log";
    };
    if name.starts_with("install-") {
        "Install log"
    } else if name.starts_with("start-") {
        "Run log"
    } else {
        "Saved log"
    }
}

fn read_tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .rev()
        .take(limit)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines.reverse();
    Ok(lines)
}

fn evaluate_running_state(state: &ComfyUiState) -> ComfyUiRunReport {
    let process_running = process_is_running(state.pid);
    let endpoint_reachable = process_running && endpoint_is_reachable(&state.host, state.port);
    let state = if endpoint_reachable {
        ComfyUiRunState::Running
    } else if process_running {
        ComfyUiRunState::Starting
    } else {
        ComfyUiRunState::Stopped
    };
    ComfyUiRunReport {
        state,
        process_running,
        endpoint_reachable,
    }
}

fn comfyui_run_state_cli_label(state: ComfyUiRunState) -> &'static str {
    match state {
        ComfyUiRunState::Running => "running",
        ComfyUiRunState::Starting => "starting",
        ComfyUiRunState::Stopped => "stopped",
    }
}

fn endpoint_is_reachable(host: &str, port: u16) -> bool {
    let Ok(addresses) = (host, port).to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok())
}

#[cfg(windows)]
fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let filter = format!("PID eq {pid}");
    let Ok(output) = Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .stdin(Stdio::null())
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.contains(&format!("\"{pid}\""))
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(not(any(windows, unix)))]
fn process_is_running(pid: u32) -> bool {
    pid != 0
}

fn load_manifest(paths: &AppPaths) -> Result<Option<ComfyUiManifest>> {
    let path = manifest_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn save_manifest(paths: &AppPaths, manifest: &ComfyUiManifest) -> Result<()> {
    let path = manifest_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("ComfyUI manifest path has no parent directory")?,
    )?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(manifest).context("failed to serialize ComfyUI manifest")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn load_state(paths: &AppPaths) -> Result<Option<ComfyUiState>> {
    let path = state_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn save_state(paths: &AppPaths, state: &ComfyUiState) -> Result<()> {
    let path = state_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("ComfyUI state path has no parent directory")?,
    )?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(state).context("failed to serialize ComfyUI state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn select_runtime(
    paths: &AppPaths,
    config: &RocmCliConfig,
    selector: Option<&str>,
) -> Result<SelectedRuntime> {
    let manifests = therock::load_runtime_manifests(paths)?;
    if manifests.is_empty() {
        bail!("Install ROCm first from Set Up ROCm, then install ComfyUI.");
    }
    let manifest = match selector.map(str::trim).filter(|value| !value.is_empty()) {
        Some(selector) => select_runtime_by_selector(&manifests, selector)?.clone(),
        None => select_default_runtime(config, &manifests)?.clone(),
    };
    if runtime_usability_status(&manifest) != "ready" {
        bail!(
            "The selected ROCm install is not ready: {}",
            runtime_usability_status(&manifest)
        );
    }
    if manifest.format != "pip" {
        bail!("ComfyUI installs require a rocm-cli managed Python ROCm install.");
    }
    let python = manifest
        .python_executable
        .as_deref()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .with_context(|| {
            "The selected ROCm install does not have a Python executable. Choose another ROCm install from /runtimes."
                .to_string()
        })?;
    Ok(SelectedRuntime { manifest, python })
}

fn select_default_runtime<'a>(
    config: &RocmCliConfig,
    manifests: &'a [therock::InstalledRuntimeManifest],
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(active_key) = config.active_runtime_key.as_deref() {
        return select_runtime_by_selector(manifests, active_key);
    }
    let Some(default_runtime_id) = config.default_runtime_id.as_deref() else {
        bail!("Choose a ROCm install from /runtimes before installing ComfyUI.");
    };
    let matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(default_runtime_id))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [manifest] => Ok(*manifest),
        [] => {
            bail!("The configured default ROCm install was not found. Choose one from /runtimes.")
        }
        _ => bail!("More than one ROCm install matches the default. Choose one from /runtimes."),
    }
}

fn select_runtime_by_selector<'a>(
    manifests: &'a [therock::InstalledRuntimeManifest],
    selector: &str,
) -> Result<&'a therock::InstalledRuntimeManifest> {
    if let Some(manifest) = manifests
        .iter()
        .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(selector))
    {
        return Ok(manifest);
    }
    let matches = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(selector))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [manifest] => Ok(*manifest),
        [] => bail!("ROCm install not found: {selector}"),
        _ => bail!(
            "More than one ROCm install matches `{selector}`. Choose the exact runtime key from /runtimes."
        ),
    }
}

fn download_and_extract_source(
    paths: &AppPaths,
    source_path: &Path,
    log: &mut fs::File,
) -> Result<()> {
    let archive_path = app_root(paths)
        .join("downloads")
        .join(COMFYUI_SOURCE_ARCHIVE_NAME);
    fs::create_dir_all(
        archive_path
            .parent()
            .context("ComfyUI archive path has no parent directory")?,
    )?;
    if !archive_path.is_file() {
        writeln!(log, "Downloading {COMFYUI_SOURCE_ARCHIVE_URL}.")?;
        download_file(COMFYUI_SOURCE_ARCHIVE_URL, &archive_path)?;
    } else {
        writeln!(
            log,
            "Using downloaded source archive {}.",
            archive_path.display()
        )?;
    }
    let extract_root = app_root(paths)
        .join("extract")
        .join(format!("source-{}", unix_time_millis()));
    fs::create_dir_all(&extract_root)
        .with_context(|| format!("failed to create {}", extract_root.display()))?;
    let archive = fs::File::open(&archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let decoder = GzDecoder::new(archive);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(&extract_root)
        .with_context(|| format!("failed to extract {}", archive_path.display()))?;
    let extracted = first_child_dir(&extract_root)?;
    if source_path.exists() {
        fs::remove_dir_all(source_path)
            .with_context(|| format!("failed to remove {}", source_path.display()))?;
    }
    fs::create_dir_all(
        source_path
            .parent()
            .context("ComfyUI source path has no parent directory")?,
    )?;
    fs::rename(&extracted, source_path).or_else(|_| {
        copy_dir_all(&extracted, source_path)?;
        fs::remove_dir_all(&extracted)?;
        Ok::<(), anyhow::Error>(())
    })?;
    fs::remove_dir_all(&extract_root).ok();
    writeln!(log, "Installed source at {}.", source_path.display())?;
    Ok(())
}

fn first_child_dir(root: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            return Ok(path);
        }
    }
    bail!("ComfyUI source archive did not contain a directory")
}

fn copy_dir_all(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).with_context(|| format!("failed to create {}", to.display()))?;
    for entry in fs::read_dir(from).with_context(|| format!("failed to read {}", from.display()))? {
        let entry = entry?;
        let path = entry.path();
        let target = to.join(entry.file_name());
        if path.is_dir() {
            copy_dir_all(&path, &target)?;
        } else {
            fs::copy(&path, &target).with_context(|| {
                format!("failed to copy {} to {}", path.display(), target.display())
            })?;
        }
    }
    Ok(())
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let response = ureq::get(url)
        .timeout(Duration::from_secs(120))
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    if response.status() != 200 {
        bail!("HTTP {} while downloading {url}", response.status());
    }
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read download {url}"))?;
    fs::write(destination, bytes)
        .with_context(|| format!("failed to write {}", destination.display()))
}

fn filtered_requirement_specs(requirements_path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(requirements_path)
        .with_context(|| format!("failed to read {}", requirements_path.display()))?;
    let mut output = Vec::new();
    for token in requirement_tokens(&text) {
        if requirement_package_name(&token)
            .map(|name| matches!(name.as_str(), "torch" | "torchvision" | "torchaudio"))
            .unwrap_or(false)
        {
            continue;
        }
        output.push(token);
    }
    Ok(output)
}

fn requirement_tokens(text: &str) -> Vec<String> {
    text.lines()
        .flat_map(|line| {
            line.split('#')
                .next()
                .unwrap_or_default()
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|token| !token.trim().is_empty())
        .collect()
}

fn requirement_package_name(spec: &str) -> Option<String> {
    let trimmed = spec.trim();
    if trimmed.starts_with('-') || trimmed.contains("://") {
        return None;
    }
    let end = trimmed
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .unwrap_or(trimmed.len());
    (end > 0).then(|| trimmed[..end].replace('_', "-").to_ascii_lowercase())
}

fn pip_install_args(pip_cache: &Path, packages: &[String]) -> Vec<String> {
    let mut args = vec![
        "-m".to_owned(),
        "pip".to_owned(),
        "install".to_owned(),
        "--upgrade".to_owned(),
        "--cache-dir".to_owned(),
        pip_cache.display().to_string(),
    ];
    args.extend(packages.iter().cloned());
    args
}

fn run_logged_command(
    program: &Path,
    args: Vec<String>,
    log: &mut fs::File,
    context_text: &str,
) -> Result<()> {
    writeln!(
        log,
        "command: {} {}",
        program.display(),
        args.iter()
            .map(|arg| quote_log_arg(arg))
            .collect::<Vec<_>>()
            .join(" ")
    )?;
    let mut child = Command::new(program)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("{context_text}: failed to run {}", program.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("child stdout was not captured")?;
    let stderr = child
        .stderr
        .take()
        .context("child stderr was not captured")?;
    let stdout_log = log
        .try_clone()
        .context("failed to clone ComfyUI install log for stdout")?;
    let stderr_log = log
        .try_clone()
        .context("failed to clone ComfyUI install log for stderr")?;
    let stdout_thread =
        thread::spawn(move || stream_logged_output(stdout, stdout_log, OutputTarget::Stdout));
    let stderr_thread =
        thread::spawn(move || stream_logged_output(stderr, stderr_log, OutputTarget::Stderr));
    let status = child
        .wait()
        .with_context(|| format!("{context_text}: failed waiting for {}", program.display()))?;
    stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{context_text}: stdout reader failed"))?
        .context("failed to stream command stdout")?;
    stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("{context_text}: stderr reader failed"))?
        .context("failed to stream command stderr")?;
    if status.success() {
        return Ok(());
    }
    bail!("{context_text}: command exited with status {status}")
}

enum OutputTarget {
    Stdout,
    Stderr,
}

fn stream_logged_output<R: Read>(
    mut reader: R,
    mut log: fs::File,
    target: OutputTarget,
) -> io::Result<()> {
    let mut buffer = [0_u8; 8192];
    loop {
        let len = reader.read(&mut buffer)?;
        if len == 0 {
            break;
        }
        log.write_all(&buffer[..len])?;
        match target {
            OutputTarget::Stdout => {
                let mut stdout = io::stdout().lock();
                stdout.write_all(&buffer[..len])?;
                stdout.flush()?;
            }
            OutputTarget::Stderr => {
                let mut stderr = io::stderr().lock();
                stderr.write_all(&buffer[..len])?;
                stderr.flush()?;
            }
        }
    }
    Ok(())
}

fn probe_comfyui(python: &Path, source_path: &Path) -> Result<ComfyUiProbe> {
    let script = format!(
        r#"
import json, sys
sys.path.insert(0, {source:?})
import torch
result = {{
    "torch_version": getattr(torch, "__version__", None),
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}}
print(json.dumps(result))
"#,
        source = source_path.display().to_string()
    );
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run GPU check with {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "ComfyUI GPU check failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).context("failed to parse ComfyUI GPU check output")
}

fn quote_log_arg(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=\\".contains(ch))
    {
        value.to_owned()
    } else {
        format!("{value:?}")
    }
}

fn open_browser(url: &str) -> Result<()> {
    let status = if cfg!(windows) {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    } else if cfg!(target_os = "macos") {
        Command::new("open")
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    } else {
        Command::new("xdg-open")
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
    }
    .context("failed to open browser")?;
    if status.success() {
        Ok(())
    } else {
        bail!("browser opener exited with status {status}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn requirements_filter_preserves_therock_torch_stack() {
        let text = "torch torchvision>=1 torchaudio\nnumpy>=1.25\n# comment\naiohttp\n";
        assert_eq!(
            requirement_tokens(text),
            vec![
                "torch".to_owned(),
                "torchvision>=1".to_owned(),
                "torchaudio".to_owned(),
                "numpy>=1.25".to_owned(),
                "aiohttp".to_owned()
            ]
        );
        let kept = requirement_tokens(text)
            .into_iter()
            .filter(|token| {
                !requirement_package_name(token)
                    .map(|name| matches!(name.as_str(), "torch" | "torchvision" | "torchaudio"))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        assert_eq!(kept, vec!["numpy>=1.25".to_owned(), "aiohttp".to_owned()]);
    }

    #[test]
    fn status_without_install_is_plain() -> Result<()> {
        let paths = test_paths("comfyui-status");
        let config = RocmCliConfig::default();
        let rendered = render_status(&paths, &config)?;
        assert!(rendered.contains("installed: no"));
        assert!(rendered.contains("next step: rocm comfyui install"));
        Ok(())
    }

    #[test]
    fn logs_render_inline_latest_output() -> Result<()> {
        let paths = test_paths("comfyui-logs");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        let install_log = logs.join("install-100.log");
        let run_log = logs.join("start-200.log");
        fs::write(&install_log, "install 1\ninstall 2\ninstall 3\n")?;
        fs::write(&run_log, "run 1\nrun 2\nrun 3\n")?;
        save_manifest(
            &paths,
            &ComfyUiManifest {
                app_id: APP_ID.to_owned(),
                runtime_key: "therock-release:gfx120X-all".to_owned(),
                runtime_id: "therock-release".to_owned(),
                runtime_version: "7.13.0a20260511".to_owned(),
                runtime_root: paths.data_dir.join("runtimes").join("runtime"),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                source_url: COMFYUI_SOURCE_ARCHIVE_URL.to_owned(),
                source_path: source_path(&paths),
                requirements_path: source_path(&paths).join("requirements.txt"),
                pip_cache_dir: app_root(&paths).join("pip-cache"),
                log_path: install_log.clone(),
                torch_version: Some("2.10.0".to_owned()),
                torch_cuda_available: true,
                installed_at_unix_ms: 100,
            },
        )?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: "http://127.0.0.1:8188".to_owned(),
                host: "127.0.0.1".to_owned(),
                port: 8188,
                pid: 42,
                source_path: source_path(&paths),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                log_path: run_log.clone(),
                started_at_unix_ms: 200,
            },
        )?;

        let rendered = render_logs(&paths, 2)?;

        assert!(rendered.contains("ComfyUI logs"));
        assert!(rendered.contains("Install log"));
        assert!(rendered.contains("Run log"));
        assert!(rendered.contains("saved file:"));
        assert!(rendered.contains(&install_log.display().to_string()));
        assert!(rendered.contains(&run_log.display().to_string()));
        assert!(rendered.contains("install 2"));
        assert!(rendered.contains("install 3"));
        assert!(!rendered.contains("install 1"));
        assert!(rendered.contains("run 2"));
        assert!(rendered.contains("run 3"));
        Ok(())
    }

    #[test]
    fn running_state_requires_saved_process_and_reachable_port() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let state = ComfyUiState {
            app_id: APP_ID.to_owned(),
            url: format!("http://127.0.0.1:{port}"),
            host: "127.0.0.1".to_owned(),
            port,
            pid: std::process::id(),
            source_path: PathBuf::from("source"),
            python_executable: PathBuf::from("python"),
            log_path: PathBuf::from("start.log"),
            started_at_unix_ms: 200,
        };

        let report = evaluate_running_state(&state);

        assert_eq!(report.state, ComfyUiRunState::Running);
        assert!(report.process_running);
        assert!(report.endpoint_reachable);
        Ok(())
    }

    #[test]
    fn status_reports_stopped_when_saved_comfyui_pid_is_gone() -> Result<()> {
        let paths = test_paths("comfyui-stale-state");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: "http://127.0.0.1:8188".to_owned(),
                host: "127.0.0.1".to_owned(),
                port: 8188,
                pid: 0,
                source_path: source_path(&paths),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                log_path: logs.join("start-200.log"),
                started_at_unix_ms: 200,
            },
        )?;

        let rendered = render_status(&paths, &RocmCliConfig::default())?;

        assert!(rendered.contains("status: stopped"));
        assert!(rendered.contains("next step: rocm comfyui start"));
        assert!(!rendered.contains("status: starting or running"));
        Ok(())
    }

    #[test]
    fn tui_status_reports_stale_saved_state_plainly() -> Result<()> {
        let paths = test_paths("comfyui-tui-stale-state");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: "http://127.0.0.1:8188".to_owned(),
                host: "127.0.0.1".to_owned(),
                port: 8188,
                pid: 0,
                source_path: source_path(&paths),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                log_path: logs.join("start-200.log"),
                started_at_unix_ms: 200,
            },
        )?;

        let rendered = render_tui_status(&paths, &RocmCliConfig::default())?;

        assert!(rendered.contains("status: stopped"));
        assert!(rendered.contains("Choose Start ComfyUI below"));
        assert!(!rendered.contains("starting or running"));
        Ok(())
    }

    #[test]
    fn tui_status_hides_technical_file_paths() -> Result<()> {
        let paths = test_paths("comfyui-tui-status");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        let install_log = logs.join("install-100.log");
        fs::write(&install_log, "install output\n")?;
        save_manifest(
            &paths,
            &ComfyUiManifest {
                app_id: APP_ID.to_owned(),
                runtime_key: "therock-release:gfx120X-all".to_owned(),
                runtime_id: "therock-release".to_owned(),
                runtime_version: "7.13.0a20260511".to_owned(),
                runtime_root: paths.data_dir.join("runtimes").join("runtime"),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                source_url: COMFYUI_SOURCE_ARCHIVE_URL.to_owned(),
                source_path: source_path(&paths),
                requirements_path: source_path(&paths).join("requirements.txt"),
                pip_cache_dir: app_root(&paths).join("pip-cache"),
                log_path: install_log.clone(),
                torch_version: Some("2.10.0".to_owned()),
                torch_cuda_available: true,
                installed_at_unix_ms: 100,
            },
        )?;

        let rendered = render_tui_status(&paths, &RocmCliConfig::default())?;

        assert!(rendered.contains("Installed"));
        assert!(rendered.contains("AMD GPU check: ready"));
        assert!(rendered.contains("Use the rows on the left"));
        assert!(!rendered.contains("python"));
        assert!(!rendered.contains("torch"));
        assert!(!rendered.contains("saved file:"));
        assert!(!rendered.contains(&install_log.display().to_string()));
        Ok(())
    }

    #[test]
    fn tui_logs_hide_file_paths_until_requested() -> Result<()> {
        let paths = test_paths("comfyui-tui-logs");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        let install_log = logs.join("install-100.log");
        fs::write(&install_log, "downloaded ComfyUI\ninstalled packages\n")?;

        let friendly = render_tui_logs(&paths, 10, false)?;
        assert!(friendly.contains("ComfyUI logs"));
        assert!(friendly.contains("Install log"));
        assert!(friendly.contains("downloaded ComfyUI"));
        assert!(friendly.contains("installed packages"));
        assert!(!friendly.contains("saved file:"));
        assert!(!friendly.contains(&install_log.display().to_string()));

        let with_files = render_tui_logs(&paths, 10, true)?;
        assert!(with_files.contains("saved file:"));
        assert!(with_files.contains(&install_log.display().to_string()));
        Ok(())
    }

    fn test_paths(name: &str) -> AppPaths {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rocm-cli-{name}-{nonce}"));
        AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }
}
