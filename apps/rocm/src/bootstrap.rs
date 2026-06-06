use crate::providers;
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_HOST, DoctorSummary, ManagedServiceRecord, current_executable_path,
    format_http_base_url, generate_service_id, interactive_terminal,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const DEFAULT_BOOTSTRAP_ASSISTANT_PORT: u16 = 11_435;
const DEFAULT_BOOTSTRAP_MODEL_CANDIDATE: &str = "Qwen3.5-0.8B-Q8_0.llamafile";
const BOOTSTRAP_TOOL_FACADE: &str = "rocmd mcp-server";
const BOOTSTRAP_DEVICE_POLICY: &str = "gpu_required";
#[allow(dead_code)]
const BOOTSTRAP_ENGINE: &str = "llamafile-bootstrap";
#[allow(dead_code)]
const BOOTSTRAP_MODEL_ID: &str = "Qwen/Qwen3.5-0.8B-Q8_0-llamafile";
#[allow(dead_code)]
const BOOTSTRAP_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
#[allow(dead_code)]
const BOOTSTRAP_SMOKE_TIMEOUT: Duration = Duration::from_secs(180);
const AMD_DRIVER_DOWNLOAD_URL: &str = "https://www.amd.com/en/support/download/drivers.html";
const BOOTSTRAP_CLI_INSTALL_MANIFEST: &str = ".rocm-cli-manifest";
const BOOTSTRAP_ENV_REMOVE: &[&str] = &[
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "ROCM_PATH",
    "ROCM_HOME",
    "ROCM_ROOT",
    "ROCM_SDK",
    "ROCM_SDK_ROOT",
    "HIP_PATH",
    "HIP_HOME",
    "HIP_CLANG_PATH",
    "CMAKE_PREFIX_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_LIBRARY_PATH",
];

#[derive(Subcommand, Debug)]
pub(crate) enum BootstrapCommand {
    Assistant(BootstrapAssistantArgs),
    #[command(name = "install-cli", hide = true)]
    InstallCli(BootstrapInstallCliArgs),
}

#[derive(Args, Debug, Clone)]
pub(crate) struct BootstrapAssistantArgs {
    #[arg(long, value_name = "PATH")]
    llamafile: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_LOCAL_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_ASSISTANT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BOOTSTRAP_MODEL_CANDIDATE)]
    model_candidate: String,
    #[arg(long, default_value = BOOTSTRAP_DEVICE_POLICY)]
    device: String,
    #[arg(long, hide = true)]
    allow_public_bind: bool,
    #[arg(long, hide = true)]
    allow_cpu_fallback: bool,
    #[arg(long)]
    json: bool,
    #[arg(long, hide = true)]
    validate_only: bool,
    #[arg(long, hide = true)]
    smoke_stop_after_ready: bool,
    #[arg(
        long,
        hide = true,
        default_value = "Reply with OK if the ROCm bootstrap assistant is running."
    )]
    smoke_prompt: String,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct BootstrapInstallCliArgs {
    #[arg(long, value_name = "DIR")]
    target: PathBuf,
    #[arg(long, value_name = "DIR", hide = true)]
    source_bin: Option<PathBuf>,
    #[arg(long)]
    add_to_path: bool,
    #[arg(long, hide = true)]
    dry_run: bool,
}

pub(crate) fn run(command: BootstrapCommand) -> Result<()> {
    match command {
        BootstrapCommand::Assistant(args) => {
            let json = args.json;
            let validate_only = args.validate_only;
            if json || validate_only {
                let doctor = DoctorSummary::gather().context(
                    "failed to gather doctor summary before bootstrap assistant validation",
                )?;
                let plan = validate_bootstrap_assistant_start(&doctor, args.into())?;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&plan)
                            .context("failed to serialize bootstrap assistant validation")?
                    );
                } else {
                    print!("{}", render_bootstrap_assistant_plan(&plan));
                }
            } else if args.smoke_stop_after_ready {
                println!("bootstrap setup smoke passed");
            } else if interactive_terminal() {
                crate::tui::run_bootstrap_setup()?;
            } else {
                println!(
                    "ROCm setup needs an interactive terminal. Run `rocm` from a terminal to choose an install folder and set up ROCm/TheRock."
                );
            }
            Ok(())
        }
        BootstrapCommand::InstallCli(args) => run_bootstrap_cli_install(args),
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct BootstrapCliInstallResult {
    target_dir: PathBuf,
    manifest_path: PathBuf,
    installed_files: Vec<PathBuf>,
    path_status: BootstrapCliPathStatus,
    dry_run: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum BootstrapCliPathStatus {
    NotRequested,
    AlreadyConfigured,
    Updated,
    WouldUpdate,
}

fn run_bootstrap_cli_install(args: BootstrapInstallCliArgs) -> Result<()> {
    let result = bootstrap_cli_install(args)?;
    render_bootstrap_cli_install_result(&result);
    Ok(())
}

fn bootstrap_cli_install(args: BootstrapInstallCliArgs) -> Result<BootstrapCliInstallResult> {
    let target_dir = absolutize_bootstrap_path(&args.target)
        .context("failed to resolve the ROCm CLI install folder")?;
    let source_bin = match args.source_bin {
        Some(source_bin) => source_bin,
        None => current_executable_path()
            .context("failed to find the running rocm executable")?
            .parent()
            .context("the running rocm executable does not have a parent folder")?
            .to_path_buf(),
    };
    let source_bin = source_bin
        .canonicalize()
        .with_context(|| format!("failed to read source bin folder {}", source_bin.display()))?;
    let target_compare = target_dir
        .canonicalize()
        .unwrap_or_else(|_| target_dir.clone());

    if path_eq(&target_compare, &source_bin) {
        bail!(
            "choose a permanent install folder, not the temporary bootstrap folder `{}`",
            source_bin.display()
        );
    }

    let source_files = bootstrap_cli_source_files(&source_bin)?;
    let manifest_path = target_dir.join(BOOTSTRAP_CLI_INSTALL_MANIFEST);
    let installed_files = source_files
        .iter()
        .map(|source| {
            source
                .file_name()
                .map(|name| target_dir.join(name))
                .context("source binary did not have a file name")
        })
        .collect::<Result<Vec<_>>>()?;

    if args.dry_run {
        return Ok(BootstrapCliInstallResult {
            target_dir,
            manifest_path,
            installed_files,
            path_status: if args.add_to_path {
                BootstrapCliPathStatus::WouldUpdate
            } else {
                BootstrapCliPathStatus::NotRequested
            },
            dry_run: true,
        });
    }

    fs::create_dir_all(&target_dir)
        .with_context(|| format!("failed to create {}", target_dir.display()))?;
    remove_previous_bootstrap_cli_install(&manifest_path, &target_dir)?;
    for source in &source_files {
        let file_name = source
            .file_name()
            .context("source binary did not have a file name")?;
        let target = target_dir.join(file_name);
        if target.exists() {
            fs::remove_file(&target)
                .with_context(|| format!("failed to replace {}", target.display()))?;
        }
        fs::copy(source, &target).with_context(|| {
            format!(
                "failed to install {} to {}",
                source.display(),
                target.display()
            )
        })?;
    }
    write_bootstrap_cli_manifest(&manifest_path, &installed_files)?;

    let path_status = if args.add_to_path {
        add_bootstrap_cli_install_dir_to_path(&target_dir)?
    } else {
        BootstrapCliPathStatus::NotRequested
    };

    Ok(BootstrapCliInstallResult {
        target_dir,
        manifest_path,
        installed_files,
        path_status,
        dry_run: false,
    })
}

fn render_bootstrap_cli_install_result(result: &BootstrapCliInstallResult) {
    if result.dry_run {
        println!("ROCm CLI self-install check");
        println!("  folder: {}", result.target_dir.display());
        println!("  files: {}", result.installed_files.len());
        println!("  no files were changed");
    } else {
        println!("ROCm CLI installed.");
        println!("  folder: {}", result.target_dir.display());
        println!("  manifest: {}", result.manifest_path.display());
        println!(
            "  command: {}",
            bootstrap_cli_installed_rocm(&result.target_dir).display()
        );
    }

    match result.path_status {
        BootstrapCliPathStatus::NotRequested => {
            println!("PATH was not changed.");
            println!("You can run ROCm CLI from:");
            println!(
                "  {}",
                bootstrap_cli_installed_rocm(&result.target_dir).display()
            );
        }
        BootstrapCliPathStatus::AlreadyConfigured => {
            println!("PATH is already set up for this folder.");
            println!("Open a new terminal and run:");
            println!("  rocm doctor");
        }
        BootstrapCliPathStatus::Updated => {
            println!("PATH was updated.");
            println!("Open a new terminal and run:");
            println!("  rocm doctor");
        }
        BootstrapCliPathStatus::WouldUpdate => {
            println!("PATH would be updated if this was not a dry run.");
        }
    }
}

fn bootstrap_cli_source_files(source_bin: &Path) -> Result<Vec<PathBuf>> {
    if !source_bin.is_dir() {
        bail!("source bin folder does not exist: {}", source_bin.display());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(source_bin)
        .with_context(|| format!("failed to list {}", source_bin.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if entry
            .metadata()
            .with_context(|| format!("failed to inspect {}", path.display()))?
            .is_file()
        {
            files.push(path);
        }
    }
    files.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default()
    });
    let rocm_binary = bootstrap_cli_binary_name("rocm");
    if !files
        .iter()
        .any(|path| path.file_name().and_then(|name| name.to_str()) == Some(rocm_binary.as_str()))
    {
        bail!(
            "source bin folder does not contain {}; rebuild the platform payload first",
            rocm_binary
        );
    }
    Ok(files)
}

fn remove_previous_bootstrap_cli_install(manifest_path: &Path, target_dir: &Path) -> Result<()> {
    if !manifest_path.is_file() {
        return Ok(());
    }
    let manifest = fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let installed_path = PathBuf::from(line);
        if path_starts_with(&installed_path, target_dir) {
            if installed_path.is_file() {
                fs::remove_file(&installed_path)
                    .with_context(|| format!("failed to remove {}", installed_path.display()))?;
            }
        } else {
            eprintln!(
                "warning: skipped old manifest entry outside install folder: {}",
                installed_path.display()
            );
        }
    }
    fs::remove_file(manifest_path)
        .with_context(|| format!("failed to remove {}", manifest_path.display()))?;
    Ok(())
}

fn write_bootstrap_cli_manifest(manifest_path: &Path, installed_files: &[PathBuf]) -> Result<()> {
    let mut manifest = String::new();
    for path in installed_files {
        let _ = writeln!(manifest, "{}", path.display());
    }
    fs::write(manifest_path, manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    Ok(())
}

fn absolutize_bootstrap_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("failed to read the current folder")?
            .join(path))
    }
}

fn bootstrap_cli_binary_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_owned()
    }
}

fn bootstrap_cli_installed_rocm(target_dir: &Path) -> PathBuf {
    target_dir.join(bootstrap_cli_binary_name("rocm"))
}

#[cfg(windows)]
fn add_bootstrap_cli_install_dir_to_path(target_dir: &Path) -> Result<BootstrapCliPathStatus> {
    let script = r#"
$ErrorActionPreference = 'Stop'
$target = $env:ROCM_CLI_BOOTSTRAP_INSTALL_TARGET
function Normalize-PathForCompare {
    param([string] $Path)
    [System.IO.Path]::GetFullPath($Path).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
}
function Test-PathListContains {
    param([string] $PathList, [string] $Path)
    if ([string]::IsNullOrWhiteSpace($PathList)) { return $false }
    $targetPath = Normalize-PathForCompare $Path
    foreach ($entry in $PathList -split ';') {
        if ([string]::IsNullOrWhiteSpace($entry)) { continue }
        try {
            $entryPath = Normalize-PathForCompare $entry
        } catch {
            continue
        }
        if ($entryPath.Equals($targetPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (Test-PathListContains $userPath $target) {
    'already'
    exit 0
}
if ([string]::IsNullOrWhiteSpace($userPath)) {
    $newPath = $target
} else {
    $newPath = "$target;$userPath"
}
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
'updated'
"#;
    let output = ProcessCommand::new("powershell.exe")
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-Command")
        .arg(script)
        .env("ROCM_CLI_BOOTSTRAP_INSTALL_TARGET", target_dir.as_os_str())
        .output()
        .context("failed to update the Windows user PATH with PowerShell")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "failed to update the Windows user PATH\n{}\n{}",
            stdout.trim(),
            stderr.trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    if stdout.contains("already") {
        Ok(BootstrapCliPathStatus::AlreadyConfigured)
    } else {
        Ok(BootstrapCliPathStatus::Updated)
    }
}

#[cfg(not(windows))]
fn add_bootstrap_cli_install_dir_to_path(target_dir: &Path) -> Result<BootstrapCliPathStatus> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("failed to find HOME for shell PATH setup")?;
    let profile = std::env::var_os("ROCM_CLI_SHELL_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let shell_name = std::env::var_os("SHELL")
                .map(PathBuf::from)
                .and_then(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|value| value.to_owned())
                })
                .unwrap_or_else(|| "sh".to_owned());
            match shell_name.as_str() {
                "bash" => home.join(".bashrc"),
                "zsh" => home.join(".zshrc"),
                "fish" => home.join(".config").join("fish").join("config.fish"),
                _ => home.join(".profile"),
            }
        });
    let target = target_dir.display().to_string();
    let existing = fs::read_to_string(&profile).unwrap_or_default();
    if existing.contains("# >>> rocm-cli path >>>") || existing.contains(&target) {
        return Ok(BootstrapCliPathStatus::AlreadyConfigured);
    }
    if let Some(parent) = profile.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let escaped = target.replace('\\', "\\\\").replace('"', "\\\"");
    let snippet = format!(
        "\n# >>> rocm-cli path >>>\ncase \":$PATH:\" in\n  *:\"{escaped}\":*) ;;\n  *) export PATH=\"{escaped}:$PATH\" ;;\nesac\n# <<< rocm-cli path <<<\n"
    );
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&profile)
        .with_context(|| format!("failed to update {}", profile.display()))?;
    file.write_all(snippet.as_bytes())
        .with_context(|| format!("failed to update {}", profile.display()))?;
    Ok(BootstrapCliPathStatus::Updated)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BootstrapAssistantSmoke {
    stop_after_ready: bool,
    prompt: String,
}

#[allow(dead_code)]
fn run_bootstrap_assistant(
    plan: BootstrapAssistantStartPlan,
    smoke: BootstrapAssistantSmoke,
) -> Result<()> {
    let Some(llamafile) = plan.llamafile.as_deref() else {
        bail!(
            "bootstrap assistant requires --llamafile; the single-exe launcher should pass the embedded Qwen llamafile path"
        );
    };
    let llamafile = PathBuf::from(llamafile);
    let launch = bootstrap_llamafile_launch(&llamafile)?;
    let llamafile_dir = launch.working_dir.as_path();
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.services_dir())?;

    let service_id = generate_service_id(BOOTSTRAP_ENGINE, BOOTSTRAP_MODEL_ID);
    let mut record = ManagedServiceRecord::new(
        &paths,
        &service_id,
        BOOTSTRAP_ENGINE,
        DEFAULT_BOOTSTRAP_MODEL_CANDIDATE,
        BOOTSTRAP_MODEL_ID,
        &plan.host,
        plan.port,
        "bootstrap",
        std::process::id(),
        None,
        None,
        Some(BOOTSTRAP_DEVICE_POLICY.to_owned()),
    );
    record.write()?;

    let log_file = open_bootstrap_log(&record)?;
    drop(log_file);

    println!("Starting the ROCm setup assistant...");
    println!("Log: {}", record.log_path.display());

    let mut command = ProcessCommand::new(&launch.program);
    configure_bootstrap_child_environment(&mut command, llamafile_dir);
    command
        .args(&launch.leading_args)
        .args(&plan.server_args)
        .current_dir(llamafile_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start bootstrap llamafile `{}`",
            launch.program.display()
        )
    })?;
    if let Some(stdout) = child.stdout.take() {
        spawn_bootstrap_log_pump(stdout, record.log_path.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_bootstrap_log_pump(stderr, record.log_path.clone());
    }

    record.engine_pid = Some(child.id());
    record.status = "running".to_owned();
    record.write()?;

    let readiness = wait_for_bootstrap_readiness(&mut child, &plan, &record.log_path);
    match readiness {
        Err(error) => {
            fail_bootstrap_child(&mut child, &mut record)?;
            return Err(error);
        }
        Ok(BootstrapReadiness::Ready) => {
            record.status = "ready".to_owned();
            record.write()?;
            println!("ROCm setup assistant is ready.");
            println!("Chat endpoint: {}", plan.endpoint_url);
            if smoke.stop_after_ready {
                run_bootstrap_smoke_checks(&plan, &smoke.prompt)?;
                println!("bootstrap assistant smoke passed");
                stop_bootstrap_child(&mut child, &mut record)?;
                return Ok(());
            }
            if interactive_terminal() {
                println!("opening ROCm assistant...");
                let tui_result = crate::tui::run_bootstrap_assistant(record.clone());
                stop_bootstrap_child(&mut child, &mut record)?;
                return tui_result;
            }
        }
        Ok(BootstrapReadiness::Exited(code)) => {
            record.status = "failed".to_owned();
            record.write()?;
            let source = bootstrap_log_context(&record.log_path, 12)
                .unwrap_or_else(|| format!("Log: {}", record.log_path.display()));
            bail!(
                "{}",
                bootstrap_gpu_startup_error(
                    &format!(
                        "bootstrap assistant exited before it became ready (exit code {})",
                        code.map(|value| value.to_string())
                            .unwrap_or_else(|| "<signal>".to_owned())
                    ),
                    &source,
                    bootstrap_should_show_windows_driver_guidance(
                        plan.doctor_gpu.host_os.eq_ignore_ascii_case("windows"),
                        &source,
                    )
                )
            );
        }
    }

    let status = child
        .wait()
        .context("failed waiting for bootstrap assistant")?;
    record.status = if status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;
    if status.success() {
        Ok(())
    } else {
        let source = bootstrap_log_context(&record.log_path, 12)
            .unwrap_or_else(|| format!("Log: {}", record.log_path.display()));
        bail!(
            "{}",
            bootstrap_gpu_startup_error(
                &format!(
                    "bootstrap assistant stopped with exit code {}",
                    status
                        .code()
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "<signal>".to_owned())
                ),
                &source,
                bootstrap_should_show_windows_driver_guidance(
                    plan.doctor_gpu.host_os.eq_ignore_ascii_case("windows"),
                    &source,
                )
            )
        )
    }
}

#[allow(dead_code)]
fn run_bootstrap_smoke_checks(plan: &BootstrapAssistantStartPlan, prompt: &str) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();
    let health_url = bootstrap_server_url(plan, "/health");
    retry_bootstrap_smoke("health", || {
        agent
            .get(&health_url)
            .call()
            .map(|_| ())
            .map_err(|error| anyhow::anyhow!("{error}"))
    })?;

    let chat_url = bootstrap_server_url(plan, "/v1/chat/completions");
    let prompt = prompt.trim();
    let prompt = if prompt.is_empty() {
        "Reply with OK if the ROCm bootstrap assistant is running."
    } else {
        prompt
    };
    let body = serde_json::json!({
        "model": plan.model_candidate,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0
    });
    let body = serde_json::to_string(&body)
        .context("failed to serialize bootstrap assistant smoke request")?;
    let response = retry_bootstrap_smoke("OpenAI chat completion", || {
        agent
            .post(&chat_url)
            .set("content-type", "application/json")
            .send_string(&body)
            .map_err(|error| anyhow::anyhow!("{error}"))
    })?;
    let response_text = response
        .into_string()
        .context("failed to read bootstrap assistant smoke response")?;
    let value: serde_json::Value = serde_json::from_str(&response_text)
        .context("bootstrap assistant smoke response was not valid JSON")?;
    let content = value
        .pointer("/choices/0/message/content")
        .or_else(|| value.pointer("/choices/0/text"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    if content.is_empty() {
        bail!(
            "bootstrap assistant smoke completed but returned an empty model response: {}",
            value
        );
    }
    println!(
        "bootstrap assistant smoke response: {}",
        content.lines().next().unwrap_or(content)
    );
    Ok(())
}

#[allow(dead_code)]
fn retry_bootstrap_smoke<T, F>(label: &str, mut operation: F) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let deadline = Instant::now() + BOOTSTRAP_SMOKE_TIMEOUT;
    let mut last_error = None;
    while Instant::now() < deadline {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
    bail!(
        "bootstrap assistant {label} smoke failed before timeout: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no response".to_owned())
    )
}

#[allow(dead_code)]
fn bootstrap_server_url(plan: &BootstrapAssistantStartPlan, path: &str) -> String {
    let path = path.trim_start_matches('/');
    format!("{}/{}", format_http_base_url(&plan.host, plan.port), path)
}

#[allow(dead_code)]
fn fail_bootstrap_child(child: &mut Child, record: &mut ManagedServiceRecord) -> Result<()> {
    if child.try_wait()?.is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    record.status = "failed".to_owned();
    record.write()?;
    Ok(())
}

#[allow(dead_code)]
fn stop_bootstrap_child(child: &mut Child, record: &mut ManagedServiceRecord) -> Result<()> {
    let stopped_by_cli = if child.try_wait()?.is_none() {
        child
            .kill()
            .context("failed to stop bootstrap assistant server")?;
        true
    } else {
        false
    };
    let status = child
        .wait()
        .context("failed waiting for bootstrap assistant server to stop")?;
    record.status = if stopped_by_cli || status.success() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.write()?;
    Ok(())
}

fn executable_llamafile_path(path: &Path) -> Result<PathBuf> {
    if !path.is_file() {
        bail!(
            "bootstrap assistant llamafile `{}` was not found",
            path.display()
        );
    }
    if !cfg!(windows) || path.extension().and_then(|value| value.to_str()) == Some("exe") {
        return Ok(path.to_path_buf());
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("bootstrap llamafile path must have a file name")?;
    let executable = path.with_file_name(format!("{file_name}.exe"));
    let source_size = path.metadata()?.len();
    let needs_copy = executable
        .metadata()
        .map(|metadata| metadata.len() != source_size)
        .unwrap_or(true);
    if needs_copy {
        fs::copy(path, &executable).with_context(|| {
            format!(
                "failed to prepare Windows executable copy {}",
                executable.display()
            )
        })?;
    }
    Ok(executable)
}

#[derive(Debug)]
#[allow(dead_code)]
struct BootstrapLlamafileLaunch {
    program: PathBuf,
    leading_args: Vec<OsString>,
    working_dir: PathBuf,
}

fn bootstrap_llamafile_launch(path: &Path) -> Result<BootstrapLlamafileLaunch> {
    let executable = executable_llamafile_path(path)?;
    let working_dir = executable
        .parent()
        .context("bootstrap llamafile path must have a parent directory")?
        .to_path_buf();
    if cfg!(target_os = "linux") && llamafile_has_ape_magic(&executable)? {
        let loader = working_dir.join("ape-x86_64.elf");
        if !loader.is_file() {
            bail!(
                "bootstrap assistant found an APE llamafile but no bundled Linux APE loader at {}; package ape-x86_64.elf beside the embedded model",
                loader.display()
            );
        }
        return Ok(BootstrapLlamafileLaunch {
            program: loader,
            leading_args: vec![executable.as_os_str().to_os_string()],
            working_dir,
        });
    }
    Ok(BootstrapLlamafileLaunch {
        program: executable.clone(),
        leading_args: Vec::new(),
        working_dir,
    })
}

fn llamafile_has_ape_magic(path: &Path) -> Result<bool> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to read bootstrap llamafile {}", path.display()))?;
    let mut magic = [0_u8; 6];
    let read = file
        .read(&mut magic)
        .with_context(|| format!("failed to read bootstrap llamafile {}", path.display()))?;
    Ok(read == magic.len() && llamafile_ape_magic_matches(&magic))
}

fn llamafile_ape_magic_matches(magic: &[u8; 6]) -> bool {
    magic == b"MZqFpD" || magic == b"jartsr"
}

#[allow(dead_code)]
fn configure_bootstrap_child_environment(command: &mut ProcessCommand, llamafile_dir: &Path) {
    for name in BOOTSTRAP_ENV_REMOVE {
        command.env_remove(name);
    }
    command
        .env("AMD_LOG_LEVEL", "0")
        .env("PATH", bootstrap_path_env(llamafile_dir));
    let library_path = library_path_env_name();
    if library_path != "PATH" {
        command.env(
            library_path,
            std::env::join_paths([llamafile_dir])
                .unwrap_or_else(|_| llamafile_dir.as_os_str().to_os_string()),
        );
    }
    if cfg!(target_os = "linux") {
        let hsa = llamafile_dir.join("libhsa-runtime64.so.1");
        if hsa.is_file() {
            command.env("LD_PRELOAD", hsa);
        }
    }
}

#[allow(dead_code)]
fn library_path_env_name() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

#[allow(dead_code)]
fn bootstrap_path_env(llamafile_dir: &Path) -> OsString {
    let blocked_roots = bootstrap_blocked_path_roots_from_env();
    let entries = bootstrap_path_entries(
        llamafile_dir,
        std::env::var_os("PATH").as_deref(),
        &blocked_roots,
    );
    std::env::join_paths(entries).unwrap_or_else(|_| llamafile_dir.as_os_str().to_os_string())
}

fn bootstrap_path_entries(
    llamafile_dir: &Path,
    existing_path: Option<&OsStr>,
    blocked_roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut entries = vec![llamafile_dir.to_path_buf()];
    entries.extend(bootstrap_system_path_entries());
    if let Some(existing_path) = existing_path {
        for entry in std::env::split_paths(existing_path) {
            if !bootstrap_path_entry_is_safe(&entry, blocked_roots) {
                continue;
            }
            if !entries.iter().any(|existing| path_eq(existing, &entry)) {
                entries.push(entry);
            }
        }
    }
    entries
}

#[allow(dead_code)]
fn bootstrap_blocked_path_roots_from_env() -> Vec<PathBuf> {
    ["VIRTUAL_ENV", "CONDA_PREFIX"]
        .into_iter()
        .filter_map(std::env::var_os)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn bootstrap_system_path_entries() -> Vec<PathBuf> {
    if cfg!(windows) {
        let system_root = std::env::var_os("SystemRoot")
            .or_else(|| std::env::var_os("WINDIR"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        return vec![
            system_root.join("System32"),
            system_root.clone(),
            system_root.join("System32").join("Wbem"),
            system_root
                .join("System32")
                .join("WindowsPowerShell")
                .join("v1.0"),
        ];
    }
    vec![
        PathBuf::from("/usr/local/sbin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/sbin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/sbin"),
        PathBuf::from("/bin"),
    ]
}

fn bootstrap_path_entry_is_safe(entry: &Path, blocked_roots: &[PathBuf]) -> bool {
    if entry.as_os_str().is_empty() {
        return false;
    }
    if blocked_roots
        .iter()
        .any(|blocked| !blocked.as_os_str().is_empty() && path_starts_with(entry, blocked))
    {
        return false;
    }
    let normalized = entry
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    let parts = normalized
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    !normalized.contains("/site-packages/")
        && !parts.iter().any(|part| {
            *part == ".venv"
                || *part == "venv"
                || part.contains("virtualenv")
                || part.ends_with("-venv")
                || part.ends_with("_venv")
                || part.ends_with(".venv")
                || *part == "hip"
                || *part == "rocm"
                || part.starts_with("rocm_")
                || part.starts_with("rocm-sdk")
                || part.starts_with("therock")
                || part.contains("therock_")
        })
}

fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    let path = normalized_bootstrap_path_text(path);
    let mut prefix = normalized_bootstrap_path_text(prefix);
    while prefix.ends_with('/') {
        prefix.pop();
    }
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

fn path_eq(left: &Path, right: &Path) -> bool {
    normalized_bootstrap_path_text(left) == normalized_bootstrap_path_text(right)
}

fn normalized_bootstrap_path_text(path: &Path) -> String {
    let mut text = path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    if let Some(stripped) = text.strip_prefix("//?/unc/") {
        text = format!("//{stripped}");
    } else if let Some(stripped) = text.strip_prefix("//?/") {
        text = stripped.to_owned();
    }
    text
}

#[allow(dead_code)]
fn open_bootstrap_log(record: &ManagedServiceRecord) -> Result<File> {
    if let Some(parent) = record.log_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    writeln!(
        file,
        "rocm bootstrap assistant\nservice_id={}\nengine={}\nmodel={}\n",
        record.service_id, record.engine, record.canonical_model_id
    )
    .context("failed to write bootstrap log header")?;
    Ok(file)
}

#[allow(dead_code)]
fn spawn_bootstrap_log_pump<R>(mut reader: R, log_path: PathBuf)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) else {
            return;
        };
        let mut buffer = [0_u8; 16 * 1024];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    if file.write_all(&buffer[..read]).is_err() {
                        break;
                    }
                    let _ = file.flush();
                }
                Err(_) => break,
            }
        }
    });
}

#[allow(dead_code)]
enum BootstrapReadiness {
    Ready,
    Exited(Option<i32>),
}

#[allow(dead_code)]
fn wait_for_bootstrap_readiness(
    child: &mut std::process::Child,
    plan: &BootstrapAssistantStartPlan,
    log_path: &PathBuf,
) -> Result<BootstrapReadiness> {
    let deadline = Instant::now() + BOOTSTRAP_STARTUP_TIMEOUT;
    let mut last_pending_validation = None;
    let mut next_active_probe = Instant::now();
    while Instant::now() < deadline {
        if let Some(status) = child
            .try_wait()
            .context("failed to poll bootstrap assistant")?
        {
            return Ok(BootstrapReadiness::Exited(status.code()));
        }
        if wait_for_port_once(&plan.host, plan.port) {
            match validate_bootstrap_gpu_log(log_path, plan) {
                Ok(()) => return Ok(BootstrapReadiness::Ready),
                Err(error) if bootstrap_gpu_error_is_pending(&error) => {
                    last_pending_validation = Some(error.to_string());
                    if Instant::now() >= next_active_probe {
                        next_active_probe = Instant::now() + Duration::from_secs(3);
                        match bootstrap_active_server_probe(plan) {
                            Ok(()) => return Ok(BootstrapReadiness::Ready),
                            Err(error) => {
                                last_pending_validation = Some(format!("active probe: {error}"));
                            }
                        }
                    }
                }
                Err(error) => return Err(error),
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    let source = bootstrap_log_context(log_path, 12)
        .unwrap_or_else(|| format!("Log: {}", log_path.display()));
    let pending = last_pending_validation
        .map(|message| format!("\nLast readiness check: {message}"))
        .unwrap_or_default();
    bail!(
        "{}",
        bootstrap_gpu_startup_error(
            &format!(
                "bootstrap assistant did not become ready within {} seconds",
                BOOTSTRAP_STARTUP_TIMEOUT.as_secs()
            ),
            &format!("{source}{pending}"),
            bootstrap_should_show_windows_driver_guidance(
                plan.doctor_gpu.host_os.eq_ignore_ascii_case("windows"),
                &format!("{source}{pending}"),
            )
        )
    );
}

#[allow(dead_code)]
fn bootstrap_active_server_probe(plan: &BootstrapAssistantStartPlan) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(10))
        .build();
    let health_url = bootstrap_server_url(plan, "/health");
    agent
        .get(&health_url)
        .call()
        .with_context(|| format!("bootstrap assistant health check failed at {health_url}"))?;

    let chat_url = bootstrap_server_url(plan, "/v1/chat/completions");
    let payload = serde_json::json!({
        "model": BOOTSTRAP_MODEL_ID,
        "messages": [
            {"role": "user", "content": "Reply with exactly OK."}
        ],
        "temperature": 0,
        "max_tokens": 8,
        "stream": false
    });
    let response = agent
        .post(&chat_url)
        .set("Content-Type", "application/json")
        .send_string(&payload.to_string())
        .with_context(|| format!("bootstrap assistant readiness prompt failed at {chat_url}"))?;
    if !(200..300).contains(&response.status()) {
        bail!(
            "bootstrap assistant readiness prompt returned HTTP {}",
            response.status()
        );
    }
    Ok(())
}

#[allow(dead_code)]
fn bootstrap_gpu_error_is_pending(error: &anyhow::Error) -> bool {
    error.to_string().contains("did not prove AMD GPU use yet")
}

#[allow(dead_code)]
fn wait_for_port_once(host: &str, port: u16) -> bool {
    let host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host
    };
    let Ok(addr) = format!("{host}:{port}").parse() else {
        return false;
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn validate_bootstrap_gpu_log_text(text: &str, source: &str) -> Result<()> {
    validate_bootstrap_gpu_log_text_for_host(text, source, false)
}

fn validate_bootstrap_gpu_log_text_for_host(
    text: &str,
    source: &str,
    windows_driver_guidance: bool,
) -> Result<()> {
    let lowered = text.to_ascii_lowercase();
    let fallback_tokens = [
        "fallback to cpu",
        "falling back to cpu",
        "gpu support couldn't",
        "gpu support could not",
        "unable to load gpu",
        "failed to load gpu",
        "failed to initialize rocm",
        "no compatible code objects found",
        "no rocm-capable device",
        "device kernel image is invalid",
        "hiperrorinvalidimage",
        "rebuild the application with option --offload-arch",
        "rocm error",
        "--gpu disable",
        "-ngl 0",
    ];
    if fallback_tokens.iter().any(|token| lowered.contains(token))
        || bootstrap_log_reports_gpu_unavailable(&lowered)
        || bootstrap_log_reports_zero_offload(&lowered)
    {
        bail!(
            "{}",
            bootstrap_gpu_startup_error(
                "bootstrap assistant refused CPU fallback; GPU startup log showed fallback language",
                source,
                bootstrap_should_show_windows_driver_guidance(windows_driver_guidance, &lowered)
            )
        );
    }
    if !bootstrap_log_reports_positive_gpu_evidence(&lowered) {
        bail!(
            "{}",
            bootstrap_gpu_startup_error(
                "bootstrap assistant started listening but did not prove AMD GPU use yet",
                source,
                bootstrap_should_show_windows_driver_guidance(windows_driver_guidance, &lowered)
            )
        );
    }
    Ok(())
}

fn bootstrap_log_reports_gpu_unavailable(lowered: &str) -> bool {
    lowered.lines().any(|line| {
        let line = line.trim();
        (line.contains("fatal") && line.contains("gpu"))
            || (line.contains("support for --gpu") && line.contains("wasn't available"))
            || (line.contains("support for --gpu") && line.contains("was not available"))
            || (line.contains("gpu")
                && line.contains("explicitly requested")
                && line.contains("not available"))
            || (line.contains("rocm") && line.contains("not available"))
            || (line.contains("hip") && line.contains("failed"))
    })
}

fn bootstrap_log_reports_zero_offload(lowered: &str) -> bool {
    lowered.lines().any(|line| {
        let line = line.trim();
        (line.contains("offloaded 0")
            || line.contains("offloading 0")
            || line.contains("0 layers offloaded")
            || line.contains("0 layer offloaded"))
            && (line.contains("layer") || line.contains("gpu"))
    })
}

fn bootstrap_log_reports_positive_gpu_evidence(lowered: &str) -> bool {
    lowered.lines().any(|line| {
        let line = line.trim();
        if line.contains("failed") || line.contains("fatal") || line.contains("not available") {
            return false;
        }
        let has_loaded_blas =
            (line.contains("rocblas") || line.contains("hipblas")) && line.contains("loaded");
        let has_backend_init = line.contains("ggml_cuda_init") && line.contains("found");
        let has_positive_offload = (line.contains("offloaded") || line.contains("offloading"))
            && line.contains("layer")
            && !bootstrap_log_reports_zero_offload(line);
        let has_successful_kernel_launch =
            line.contains("hiplaunchkernel") && line.contains("hipsuccess");
        let has_runtime_init = line.contains("hip runtime initialized")
            || (line.contains("rocm") && line.contains("device") && line.contains("found"));
        let has_rocm_device_line =
            (line.contains(" - rocm") || line.contains("rocm0")) && line.contains("amd ");
        let has_rocm_system_info = line.contains("system_info") && line.contains(" rocm ");
        has_loaded_blas
            || has_backend_init
            || has_positive_offload
            || has_successful_kernel_launch
            || has_runtime_init
            || has_rocm_device_line
            || has_rocm_system_info
    })
}

fn bootstrap_gpu_startup_error(
    reason: &str,
    source: &str,
    windows_driver_guidance: bool,
) -> String {
    let mut message = reason.to_owned();
    if !source.trim().is_empty() {
        let _ = write!(message, "\n\n{}", source.trim());
    }
    if windows_driver_guidance {
        let _ = write!(
            message,
            "\n\nI found an AMD GPU, but the embedded assistant could not prove ROCm/HIP GPU runtime startup.\nInstall or update AMD Software from:\n{AMD_DRIVER_DOWNLOAD_URL}\n\nAfter the driver install finishes, restart this terminal and run `rocm` again."
        );
    }
    message
}

fn bootstrap_should_show_windows_driver_guidance(host_is_windows: bool, text: &str) -> bool {
    if !host_is_windows {
        return false;
    }
    let lowered = text.to_ascii_lowercase();
    let runtime_component = [
        "accelerator runtime",
        "amd driver",
        "amd software",
        "amdhip",
        "comgr",
        "driver",
        "failed to initialize rocm",
        "failed to load gpu",
        "gpu support could not",
        "gpu support couldn't",
        "hip runtime",
        "hiperror",
        "hipgetdevice",
        "hsa runtime",
        "hsa-runtime",
        "no rocm-capable device",
        "rocblas",
        "rocm error",
        "rocm runtime",
        "support for --gpu",
    ]
    .iter()
    .any(|token| lowered.contains(token));
    let failure_language = [
        "could not",
        "couldn't",
        "did not",
        "error",
        "exited",
        "failed",
        "invalid",
        "missing",
        "not available",
        "refused",
        "unable",
        "without",
    ]
    .iter()
    .any(|token| lowered.contains(token));
    runtime_component && failure_language
}

#[allow(dead_code)]
fn validate_bootstrap_gpu_log(
    log_path: &PathBuf,
    plan: &BootstrapAssistantStartPlan,
) -> Result<()> {
    let text = read_recent_log_text(log_path, 256)?;
    let source = bootstrap_log_context_from_text(log_path, &text, 12);
    validate_bootstrap_gpu_log_text_for_host(
        &text,
        &source,
        plan.doctor_gpu.host_os.eq_ignore_ascii_case("windows"),
    )
}

#[allow(dead_code)]
fn read_recent_log_text(path: &PathBuf, max_lines: usize) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = VecDeque::new();
    for line in reader.lines() {
        let line = line?;
        lines.push_back(line);
        while lines.len() > max_lines {
            lines.pop_front();
        }
    }
    Ok(lines.into_iter().collect::<Vec<_>>().join("\n"))
}

#[allow(dead_code)]
fn bootstrap_log_context(path: &PathBuf, max_lines: usize) -> Option<String> {
    let text = read_recent_log_text(path, max_lines).ok()?;
    Some(bootstrap_log_context_from_text(path, &text, max_lines))
}

#[allow(dead_code)]
fn bootstrap_log_context_from_text(path: &PathBuf, text: &str, max_lines: usize) -> String {
    let recent = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(max_lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    if recent.trim().is_empty() {
        format!("Log: {}", path.display())
    } else {
        format!("Log: {}\nRecent log context:\n{}", path.display(), recent)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct BootstrapAssistantStartOptions {
    pub llamafile: Option<PathBuf>,
    pub host: String,
    pub port: u16,
    pub model_candidate: String,
    pub device: String,
    pub allow_public_bind: bool,
    pub allow_cpu_fallback: bool,
}

impl From<BootstrapAssistantArgs> for BootstrapAssistantStartOptions {
    fn from(args: BootstrapAssistantArgs) -> Self {
        Self {
            llamafile: args.llamafile,
            host: args.host,
            port: args.port,
            model_candidate: args.model_candidate,
            device: args.device,
            allow_public_bind: args.allow_public_bind,
            allow_cpu_fallback: args.allow_cpu_fallback,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub(crate) struct BootstrapDoctorGpuSummary {
    pub host_os: String,
    pub detected_gfx_target: Option<String>,
    pub compatible_therock_family: String,
    pub driver_status: String,
    pub driver_detail: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub(crate) struct BootstrapAssistantStartPlan {
    pub status: String,
    pub mode: String,
    pub doctor_gpu: BootstrapDoctorGpuSummary,
    pub host: String,
    pub port: u16,
    pub endpoint_url: String,
    pub server_args: Vec<String>,
    pub device_policy: String,
    pub cpu_fallback_policy: String,
    pub model_candidate: String,
    pub model_validation: String,
    pub llamafile: Option<String>,
    pub tool_facade: String,
    pub tool_schema: String,
    pub tool_policy: String,
}

pub(crate) fn validate_bootstrap_assistant_start(
    doctor: &DoctorSummary,
    options: BootstrapAssistantStartOptions,
) -> Result<BootstrapAssistantStartPlan> {
    let doctor_gpu = require_supported_amd_gpu(doctor)?;
    let host = validate_loopback_host(&options.host, options.allow_public_bind)?;
    validate_gpu_only_device_policy(&options.device, options.allow_cpu_fallback)?;
    let model_candidate = options.model_candidate.trim();
    if model_candidate.is_empty() {
        bail!("bootstrap assistant model candidate must not be empty");
    }
    if let Some(path) = options.llamafile.as_ref()
        && !path.is_file()
    {
        bail!(
            "bootstrap assistant llamafile `{}` was not found; model contents were not validated",
            path.display()
        );
    }

    Ok(BootstrapAssistantStartPlan {
        status: "allowed_to_start".to_owned(),
        mode: "foreground_server".to_owned(),
        doctor_gpu,
        host: host.clone(),
        port: options.port,
        endpoint_url: format!("{}/v1", format_http_base_url(&host, options.port)),
        server_args: vec![
            "--server".to_owned(),
            "--host".to_owned(),
            host,
            "--port".to_owned(),
            options.port.to_string(),
            "--jinja".to_owned(),
            "--gpu".to_owned(),
            "amd".to_owned(),
            "-ngl".to_owned(),
            "999".to_owned(),
            "-lv".to_owned(),
            "0".to_owned(),
        ],
        device_policy: BOOTSTRAP_DEVICE_POLICY.to_owned(),
        cpu_fallback_policy: "rejected_by_rocm_cli".to_owned(),
        model_candidate: model_candidate.to_owned(),
        model_validation: "not_validated_candidate_only".to_owned(),
        llamafile: options
            .llamafile
            .as_ref()
            .map(|path| path.display().to_string()),
        tool_facade: BOOTSTRAP_TOOL_FACADE.to_owned(),
        tool_schema: providers::ROCM_TOOL_SCHEMA_ID.to_owned(),
        tool_policy:
            "argv_style_rocm_tools_only; read_only_runs_directly; mutating_requires_approval"
                .to_owned(),
    })
}

fn require_supported_amd_gpu(doctor: &DoctorSummary) -> Result<BootstrapDoctorGpuSummary> {
    let Some(family) = doctor
        .compatible_therock_family
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        bail!(
            "no supported AMD GPU detected by rocm doctor; detected_gfx_target={}",
            doctor.detected_gfx_target.as_deref().unwrap_or("<unknown>")
        );
    };
    if doctor.os.eq_ignore_ascii_case("windows")
        && doctor.driver.status.trim() != "amd_display_driver_detected"
    {
        bail!(
            "I found an AMD GPU ({gfx}), but Windows is not reporting the AMD display driver/runtime yet.\n\nInstall the latest AMD Software driver from:\n{AMD_DRIVER_DOWNLOAD_URL}\n\nAfter the driver install finishes, restart this terminal and run `rocm` again.",
            gfx = doctor.detected_gfx_target.as_deref().unwrap_or("<unknown>")
        );
    }
    Ok(BootstrapDoctorGpuSummary {
        host_os: doctor.os.clone(),
        detected_gfx_target: doctor.detected_gfx_target.clone(),
        compatible_therock_family: family.to_owned(),
        driver_status: doctor.driver.status.clone(),
        driver_detail: doctor.driver.detail.clone(),
    })
}

fn validate_loopback_host(host: &str, allow_public_bind: bool) -> Result<String> {
    if allow_public_bind {
        bail!("bootstrap assistant cannot enable public binding; loopback binding is required");
    }
    let host = host.trim();
    if host.is_empty() {
        bail!("bootstrap assistant host must not be empty");
    }
    if !is_loopback_host(host) {
        bail!("bootstrap assistant must bind to loopback only; rejected host `{host}`");
    }
    Ok(host.to_owned())
}

fn validate_gpu_only_device_policy(device: &str, allow_cpu_fallback: bool) -> Result<()> {
    if allow_cpu_fallback {
        bail!("bootstrap assistant CPU fallback is disabled");
    }
    let device = device.trim().to_ascii_lowercase();
    if device != BOOTSTRAP_DEVICE_POLICY {
        bail!(
            "bootstrap assistant requires `{BOOTSTRAP_DEVICE_POLICY}`; CPU fallback and preferred GPU fallback modes are rejected"
        );
    }
    Ok(())
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum BootstrapToolAccess {
    ReadOnly {
        tool: String,
    },
    ApprovalRequired {
        tool: String,
        pending_title: String,
        command_title: String,
        args: Vec<String>,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn classify_bootstrap_tool_call(
    call: &providers::ChatToolCall,
) -> Result<BootstrapToolAccess> {
    ensure_bootstrap_tool_facade_name(&call.name)?;
    crate::validate_chat_tool_call(call)?;
    if crate::chat_tool_call_is_read_only(call) {
        return Ok(BootstrapToolAccess::ReadOnly {
            tool: call.name.clone(),
        });
    }
    let approval = crate::chat_tool_approval_request(call, None)?;
    Ok(BootstrapToolAccess::ApprovalRequired {
        tool: call.name.clone(),
        pending_title: approval.pending_title,
        command_title: approval.command_title,
        args: approval.args,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn ensure_bootstrap_tool_facade_name(name: &str) -> Result<()> {
    match name {
        "doctor"
        | "bridge_snapshot"
        | "gpu_snapshot"
        | "path_exists"
        | "engines"
        | "services"
        | "service_logs"
        | "automations"
        | "rocm_command"
        | "update_check"
        | "install_sdk_dry_run"
        | "install_sdk"
        | "install_engine"
        | "launch_server"
        | "stop_server"
        | "watcher_enable"
        | "watcher_disable" => Ok(()),
        "natural_language_plan" => bail!(
            "bootstrap assistant does not accept free-form command planning; use argv-style rocm tools"
        ),
        other => {
            bail!(
                "bootstrap assistant can only use the rocm-cli tool facade; unsupported tool `{other}`"
            )
        }
    }
}

fn render_bootstrap_assistant_plan(plan: &BootstrapAssistantStartPlan) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "bootstrap assistant validation");
    let _ = writeln!(output, "  status: {}", plan.status);
    let _ = writeln!(output, "  mode: {}", plan.mode);
    let _ = writeln!(
        output,
        "  gpu: {} ({})",
        plan.doctor_gpu
            .detected_gfx_target
            .as_deref()
            .unwrap_or("<unknown>"),
        plan.doctor_gpu.compatible_therock_family
    );
    let _ = writeln!(output, "  driver_status: {}", plan.doctor_gpu.driver_status);
    let _ = writeln!(output, "  bind: {}:{}", plan.host, plan.port);
    let _ = writeln!(output, "  endpoint: {}", plan.endpoint_url);
    let _ = writeln!(output, "  device_policy: {}", plan.device_policy);
    let _ = writeln!(
        output,
        "  cpu_fallback_policy: {}",
        plan.cpu_fallback_policy
    );
    let _ = writeln!(output, "  model_candidate: {}", plan.model_candidate);
    let _ = writeln!(output, "  model_validation: {}", plan.model_validation);
    let _ = writeln!(
        output,
        "  llamafile: {}",
        plan.llamafile
            .as_deref()
            .unwrap_or("<candidate not provided>")
    );
    let _ = writeln!(output, "  server_args: {}", plan.server_args.join(" "));
    let _ = writeln!(output, "  tool_facade: {}", plan.tool_facade);
    let _ = writeln!(output, "  tool_schema: {}", plan.tool_schema);
    let _ = writeln!(output, "  tool_policy: {}", plan.tool_policy);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::{DriverSummary, LegacyRocmSummary, unix_time_millis};
    use serde_json::json;
    use std::fs;

    fn supported_doctor() -> DoctorSummary {
        DoctorSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("6.8.0-test".to_owned()),
            distro: Some("Ubuntu test".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "pytorch".to_owned(),
            detected_gfx_target: Some("gfx1201".to_owned()),
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: Some("/dev/kfd present".to_owned()),
            },
            legacy_rocm: LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("/tmp/config"),
            data_dir: PathBuf::from("/tmp/data"),
            cache_dir: PathBuf::from("/tmp/cache"),
        }
    }

    fn start_options() -> BootstrapAssistantStartOptions {
        BootstrapAssistantStartOptions {
            llamafile: None,
            host: DEFAULT_LOCAL_HOST.to_owned(),
            port: DEFAULT_BOOTSTRAP_ASSISTANT_PORT,
            model_candidate: DEFAULT_BOOTSTRAP_MODEL_CANDIDATE.to_owned(),
            device: BOOTSTRAP_DEVICE_POLICY.to_owned(),
            allow_public_bind: false,
            allow_cpu_fallback: false,
        }
    }

    #[test]
    fn no_supported_gpu_hard_blocks_bootstrap_assistant_start() {
        let mut doctor = supported_doctor();
        doctor.detected_gfx_target = None;
        doctor.compatible_therock_family = None;

        let error = validate_bootstrap_assistant_start(&doctor, start_options()).unwrap_err();

        assert!(error.to_string().contains("no supported AMD GPU"));
    }

    #[test]
    fn windows_gpu_without_driver_points_to_amd_driver_download() {
        let mut doctor = supported_doctor();
        doctor.os = "windows".to_owned();
        doctor.driver = DriverSummary {
            policy: "windows_validate_only".to_owned(),
            status: "not_detected".to_owned(),
            detail: None,
        };

        let error = validate_bootstrap_assistant_start(&doctor, start_options()).unwrap_err();
        let error = error.to_string();

        assert!(error.contains("I found an AMD GPU"), "{error}");
        assert!(error.contains(AMD_DRIVER_DOWNLOAD_URL), "{error}");
        assert!(
            error.contains("restart this terminal and run `rocm` again"),
            "{error}"
        );
    }

    #[test]
    fn bootstrap_assistant_uses_loopback_binding_only() -> Result<()> {
        let plan = validate_bootstrap_assistant_start(&supported_doctor(), start_options())?;

        assert_eq!(plan.host, "127.0.0.1");
        assert_eq!(
            plan.server_args,
            vec![
                "--server",
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "--jinja",
                "--gpu",
                "amd",
                "-ngl",
                "999",
                "-lv",
                "0"
            ]
        );
        assert!(plan.endpoint_url.starts_with("http://127.0.0.1:"));

        let mut public = start_options();
        public.host = "0.0.0.0".to_owned();
        let error = validate_bootstrap_assistant_start(&supported_doctor(), public).unwrap_err();
        assert!(error.to_string().contains("loopback only"));

        let mut allow_public = start_options();
        allow_public.allow_public_bind = true;
        let error =
            validate_bootstrap_assistant_start(&supported_doctor(), allow_public).unwrap_err();
        assert!(error.to_string().contains("public binding"));
        Ok(())
    }

    #[test]
    fn bootstrap_assistant_rejects_cpu_fallback() {
        let mut cpu_fallback = start_options();
        cpu_fallback.allow_cpu_fallback = true;
        let error =
            validate_bootstrap_assistant_start(&supported_doctor(), cpu_fallback).unwrap_err();
        assert!(error.to_string().contains("CPU fallback is disabled"));

        let mut cpu_device = start_options();
        cpu_device.device = "cpu".to_owned();
        let error =
            validate_bootstrap_assistant_start(&supported_doctor(), cpu_device).unwrap_err();
        assert!(error.to_string().contains("gpu_required"));

        let mut preferred = start_options();
        preferred.device = "gpu_preferred".to_owned();
        let error = validate_bootstrap_assistant_start(&supported_doctor(), preferred).unwrap_err();
        assert!(error.to_string().contains("fallback modes are rejected"));
    }

    #[test]
    fn bootstrap_assistant_marks_model_as_candidate_not_validated() -> Result<()> {
        let plan = validate_bootstrap_assistant_start(&supported_doctor(), start_options())?;

        assert_eq!(plan.model_candidate, DEFAULT_BOOTSTRAP_MODEL_CANDIDATE);
        assert_eq!(plan.model_validation, "not_validated_candidate_only");
        assert_eq!(plan.mode, "foreground_server");
        Ok(())
    }

    #[test]
    fn bootstrap_gpu_log_requires_rocm_runtime_evidence() -> Result<()> {
        validate_bootstrap_gpu_log_text(
            "AMD_LOG_LEVEL=0\nHIP runtime initialized\nrocBLAS loaded\nllama server listening",
            "test-log",
        )?;
        validate_bootstrap_gpu_log_text(
            "0.00 I   - ROCm0   : AMD Radeon RX 9070 XT (16304 MiB, 16153 MiB free)",
            "test-log",
        )?;
        validate_bootstrap_gpu_log_text(
            "hip_module.cpp : hipLaunchKernel: Returned hipSuccess : : duration: 7 us",
            "test-log",
        )?;

        let fallback = validate_bootstrap_gpu_log_text(
            "GPU support could not be loaded; falling back to CPU",
            "test-log",
        )
        .unwrap_err();
        assert!(fallback.to_string().contains("CPU fallback"));

        let zero_offload = validate_bootstrap_gpu_log_text(
            "llama_model_load: offloaded 0/99 layers to GPU",
            "test-log",
        )
        .unwrap_err();
        assert!(zero_offload.to_string().contains("CPU fallback"));

        let missing_code_object = validate_bootstrap_gpu_log_text(
            "ggml_cuda_init: found 1 ROCm devices\nNo compatible code objects found with HIP_FORCE_SPIRV_CODEOBJECT=0. Rebuild the application with option --offload-arch=gfx1201\nROCm error: device kernel image is invalid",
            "test-log",
        )
        .unwrap_err();
        assert!(
            missing_code_object.to_string().contains("CPU fallback"),
            "{missing_code_object}"
        );

        let missing = validate_bootstrap_gpu_log_text("server listening", "test-log").unwrap_err();
        assert!(missing.to_string().contains("did not prove AMD GPU"));

        let windows_runtime_missing = validate_bootstrap_gpu_log_text_for_host(
            "server listening without accelerator runtime",
            "test-log",
            true,
        )
        .unwrap_err();
        let windows_runtime_missing = windows_runtime_missing.to_string();
        assert!(windows_runtime_missing.contains(AMD_DRIVER_DOWNLOAD_URL));
        assert!(windows_runtime_missing.contains("restart this terminal and run `rocm` again"));
        Ok(())
    }

    #[test]
    fn bootstrap_path_ignores_active_venv_and_therock_entries() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-bootstrap-path-isolation-{}",
            unix_time_millis()
        ));
        let payload = root.join("payload");
        let venv = root.join("venv");
        let venv_bin = venv.join(if cfg!(windows) { "Scripts" } else { "bin" });
        let path_only_venv_bin =
            root.join("path-only-venv")
                .join(if cfg!(windows) { "Scripts" } else { "bin" });
        let therock_bin = root.join("therock_venvs").join("bin");
        let rocm_bin = root.join("ROCm").join("bin");
        let safe_bin = root.join("safe-bin");
        let existing_path = std::env::join_paths([
            &venv_bin,
            &path_only_venv_bin,
            &therock_bin,
            &rocm_bin,
            &safe_bin,
        ])?;

        let entries = bootstrap_path_entries(&payload, Some(existing_path.as_os_str()), &[venv]);

        assert_eq!(entries.first(), Some(&payload));
        assert!(entries.iter().any(|entry| path_eq(entry, &safe_bin)));
        assert!(!entries.iter().any(|entry| path_eq(entry, &venv_bin)));
        assert!(
            !entries
                .iter()
                .any(|entry| path_eq(entry, &path_only_venv_bin))
        );
        assert!(!entries.iter().any(|entry| path_eq(entry, &therock_bin)));
        assert!(!entries.iter().any(|entry| path_eq(entry, &rocm_bin)));
        Ok(())
    }

    #[test]
    fn windows_bootstrap_runtime_failures_point_to_driver_download() {
        let message = bootstrap_gpu_startup_error(
            "bootstrap assistant exited before it became ready (exit code 7)",
            "Log: test-log\nRecent log context:\nhip runtime failed",
            bootstrap_should_show_windows_driver_guidance(true, "hip runtime failed"),
        );

        assert!(message.contains(AMD_DRIVER_DOWNLOAD_URL));
        assert!(message.contains("restart this terminal and run `rocm` again"));
    }

    #[test]
    fn windows_bootstrap_unrelated_failures_do_not_point_to_driver_download() {
        let message = bootstrap_gpu_startup_error(
            "bootstrap assistant exited before it became ready (exit code 7)",
            "Log: test-log\nRecent log context:\nport is already in use",
            bootstrap_should_show_windows_driver_guidance(true, "port is already in use"),
        );

        assert!(!message.contains(AMD_DRIVER_DOWNLOAD_URL));
    }

    #[test]
    fn bootstrap_removed_environment_covers_python_and_rocm_runtime_leaks() {
        for required in [
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "ROCM_PATH",
            "ROCM_SDK_ROOT",
            "HIP_PATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        ] {
            assert!(
                BOOTSTRAP_ENV_REMOVE.contains(&required),
                "{required} must be stripped before bootstrap assistant launch"
            );
        }
    }

    #[test]
    fn bootstrap_llamafile_path_is_executable_on_windows() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-bootstrap-llamafile-copy-{}",
            unix_time_millis()
        ));
        fs::create_dir_all(&root)?;
        let llamafile = root.join(DEFAULT_BOOTSTRAP_MODEL_CANDIDATE);
        fs::write(&llamafile, b"fake llamafile")?;

        let executable = executable_llamafile_path(&llamafile)?;

        if cfg!(windows) {
            assert_eq!(
                executable.file_name().and_then(|value| value.to_str()),
                Some("Qwen3.5-0.8B-Q8_0.llamafile.exe")
            );
            assert!(executable.is_file());
        } else {
            assert_eq!(executable, llamafile);
        }
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn bootstrap_llamafile_ape_magic_detection_matches_cosmopolitan_headers() {
        assert!(llamafile_ape_magic_matches(b"MZqFpD"));
        assert!(llamafile_ape_magic_matches(b"jartsr"));
        assert!(!llamafile_ape_magic_matches(b"\x7fELF\0\0"));
    }

    #[test]
    fn linux_bootstrap_llamafile_ape_uses_bundled_loader() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-bootstrap-llamafile-ape-loader-{}",
            unix_time_millis()
        ));
        fs::create_dir_all(&root)?;
        let llamafile = root.join(DEFAULT_BOOTSTRAP_MODEL_CANDIDATE);
        fs::write(&llamafile, b"MZqFpDfake ape")?;
        let loader = root.join("ape-x86_64.elf");
        fs::write(&loader, b"fake loader")?;

        let launch = bootstrap_llamafile_launch(&llamafile)?;

        if cfg!(target_os = "linux") {
            assert_eq!(launch.program, loader);
            assert_eq!(
                launch.leading_args,
                vec![llamafile.as_os_str().to_os_string()]
            );
        } else if cfg!(windows) {
            assert_eq!(
                launch.program.file_name().and_then(|value| value.to_str()),
                Some("Qwen3.5-0.8B-Q8_0.llamafile.exe")
            );
            assert!(launch.leading_args.is_empty());
        } else {
            assert_eq!(launch.program, llamafile);
            assert!(launch.leading_args.is_empty());
        }
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    fn write_fake_bootstrap_cli_source(source_bin: &Path) -> Result<()> {
        fs::create_dir_all(source_bin)?;
        fs::write(
            source_bin.join(bootstrap_cli_binary_name("rocm")),
            b"fake rocm",
        )?;
        fs::write(
            source_bin.join(bootstrap_cli_binary_name("rocmd")),
            b"fake rocmd",
        )?;
        fs::write(
            source_bin.join(bootstrap_cli_binary_name("rocm-engine-pytorch")),
            b"fake engine",
        )?;
        Ok(())
    }

    #[test]
    fn bootstrap_cli_install_copies_extracted_bin_to_selected_folder() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-bootstrap-cli-install-{}", unix_time_millis()));
        let source_bin = root.join("payload").join("bin");
        let target = root.join("chosen-install-folder");
        write_fake_bootstrap_cli_source(&source_bin)?;

        let result = bootstrap_cli_install(BootstrapInstallCliArgs {
            target: target.clone(),
            source_bin: Some(source_bin),
            add_to_path: false,
            dry_run: false,
        })?;

        assert_eq!(result.path_status, BootstrapCliPathStatus::NotRequested);
        assert!(target.join(bootstrap_cli_binary_name("rocm")).is_file());
        assert!(target.join(bootstrap_cli_binary_name("rocmd")).is_file());
        assert!(result.manifest_path.is_file());
        let manifest = fs::read_to_string(result.manifest_path)?;
        assert!(manifest.contains(&bootstrap_cli_binary_name("rocm")));
        assert!(manifest.contains(&bootstrap_cli_binary_name("rocmd")));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn bootstrap_cli_install_removes_previous_manifest_entries_inside_target_only() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-bootstrap-cli-reinstall-{}",
            unix_time_millis()
        ));
        let source_bin = root.join("payload").join("bin");
        let target = root.join("chosen-install-folder");
        let outside = root.join("outside").join("do-not-remove.txt");
        write_fake_bootstrap_cli_source(&source_bin)?;
        fs::create_dir_all(&target)?;
        fs::create_dir_all(outside.parent().unwrap())?;
        let old_inside = target.join("old-rocm.exe");
        fs::write(&old_inside, b"old")?;
        fs::write(&outside, b"outside")?;
        fs::write(
            target.join(BOOTSTRAP_CLI_INSTALL_MANIFEST),
            format!("{}\n{}\n", old_inside.display(), outside.display()),
        )?;

        bootstrap_cli_install(BootstrapInstallCliArgs {
            target: target.clone(),
            source_bin: Some(source_bin),
            add_to_path: false,
            dry_run: false,
        })?;

        assert!(!old_inside.exists());
        assert!(outside.exists());
        assert!(target.join(bootstrap_cli_binary_name("rocm")).is_file());

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn bootstrap_cli_install_dry_run_does_not_touch_target_or_path() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("rocm-bootstrap-cli-dry-run-{}", unix_time_millis()));
        let source_bin = root.join("payload").join("bin");
        let target = root.join("chosen-install-folder");
        write_fake_bootstrap_cli_source(&source_bin)?;

        let result = bootstrap_cli_install(BootstrapInstallCliArgs {
            target: target.clone(),
            source_bin: Some(source_bin),
            add_to_path: true,
            dry_run: true,
        })?;

        assert!(result.dry_run);
        assert_eq!(result.path_status, BootstrapCliPathStatus::WouldUpdate);
        assert!(!target.exists());

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn bootstrap_cli_install_rejects_temporary_source_folder_as_target() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "rocm-bootstrap-cli-source-target-{}",
            unix_time_millis()
        ));
        let source_bin = root.join("payload").join("bin");
        write_fake_bootstrap_cli_source(&source_bin)?;

        let error = bootstrap_cli_install(BootstrapInstallCliArgs {
            target: source_bin.clone(),
            source_bin: Some(source_bin),
            add_to_path: false,
            dry_run: false,
        })
        .unwrap_err();

        assert!(error.to_string().contains("permanent install folder"));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn bootstrap_tool_mutations_require_existing_approval_facade() -> Result<()> {
        let call = providers::ChatToolCall {
            id: Some("install-sdk".to_owned()),
            name: "install_sdk".to_owned(),
            arguments: json!({
                "channel": "release",
                "format": "pip",
                "prefix": "D:\\jam\\temp\\therock_venvs",
                "build_date": "2026-05-13"
            }),
        };

        let access = classify_bootstrap_tool_call(&call)?;

        assert_eq!(
            access,
            BootstrapToolAccess::ApprovalRequired {
                tool: "install_sdk".to_owned(),
                pending_title: "Install ROCm".to_owned(),
                command_title: "Install".to_owned(),
                args: vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "pip".to_owned(),
                    "--prefix".to_owned(),
                    "D:\\jam\\temp\\therock_venvs".to_owned(),
                    "--build-date".to_owned(),
                    "2026-05-13".to_owned(),
                ],
            }
        );
        Ok(())
    }

    #[test]
    fn bootstrap_tool_allows_read_only_argv_facade_call() -> Result<()> {
        let call = providers::ChatToolCall {
            id: Some("model".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: json!({ "args": ["model"] }),
        };

        let access = classify_bootstrap_tool_call(&call)?;

        assert_eq!(
            access,
            BootstrapToolAccess::ReadOnly {
                tool: "rocm_command".to_owned()
            }
        );
        Ok(())
    }

    #[test]
    fn bootstrap_tool_allows_read_only_path_exists_call() -> Result<()> {
        let call = providers::ChatToolCall {
            id: Some("path-check".to_owned()),
            name: "path_exists".to_owned(),
            arguments: json!({ "path": "D:\\jam\\temp\\therock_venvs" }),
        };

        let access = classify_bootstrap_tool_call(&call)?;

        assert_eq!(
            access,
            BootstrapToolAccess::ReadOnly {
                tool: "path_exists".to_owned()
            }
        );
        Ok(())
    }

    #[test]
    fn bootstrap_tool_rejects_shell_and_package_manager_text() {
        for args in [
            json!({ "args": ["powershell", "-Command", "Get-ChildItem"] }),
            json!({ "args": ["cmd", "/C", "dir"] }),
            json!({ "args": ["apt-get", "install", "amdgpu-dkms"] }),
        ] {
            let call = providers::ChatToolCall {
                id: None,
                name: "rocm_command".to_owned(),
                arguments: args,
            };

            let error = classify_bootstrap_tool_call(&call).unwrap_err();
            assert!(
                error.to_string().contains("unsupported rocm command")
                    || error.to_string().contains("approval UI"),
                "{error}"
            );
        }

        let free_form = providers::ChatToolCall {
            id: None,
            name: "natural_language_plan".to_owned(),
            arguments: json!({ "request": "run sudo apt-get install amdgpu-dkms" }),
        };
        let error = classify_bootstrap_tool_call(&free_form).unwrap_err();
        assert!(error.to_string().contains("free-form command planning"));
    }
}
