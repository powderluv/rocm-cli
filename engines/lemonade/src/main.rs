use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rocm_core::{AppPaths, DEFAULT_LOCAL_PORT, format_http_base_url, require_nonempty};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest,
    EndpointResponse, EngineCapabilities, EngineDeviceAvailability, EngineMethod, EngineRecipeHint,
    EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest, HealthcheckResponse,
    InstallRequest, InstallResponse, LaunchRequest, LaunchResponse, LogsRequest, LogsResponse,
    ResolveModelRequest, ResolveModelResponse, StopRequest, StopResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "lemonade";
const LEMONADE_VERSION: &str = "10.6.0";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_MODEL: &str = "Qwen3-0.6B-GGUF";
const ROCM_BACKEND_RECIPE: &str = "llamacpp";
const ROCM_BACKEND_NAME: &str = "rocm";
const DEFAULT_LOG_TAIL_LINES: usize = 200;

#[cfg(windows)]
const EMBEDDABLE_ARCHIVE_NAME: &str = "lemonade-embeddable-10.6.0-windows-x64.zip";
#[cfg(not(windows))]
const EMBEDDABLE_ARCHIVE_NAME: &str = "lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz";

#[cfg(windows)]
const EMBEDDABLE_URL: &str = "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-windows-x64.zip";
#[cfg(not(windows))]
const EMBEDDABLE_URL: &str = "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz";

#[derive(Parser)]
#[command(name = "rocm-engine-lemonade")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long)]
        runtime_id: String,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
    },
    Launch {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
    },
    Stdio,
    ServeHttp {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        log_path: Option<PathBuf>,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LemonadeInstallManifest {
    env_id: String,
    version: String,
    runtime_dir: PathBuf,
    lemond: PathBuf,
    lemonade: PathBuf,
    backend_recipe: String,
    backend_name: String,
    installed_at_unix_ms: u128,
}

#[derive(Debug, Clone)]
struct LemonadeRuntime {
    manifest: LemonadeInstallManifest,
}

#[derive(Debug, Clone)]
struct ServeHttpRequest {
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
}

#[derive(Debug, Clone)]
struct ServiceFiles {
    state_path: PathBuf,
    log_path: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response())?,
        CommandKind::Capabilities => print_json(&capabilities())?,
        CommandKind::Install {
            runtime_id,
            reinstall,
        } => print_json(&install_response(InstallRequest {
            runtime_id,
            python_version: None,
            reinstall,
        })?)?,
        CommandKind::ResolveModel { model_ref } => {
            print_json(&resolve_model_response(ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: None,
                recipe_override: None,
                engine_recipe: None,
            })?)?
        }
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
        } => print_json(&launch_service(LaunchRequest {
            service_id,
            env_id,
            runtime_id,
            model_ref,
            host,
            port,
            device_policy: Some(parse_device_policy_arg(device_policy.as_deref())?),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
        })?)?,
        CommandKind::Stdio => {
            let envelope = read_request()?;
            print_json(&handle_envelope(envelope))?;
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe_json,
        } => serve_http(ServeHttpRequest {
            service_id,
            model_ref,
            host,
            port,
            device_policy: parse_device_policy_arg(device_policy.as_deref())?,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe: parse_engine_recipe_json(engine_recipe_json)?,
        })?,
    }
    Ok(())
}

fn handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    match envelope.method {
        EngineMethod::Detect => {
            deserialize_and_respond::<DetectRequest, _, _>(envelope.payload, |_| {
                Ok(detect_response())
            })
        }
        EngineMethod::Capabilities => EngineResponseEnvelope::success(capabilities()),
        EngineMethod::Install => {
            deserialize_and_respond::<InstallRequest, _, _>(envelope.payload, install_response)
        }
        EngineMethod::ResolveModel => deserialize_and_respond::<ResolveModelRequest, _, _>(
            envelope.payload,
            resolve_model_response,
        ),
        EngineMethod::Launch => {
            deserialize_and_respond::<LaunchRequest, _, _>(envelope.payload, launch_service)
        }
        EngineMethod::Healthcheck => deserialize_and_respond::<HealthcheckRequest, _, _>(
            envelope.payload,
            healthcheck_service,
        ),
        EngineMethod::Endpoint => {
            deserialize_and_respond::<EndpointRequest, _, _>(envelope.payload, endpoint_response)
        }
        EngineMethod::Stop => {
            deserialize_and_respond::<StopRequest, _, _>(envelope.payload, stop_service)
        }
        EngineMethod::Logs => {
            deserialize_and_respond::<LogsRequest, _, _>(envelope.payload, logs_response)
        }
    }
}

fn deserialize_and_respond<T, F, U>(payload: Value, handler: F) -> EngineResponseEnvelope
where
    T: for<'de> Deserialize<'de>,
    F: FnOnce(T) -> Result<U>,
    U: Serialize,
{
    match serde_json::from_value::<T>(payload) {
        Ok(request) => match handler(request) {
            Ok(response) => EngineResponseEnvelope::success(response),
            Err(error) => EngineResponseEnvelope::failure("request_failed", error.to_string()),
        },
        Err(error) => EngineResponseEnvelope::failure("invalid_payload", error.to_string()),
    }
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: true,
        multi_gpu: true,
        openai_compatible: true,
        tool_calling: true,
        quantized_models: "GGUF through Lemonade llamacpp:rocm".to_owned(),
        distributed_serving: false,
        reasoning_parser: false,
    }
}

fn detect_response() -> DetectResponse {
    let runtime = resolve_runtime().ok();
    let mut notes = Vec::new();
    if let Some(runtime) = runtime.as_ref() {
        notes.push(format!(
            "Lemonade embeddable {} is installed at {}",
            runtime.manifest.version,
            runtime.manifest.runtime_dir.display()
        ));
        notes.push("Lemonade is configured for llamacpp:rocm; no CPU fallback is used".to_owned());
    } else {
        notes.push(
            "Lemonade embeddable is not installed yet; run `rocm engines install lemonade`"
                .to_owned(),
        );
    }
    DetectResponse {
        installed: runtime.is_some(),
        env_id: runtime
            .as_ref()
            .map(|runtime| runtime.manifest.env_id.clone()),
        runtime_kind: Some("lemonade_embeddable".to_owned()),
        runtime_executable: runtime
            .as_ref()
            .map(|runtime| runtime.manifest.lemond.display().to_string()),
        managed_env: Some(true),
        python_version: None,
        torch_version: None,
        transformers_version: None,
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: runtime.is_some(),
            reason: if runtime.is_some() {
                None
            } else {
                Some("Lemonade embeddable runtime is not installed".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    eprintln!("Preparing Lemonade embeddable {LEMONADE_VERSION}...");
    let manifest = prepare_embeddable(&paths, request.reinstall)?;
    eprintln!("Checking Lemonade ROCm backend support...");
    install_rocm_backend(&manifest)?;
    write_manifest(&paths, &manifest)?;
    Ok(InstallResponse {
        env_id: manifest.env_id.clone(),
        env_path: manifest.runtime_dir.display().to_string(),
        python_executable: manifest.lemonade.display().to_string(),
        runtime_kind: Some("lemonade_embeddable".to_owned()),
        runtime_executable: Some(manifest.lemond.display().to_string()),
        managed_env: Some(true),
        installed_packages: vec![
            format!("lemonade-embeddable=={}", manifest.version),
            "lemonade-backend=llamacpp:rocm".to_owned(),
        ],
        capabilities: capabilities(),
        lock_hash: manifest_lock_hash(&manifest),
        warnings: vec![
            "Lemonade is installed as a rocm-cli managed embeddable runtime".to_owned(),
            "Only the ROCm GPU backend is accepted; no CPU fallback is used".to_owned(),
        ],
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    let device_policy = normalize_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let canonical_model_id = resolve_lemonade_model_ref(&request.model_ref);
    Ok(ResolveModelResponse {
        canonical_model_id,
        task: "chat-completions".to_owned(),
        source: "lemonade".to_owned(),
        revision: "main".to_owned(),
        loader: "llamacpp:rocm".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "lemonade".to_owned(),
        dtype: "gguf".to_owned(),
        device_policy,
        estimated_memory: "about 1 GiB plus context for Qwen3-0.6B-GGUF".to_owned(),
        launch_defaults: json!({
            "host": DEFAULT_HOST,
            "port": DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings: vec![
            "Lemonade serving is GPU-required in rocm-cli; CPU, Vulkan, and NPU backends are not selected automatically".to_owned(),
        ],
    })
}

fn launch_service(mut request: LaunchRequest) -> Result<LaunchResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    request.device_policy = Some(normalize_device_policy(request.device_policy.clone())?);
    request.engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let runtime = resolve_runtime()?;
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.engine_logs_dir(ENGINE_NAME))?;
    fs::create_dir_all(paths.engine_state_dir(ENGINE_NAME))?;
    let log_path = paths
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{}.log", request.service_id));
    let state_path = paths
        .engine_state_dir(ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let endpoint_url = endpoint_url(&request.host, request.port);
    let serve_request = ServeHttpRequest {
        service_id: request.service_id.clone(),
        model_ref: resolve_lemonade_model_ref(&request.model_ref),
        host: request.host.clone(),
        port: request.port,
        device_policy: request
            .device_policy
            .clone()
            .unwrap_or(DevicePolicy::GpuRequired),
        runtime_id: request.runtime_id.clone(),
        env_id: request.env_id.clone(),
        state_path: state_path.clone(),
        log_path: Some(log_path.clone()),
        engine_recipe: request.engine_recipe.clone(),
    };
    let current_exe =
        std::env::current_exe().context("failed to discover current Lemonade engine binary")?;
    let args = serve_http_command_args(&serve_request);
    write_running_state(
        &serve_request,
        &runtime,
        std::process::id(),
        None,
        "starting",
    )?;
    let wrapper_pid = spawn_serve_http_background(&current_exe, &args)?;
    merge_json_state(
        &state_path,
        &json!({
            "pid": wrapper_pid,
            "wrapper_pid": wrapper_pid,
        }),
    )?;
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid: wrapper_pid,
        endpoint_url,
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

fn serve_http(request: ServeHttpRequest) -> Result<()> {
    require_gpu_required(&request.device_policy)?;
    let runtime = resolve_runtime()?;
    let log_path = request.log_path.as_deref();
    write_running_state(&request, &runtime, std::process::id(), None, "starting")?;
    let mut child = spawn_lemond(&runtime.manifest, &request.host, request.port, log_path)?;
    write_running_state(
        &request,
        &runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    wait_for_health(&request.host, request.port, Duration::from_secs(20))
        .context("Lemonade server did not become ready")?;
    let system_info = query_system_info(&request.host, request.port)?;
    let backend = lemonade_rocm_backend(&system_info)?;
    if !backend.backend_is_ready() {
        bail!(
            "Lemonade ROCm backend is {}; run `rocm engines install lemonade` first. No CPU fallback is used.",
            backend.state
        );
    }
    let load_response = post_load_model_rocm(&request.host, request.port, &request.model_ref)
        .with_context(|| {
            format!(
                "failed to load {} with Lemonade llamacpp:rocm; no CPU or Vulkan fallback is allowed",
                request.model_ref
            )
        })?;
    let loaded_health = wait_for_model_loaded(
        &request.host,
        request.port,
        &request.model_ref,
        Duration::from_secs(900),
    )
    .with_context(|| {
        format!(
            "Lemonade did not report {} as loaded with the ROCm backend",
            request.model_ref
        )
    })?;
    merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            "server_pid": child.id(),
            "backend_state": backend.state,
            "backend_requested": ROCM_BACKEND_NAME,
            "load_response": load_response,
            "loaded_health": loaded_health,
            "system_info": system_info,
        }),
    )?;
    let status = child.wait().context("failed waiting for Lemonade server")?;
    mark_json_status(
        &request.state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        bail!("Lemonade server exited with status {status}")
    }
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let model_ref = state
        .as_ref()
        .and_then(|value| {
            value_string(value, "canonical_model_id").or_else(|| value_string(value, "model_ref"))
        })
        .unwrap_or_default();
    let ready = state_status == "ready"
        && !model_ref.is_empty()
        && endpoint_url
            .as_deref()
            .map(|endpoint| query_loaded_model_endpoint(endpoint, &model_ref))
            .transpose()
            .unwrap_or(None)
            .unwrap_or(false);
    let status = if ready {
        "ready".to_owned()
    } else {
        state_status
    };
    Ok(HealthcheckResponse {
        status,
        model_loaded: ready,
        device: if ready {
            "rocm_gpu".to_owned()
        } else {
            "unknown".to_owned()
        },
        uptime_sec: 0,
        queue_depth: 0,
        last_error: None,
        tokens_per_sec: None,
    })
}

fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path)
        .with_context(|| format!("service state not found for `{}`", request.service_id))?;
    let endpoint_url = endpoint_url_from_state(&state)
        .with_context(|| format!("service `{}` has no endpoint URL", request.service_id))?;
    Ok(EndpointResponse {
        endpoint_url,
        api_style: "openai".to_owned(),
        supported_routes: vec![
            "/v1/health".to_owned(),
            "/v1/models".to_owned(),
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
        ],
    })
}

fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines: if files.log_path.is_file() {
            tail_lines(&files.log_path, limit)?
        } else {
            Vec::new()
        },
    })
}

fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let stopped = match state.as_ref().and_then(pid_to_terminate_from_state) {
        Some(pid) => terminate_pid(pid, request.force),
        None => false,
    };
    if stopped {
        mark_json_status(&files.state_path, "stopped")?;
    }
    Ok(StopResponse {
        stopped,
        graceful: stopped && !request.force,
    })
}

fn prepare_embeddable(paths: &AppPaths, reinstall: bool) -> Result<LemonadeInstallManifest> {
    let root = paths.engine_dir(ENGINE_NAME);
    let downloads = root.join("downloads");
    let archive = downloads.join(EMBEDDABLE_ARCHIVE_NAME);
    fs::create_dir_all(&downloads)?;
    if !archive.is_file() {
        eprintln!("Downloading {EMBEDDABLE_ARCHIVE_NAME}...");
        download_file(EMBEDDABLE_URL, &archive)?;
    } else {
        eprintln!("Using cached {EMBEDDABLE_ARCHIVE_NAME}.");
    }
    let runtime_dir = runtime_dir(paths);
    if reinstall || !lemond_path_in(&runtime_dir).is_file() {
        if runtime_dir.exists() {
            fs::remove_dir_all(&runtime_dir)
                .with_context(|| format!("failed to clear {}", runtime_dir.display()))?;
        }
        let extract_root = root
            .join("extract")
            .join(format!("{}", current_unix_millis()));
        fs::create_dir_all(&extract_root)?;
        extract_archive(&archive, &extract_root)?;
        let embeddable_root = find_embeddable_root(&extract_root)?;
        copy_tree(&embeddable_root, &runtime_dir)?;
        fs::remove_dir_all(&extract_root).ok();
    }
    let lemond = lemond_path_in(&runtime_dir);
    let lemonade = lemonade_path_in(&runtime_dir);
    if !lemond.is_file() || !lemonade.is_file() {
        bail!(
            "Lemonade embeddable extraction did not produce lemonade/lemond binaries in {}",
            runtime_dir.display()
        );
    }
    Ok(LemonadeInstallManifest {
        env_id: format!("lemonade-embeddable-{LEMONADE_VERSION}"),
        version: LEMONADE_VERSION.to_owned(),
        runtime_dir,
        lemond,
        lemonade,
        backend_recipe: ROCM_BACKEND_RECIPE.to_owned(),
        backend_name: ROCM_BACKEND_NAME.to_owned(),
        installed_at_unix_ms: current_unix_millis(),
    })
}

fn install_rocm_backend(manifest: &LemonadeInstallManifest) -> Result<()> {
    let port = free_local_port()?;
    let log_path = manifest.runtime_dir.join("install-lemond.log");
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, Some(&log_path))?;
    let result = (|| -> Result<()> {
        wait_for_health(DEFAULT_HOST, port, Duration::from_secs(30))?;
        let system_info = query_system_info(DEFAULT_HOST, port)?;
        let backend = lemonade_rocm_backend(&system_info)?;
        match backend.state.as_str() {
            "installed" | "ready" => Ok(()),
            "installable" | "not_installed" | "available" | "update_required" => {
                eprintln!("Installing Lemonade llamacpp:rocm backend...");
                post_install_backend(DEFAULT_HOST, port)
            }
            other => bail!(
                "Lemonade llamacpp:rocm backend is not usable: state={other} detail={}. No CPU fallback is used.",
                backend.message.unwrap_or_else(|| "no detail".to_owned())
            ),
        }
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

fn resolve_runtime() -> Result<LemonadeRuntime> {
    let paths = AppPaths::discover()?;
    let manifest = read_manifest(&paths)?;
    if !manifest.lemond.is_file() {
        bail!(
            "Lemonade runtime is missing {}; run `rocm engines install lemonade`",
            manifest.lemond.display()
        );
    }
    Ok(LemonadeRuntime { manifest })
}

fn manifest_path(paths: &AppPaths) -> PathBuf {
    paths.engine_manifests_dir(ENGINE_NAME).join("runtime.json")
}

fn runtime_dir(paths: &AppPaths) -> PathBuf {
    paths.engine_dir(ENGINE_NAME).join("runtime")
}

fn read_manifest(paths: &AppPaths) -> Result<LemonadeInstallManifest> {
    let path = manifest_path(paths);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_manifest(paths: &AppPaths, manifest: &LemonadeInstallManifest) -> Result<()> {
    let path = manifest_path(paths);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}

fn lemond_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemond"))
}

fn lemonade_path_in(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join(platform_binary_name("lemonade"))
}

fn platform_binary_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let response = ureq::get(url)
        .timeout(Duration::from_secs(900))
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut reader = response.into_reader();
    let mut file = fs::File::create(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write {}", destination.display()))?;
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path) -> Result<()> {
    if cfg!(windows) {
        let output = ProcessCommand::new("powershell.exe")
            .arg("-NoProfile")
            .arg("-NonInteractive")
            .arg("-ExecutionPolicy")
            .arg("Bypass")
            .arg("-Command")
            .arg(
                "& { param($archive, $destination) \
                 Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                 [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $destination) }",
            )
            .arg(archive)
            .arg(destination)
            .stdin(Stdio::null())
            .output()
            .context("failed to run PowerShell ZIP extraction")?;
        if output.status.success() {
            return Ok(());
        }
        bail!(
            "PowerShell ZIP extraction failed with status {}; stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let output = ProcessCommand::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(destination)
        .stdin(Stdio::null())
        .output()
        .context("failed to run tar")?;
    if output.status.success() {
        Ok(())
    } else {
        bail!(
            "tar extraction failed with status {}; stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
}

fn find_embeddable_root(extract_root: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    collect_embeddable_roots(extract_root, &mut candidates, 0)?;
    candidates.into_iter().next().with_context(|| {
        format!(
            "no Lemonade embeddable root found in {}",
            extract_root.display()
        )
    })
}

fn collect_embeddable_roots(
    path: &Path,
    candidates: &mut Vec<PathBuf>,
    depth: usize,
) -> Result<()> {
    if depth > 4 {
        return Ok(());
    }
    if lemond_path_in(path).is_file() && lemonade_path_in(path).is_file() {
        candidates.push(path.to_path_buf());
        return Ok(());
    }
    if !path.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.path().is_dir() {
            collect_embeddable_roots(&entry.path(), candidates, depth + 1)?;
        }
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else if source_path.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
            if !cfg!(windows) {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let metadata = fs::metadata(&source_path)?;
                    fs::set_permissions(
                        &destination_path,
                        fs::Permissions::from_mode(metadata.permissions().mode()),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn spawn_lemond(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
) -> Result<std::process::Child> {
    let mut command = ProcessCommand::new(&manifest.lemond);
    command
        .arg(&manifest.runtime_dir)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null());
    if let Some(log_path) = log_path {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let log = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    command
        .spawn()
        .with_context(|| format!("failed to start {}", manifest.lemond.display()))
}

fn post_install_backend(host: &str, port: u16) -> Result<()> {
    let url = format!("{}/v1/install", format_http_base_url(host, port));
    let body = json!({
        "recipe": ROCM_BACKEND_RECIPE,
        "backend": ROCM_BACKEND_NAME,
        "stream": false,
    });
    let response = ureq::post(&url)
        .timeout(Duration::from_secs(1800))
        .send_json(body)
        .with_context(|| format!("failed to request Lemonade backend install at {url}"))?;
    let text = response.into_string().unwrap_or_default();
    if !text.trim().is_empty()
        && let Ok(value) = serde_json::from_str::<Value>(&text)
        && value.get("error").is_some()
    {
        bail!("Lemonade backend install failed: {value}");
    }
    Ok(())
}

fn post_load_model_rocm(host: &str, port: u16, model_ref: &str) -> Result<Value> {
    let url = format!("{}/v1/load", format_http_base_url(host, port));
    let body = json!({
        "model_name": model_ref,
        "llamacpp_backend": ROCM_BACKEND_NAME,
        "save_options": true,
    });
    let response = ureq::post(&url)
        .timeout(Duration::from_secs(900))
        .send_json(body)
        .with_context(|| format!("failed to request Lemonade model load at {url}"))?;
    let text = response.into_string().unwrap_or_default();
    if text.trim().is_empty() {
        return Ok(json!({
            "status": "unknown",
            "message": "empty Lemonade load response"
        }));
    }
    let value = serde_json::from_str::<Value>(&text)
        .with_context(|| format!("failed to parse Lemonade load response: {text}"))?;
    if value.get("error").is_some()
        || value
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("error"))
    {
        bail!("Lemonade ROCm model load failed: {value}");
    }
    Ok(value)
}

fn query_system_info(host: &str, port: u16) -> Result<Value> {
    let url = format!("{}/v1/system-info", format_http_base_url(host, port));
    let text = ureq::get(&url)
        .timeout(Duration::from_secs(15))
        .call()
        .with_context(|| format!("failed to query Lemonade system info at {url}"))?
        .into_string()
        .context("failed to read Lemonade system info")?;
    serde_json::from_str(&text).context("failed to parse Lemonade system info JSON")
}

#[derive(Debug, Clone)]
struct LemonadeBackendState {
    state: String,
    message: Option<String>,
}

impl LemonadeBackendState {
    fn backend_is_ready(&self) -> bool {
        matches!(self.state.as_str(), "installed" | "ready")
    }
}

fn lemonade_rocm_backend(system_info: &Value) -> Result<LemonadeBackendState> {
    let backends = system_info
        .get("recipes")
        .and_then(|recipes| recipes.get(ROCM_BACKEND_RECIPE))
        .and_then(|recipe| recipe.get("backends"))
        .with_context(|| "Lemonade system-info did not include llamacpp backends")?;
    let backend = backends
        .get(ROCM_BACKEND_NAME)
        .with_context(|| "Lemonade system-info did not include llamacpp:rocm")?;
    let state = backend
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let message = backend
        .get("message")
        .or_else(|| backend.get("error"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if state == "unsupported" {
        bail!(
            "Lemonade reports llamacpp:rocm is unsupported on this host: {}. No CPU fallback is used.",
            message.unwrap_or_else(|| "no detail".to_owned())
        );
    }
    Ok(LemonadeBackendState { state, message })
}

fn wait_for_health(host: &str, port: u16, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    let mut last_error = None;
    while start.elapsed() < timeout {
        match query_health(host, port) {
            Ok(true) => return Ok(()),
            Ok(false) => last_error = Some("health endpoint returned not ok".to_owned()),
            Err(error) => last_error = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    bail!(
        "Lemonade health endpoint did not become ready: {}",
        last_error.unwrap_or_else(|| "not checked".to_owned())
    )
}

fn query_health(host: &str, port: u16) -> Result<bool> {
    let value = query_health_json(host, port)?;
    Ok(value
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "ok" | "ready" | "running")))
}

fn query_health_json(host: &str, port: u16) -> Result<Value> {
    let url = format!("{}/v1/health", format_http_base_url(host, port));
    ureq::get(&url)
        .timeout(Duration::from_secs(3))
        .call()
        .with_context(|| format!("failed to query Lemonade health at {url}"))?
        .into_json()
        .context("failed to parse Lemonade health JSON")
}

fn query_loaded_model_endpoint(endpoint_url: &str, model_ref: &str) -> Result<bool> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let health = query_health_json(&host, port)?;
    Ok(health_has_loaded_model(&health, model_ref))
}

fn wait_for_model_loaded(
    host: &str,
    port: u16,
    model_ref: &str,
    timeout: Duration,
) -> Result<Value> {
    let start = std::time::Instant::now();
    let mut last_health = None;
    let mut last_error = None;
    while start.elapsed() < timeout {
        match query_health_json(host, port) {
            Ok(health) => {
                if health_has_loaded_model(&health, model_ref) {
                    return Ok(health);
                }
                last_health = Some(health);
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if let Some(error) = last_error {
        bail!("failed while waiting for Lemonade model load: {error}");
    }
    bail!(
        "timed out waiting for Lemonade to load {model_ref}; last health: {}",
        last_health
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned())
    )
}

fn health_has_loaded_model(health: &Value, model_ref: &str) -> bool {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        return false;
    }
    health
        .get("all_models_loaded")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|model| {
            let name_matches = ["model_name", "id", "name"]
                .into_iter()
                .filter_map(|field| model.get(field).and_then(Value::as_str))
                .any(|loaded| model_names_match(loaded, model_ref));
            let backend_is_rocm = model
                .get("recipe_options")
                .and_then(|options| options.get("llamacpp_backend"))
                .and_then(Value::as_str)
                .is_some_and(lemonade_backend_is_rocm);
            name_matches && backend_is_rocm
        })
}

fn model_names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
        || resolve_lemonade_model_ref(left).eq_ignore_ascii_case(&resolve_lemonade_model_ref(right))
}

fn lemonade_backend_is_rocm(value: &str) -> bool {
    value
        .trim()
        .to_ascii_lowercase()
        .starts_with(ROCM_BACKEND_NAME)
}

fn serve_http_command_args(request: &ServeHttpRequest) -> Vec<String> {
    let mut args = vec![
        "serve-http".to_owned(),
        request.service_id.clone(),
        request.model_ref.clone(),
        "--host".to_owned(),
        request.host.clone(),
        "--port".to_owned(),
        request.port.to_string(),
        "--device-policy".to_owned(),
        device_policy_name(&request.device_policy).to_owned(),
        "--state-path".to_owned(),
        request.state_path.display().to_string(),
    ];
    if let Some(runtime_id) = request.runtime_id.as_deref() {
        args.extend(["--runtime-id".to_owned(), runtime_id.to_owned()]);
    }
    if let Some(env_id) = request.env_id.as_deref() {
        args.extend(["--env-id".to_owned(), env_id.to_owned()]);
    }
    if let Some(log_path) = request.log_path.as_ref() {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    if let Some(engine_recipe) = request.engine_recipe.as_ref() {
        args.extend([
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe).expect("engine recipe serializes"),
        ]);
    }
    args
}

#[cfg(windows)]
fn spawn_serve_http_background(current_exe: &Path, serve_args: &[String]) -> Result<u32> {
    let output = ProcessCommand::new("powershell.exe")
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-Command")
        .arg("$p = Start-Process -FilePath $args[0] -ArgumentList $args[1..($args.Count-1)] -WindowStyle Hidden -PassThru; [Console]::Out.Write($p.Id)")
        .arg(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .output()
        .context("failed to invoke PowerShell background launcher")?;
    if !output.status.success() {
        bail!(
            "PowerShell background launcher failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let pid_text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    pid_text
        .parse::<u32>()
        .with_context(|| format!("invalid launcher pid `{pid_text}`"))
}

#[cfg(not(windows))]
fn spawn_serve_http_background(current_exe: &Path, serve_args: &[String]) -> Result<u32> {
    let child = ProcessCommand::new(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to launch Lemonade serve-http background process")?;
    Ok(child.id())
}

fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    pid: u32,
    server_pid: Option<u32>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server_pid,
            "model_ref": request.model_ref,
            "canonical_model_id": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": device_policy_name(&request.device_policy),
            "runtime_id": request.runtime_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "env_id": request.env_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "runtime_kind": "lemonade_embeddable",
            "runtime_executable": runtime.manifest.lemond,
            "log_path": request.log_path.as_ref().map(|path| path.display().to_string()),
            "engine_recipe": request.engine_recipe,
            "started_at_unix_ms": current_unix_millis()
        }),
    )
}

fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    Ok(ServiceFiles {
        state_path: paths
            .engine_state_dir(ENGINE_NAME)
            .join(format!("{service_id}.json")),
        log_path: paths
            .engine_logs_dir(ENGINE_NAME)
            .join(format!("{service_id}.log")),
    })
}

fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    let mut value = read_service_state(path).unwrap_or_else(|_| json!({}));
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    if let Some(patch) = patch.as_object() {
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    write_state(path, &value)
}

fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    merge_json_state(
        path,
        &json!({
            "engine": ENGINE_NAME,
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn value_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn pid_to_terminate_from_state(state: &Value) -> Option<u32> {
    value_u32(state, "server_pid").or_else(|| value_u32(state, "pid"))
}

fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    Ok(lines.into_iter().collect())
}

fn terminate_pid(pid: u32, force: bool) -> bool {
    #[cfg(windows)]
    {
        let mut command = ProcessCommand::new("taskkill");
        command.arg("/PID").arg(pid.to_string()).arg("/T");
        if force {
            command.arg("/F");
        }
        command
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
    #[cfg(not(windows))]
    {
        let signal = if force { "-KILL" } else { "-TERM" };
        ProcessCommand::new("kill")
            .arg(signal)
            .arg(pid.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
}

fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0)).context("failed to reserve local port")?;
    Ok(listener.local_addr()?.port())
}

fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

fn normalize_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

fn require_gpu_required(policy: &DevicePolicy) -> Result<()> {
    match policy {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(()),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

fn parse_device_policy_arg(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unknown device policy `{other}`"),
    }
}

fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn accepted_engine_recipe(
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<Option<EngineRecipeHint>> {
    if let Some(hint) = &engine_recipe {
        if hint.engine != ENGINE_NAME {
            bail!(
                "engine_recipe target `{}` does not match adapter `{}`",
                hint.engine,
                ENGINE_NAME
            );
        }
        if hint.contract_version != ENGINE_RECIPE_CONTRACT_VERSION {
            bail!(
                "engine_recipe contract `{}` is unsupported; expected `{}`",
                hint.contract_version,
                ENGINE_RECIPE_CONTRACT_VERSION
            );
        }
    }
    Ok(engine_recipe)
}

fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

fn resolve_lemonade_model_ref(model_ref: &str) -> String {
    let trimmed = model_ref.trim();
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.is_empty()
        || matches!(
            lower.as_str(),
            "qwen"
                | "assistant"
                | "default"
                | "small"
                | "tiny"
                | "lemonade-qwen"
                | "qwen-gguf"
                | "qwen3-0.6b-gguf"
        )
        || lower.contains("qwen2.5-1.5b")
        || lower.contains("qwen3.5-0.8b")
    {
        DEFAULT_MODEL.to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn manifest_lock_hash(manifest: &LemonadeInstallManifest) -> String {
    let mut hasher = DefaultHasher::new();
    manifest.env_id.hash(&mut hasher);
    manifest.version.hash(&mut hasher);
    manifest.runtime_dir.hash(&mut hasher);
    manifest.lemond.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn read_request() -> Result<EngineRequestEnvelope> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read stdin for engine request")?;
    serde_json::from_str(&buffer).context("failed to parse engine request envelope")
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, value)?;
    writeln!(&mut handle)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_alias_resolves_to_small_gpu_gguf_model() {
        assert_eq!(resolve_lemonade_model_ref("qwen"), DEFAULT_MODEL);
        assert_eq!(
            resolve_lemonade_model_ref("Qwen/Qwen2.5-1.5B-Instruct"),
            DEFAULT_MODEL
        );
    }

    #[test]
    fn device_policy_rejects_cpu_without_fallback() {
        let error = normalize_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("cpu should be rejected")
            .to_string();
        assert!(error.contains("no CPU fallback"));
    }

    #[test]
    fn endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    #[test]
    fn serve_http_args_preserve_runtime_selection() {
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: DEFAULT_MODEL.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: DevicePolicy::GpuRequired,
            runtime_id: Some("runtime".to_owned()),
            env_id: Some("env".to_owned()),
            state_path: PathBuf::from("state.json"),
            log_path: Some(PathBuf::from("service.log")),
            engine_recipe: None,
        };
        let args = serve_http_command_args(&request);
        assert!(args.contains(&"--runtime-id".to_owned()));
        assert!(args.contains(&"runtime".to_owned()));
        assert!(args.contains(&"--env-id".to_owned()));
        assert!(args.contains(&"env".to_owned()));
        assert!(!args.iter().any(|arg| arg == "cpu"));
    }

    #[test]
    fn rocm_backend_parser_rejects_unsupported_state() {
        let value = json!({
            "recipes": {
                "llamacpp": {
                    "backends": {
                        "rocm": {
                            "state": "unsupported",
                            "message": "Unsupported GPU"
                        }
                    }
                }
            }
        });
        let error = lemonade_rocm_backend(&value)
            .expect_err("unsupported backend should fail")
            .to_string();
        assert!(error.contains("unsupported"));
        assert!(error.contains("No CPU fallback"));
    }

    #[test]
    fn health_parser_requires_loaded_requested_model() {
        let unloaded = json!({
            "status": "ok",
            "model_loaded": null,
            "all_models_loaded": []
        });
        assert!(!health_has_loaded_model(&unloaded, DEFAULT_MODEL));

        let loaded = json!({
            "status": "ok",
            "model_loaded": DEFAULT_MODEL,
            "all_models_loaded": [{
                "model_name": DEFAULT_MODEL,
                "recipe": "llamacpp",
                "recipe_options": {
                    "llamacpp_backend": "rocm"
                }
            }]
        });
        assert!(health_has_loaded_model(&loaded, "lemonade-qwen"));
    }
}
