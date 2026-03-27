use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, detect_host_therock_family, extract_first_gfx_token,
    interactive_terminal, normalize_therock_family, require_nonempty,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, DevicePolicy, EndpointResponse, EngineCapabilities,
    EngineDeviceAvailability, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope,
    InstallRequest, InstallResponse, LaunchRequest, LaunchResponse, ResolveModelRequest,
    ResolveModelResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ENGINE_NAME: &str = "pytorch";
const DEFAULT_RUNTIME_ID: &str = "therock-release";
const THEROCK_SIMPLE_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/v2";
const PYTHON_WORKER_SOURCE: &str = include_str!("python_worker.py");
const DEFAULT_PIP_TIMEOUT_SECS: u64 = 600;
const DEFAULT_PIP_RETRIES: u32 = 8;
const ENGINE_DEPENDENCIES: &[&str] = &[
    "fastapi",
    "uvicorn",
    "pydantic",
    "transformers",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "jinja2",
];
const THEROCK_TORCH_PACKAGES: &[&str] = &["torch"];
const TORCH_STACK_DEPENDENCIES: &[&str] = &["accelerate"];
const KNOWN_THEROCK_FAMILIES: &[&str] = &[
    "gfx94X-dcgpu",
    "gfx950-dcgpu",
    "gfx110X-all",
    "gfx1151",
    "gfx120X-all",
];

#[derive(Parser, Debug)]
#[command(
    name = "rocm-engine-pytorch",
    about = "rocm-cli PyTorch engine",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = DEFAULT_RUNTIME_ID)]
        runtime_id: String,
        #[arg(long)]
        python_version: Option<String>,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
        #[arg(long)]
        device_policy: Option<DevicePolicyArg>,
    },
    Launch {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<DevicePolicyArg>,
    },
    Stdio,
    #[command(hide = true)]
    ServeHttp {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long, default_value = "gpu_preferred")]
        device_policy: String,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DevicePolicyArg {
    GpuRequired,
    GpuPreferred,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EngineEnvManifest {
    env_id: String,
    runtime_id: String,
    requested_python_version: Option<String>,
    python_launcher: String,
    python_executable: String,
    env_path: PathBuf,
    manifest_path: PathBuf,
    lock_path: PathBuf,
    installed_packages: Vec<String>,
    lock_hash: String,
    #[serde(default)]
    therock_channel: Option<String>,
    #[serde(default)]
    therock_family: Option<String>,
    #[serde(default)]
    therock_index_url: Option<String>,
    #[serde(default)]
    therock_packages: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct PythonLauncher {
    program: String,
    args: Vec<String>,
    display: String,
}

#[derive(Debug, Clone, Copy)]
enum TheRockChannel {
    Release,
    Nightly,
}

impl TheRockChannel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }
}

#[derive(Debug, Clone)]
struct TheRockRuntimeRequest {
    channel: TheRockChannel,
    family_override: Option<String>,
}

#[derive(Debug, Clone)]
struct TheRockTorchResolution {
    channel: TheRockChannel,
    family: String,
    index_url: String,
    packages: Vec<String>,
    source: String,
}

#[derive(Debug, Clone, Copy)]
enum ModelFamily {
    Generic,
    Qwen,
    Glm,
    Llama,
    Gpt2,
}

#[derive(Debug, Clone)]
struct ModelRecipe {
    canonical_model_id: String,
    task: &'static str,
    source: String,
    loader: &'static str,
    trust_remote_code: bool,
    chat_template_mode: &'static str,
    preferred_dtype: &'static str,
    estimated_memory: String,
    min_gpu_mem_gb: Option<u32>,
    warnings: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => {
            print_json(&detect_response())?;
        }
        CommandKind::Capabilities => {
            print_json(&capabilities())?;
        }
        CommandKind::Install {
            runtime_id,
            python_version,
            reinstall,
        } => {
            let response = install_response(InstallRequest {
                runtime_id,
                python_version,
                reinstall,
            })?;
            print_json(&response)?;
        }
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => {
            let response = resolve_model_response(ResolveModelRequest {
                model_ref,
                runtime_id: None,
                device_policy: device_policy.map(Into::into),
                recipe_override: None,
            })?;
            print_json(&response)?;
        }
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
        } => {
            let response = launch_service(LaunchRequest {
                service_id,
                env_id: None,
                runtime_id: None,
                model_ref,
                host,
                port,
                device_policy: device_policy.map(Into::into),
                endpoint_mode: Some("openai".to_owned()),
            })?;
            print_json(&response)?;
        }
        CommandKind::Stdio => {
            let envelope = read_request()?;
            let response = handle_envelope(envelope);
            print_json(&response)?;
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            env_id,
            runtime_id,
            state_path,
        } => {
            serve_http(
                service_id,
                model_ref,
                host,
                port,
                parse_device_policy(&device_policy)?,
                env_id,
                runtime_id,
                state_path,
            )?;
        }
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
        EngineMethod::Endpoint => EngineResponseEnvelope::success(EndpointResponse {
            endpoint_url: format!("http://127.0.0.1:{DEFAULT_LOCAL_PORT}/v1"),
            api_style: "openai".to_owned(),
            supported_routes: vec![
                "/healthz".to_owned(),
                "/v1/models".to_owned(),
                "/v1/chat/completions".to_owned(),
                "/v1/completions".to_owned(),
            ],
        }),
        other => EngineResponseEnvelope::failure(
            "unimplemented_method",
            format!("method {other:?} is not implemented by the PyTorch engine"),
        ),
    }
}

fn deserialize_and_respond<T, F, U>(
    payload: serde_json::Value,
    handler: F,
) -> EngineResponseEnvelope
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

fn detect_response() -> DetectResponse {
    let manifest = latest_env_manifest().ok().flatten();
    let installed = manifest.is_some();
    let detected_family = detect_host_therock_family();
    let env_id = manifest.as_ref().map(|value| value.env_id.clone());
    let python_version = manifest
        .as_ref()
        .and_then(|value| value.requested_python_version.clone())
        .or_else(|| Some(default_python_version().to_owned()));
    let torch_version = manifest
        .as_ref()
        .and_then(|value| find_installed_package(&value.installed_packages, "torch"));
    let transformers_version = manifest
        .as_ref()
        .and_then(|value| find_installed_package(&value.installed_packages, "transformers"));

    let mut notes = Vec::new();
    if let Some(manifest) = &manifest {
        notes.push(format!(
            "managed env detected at {}",
            manifest.env_path.display()
        ));
        if let Some(family) = manifest.therock_family.as_deref() {
            notes.push(format!("TheRock family: {family}"));
        }
        if let Some(channel) = manifest.therock_channel.as_deref() {
            notes.push(format!("TheRock channel: {channel}"));
        }
        if let Some(index_url) = manifest.therock_index_url.as_deref() {
            notes.push(format!("TheRock index: {index_url}"));
        }
        notes.extend(manifest.warnings.iter().cloned());
    } else {
        notes.push("no managed PyTorch envs found; run `rocm engines install pytorch`".to_owned());
    }

    DetectResponse {
        installed,
        env_id,
        python_version,
        torch_version,
        transformers_version,
        available_devices: vec![
            EngineDeviceAvailability {
                kind: "cpu".to_owned(),
                available: true,
                reason: None,
            },
            EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: detected_family.is_some(),
                reason: detected_family
                    .as_ref()
                    .map(|family| format!("detected host family {family}"))
                    .or_else(|| {
                        Some("no supported TheRock GPU family detected on this host".to_owned())
                    }),
            },
        ],
        capabilities: capabilities(),
        notes,
    }
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: true,
        rocm_gpu: true,
        multi_gpu: true,
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "limited".to_owned(),
        distributed_serving: false,
        reasoning_parser: false,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    require_nonempty(&request.runtime_id, "runtime_id")?;
    let manifest = create_or_update_env_manifest(&request)?;
    Ok(InstallResponse {
        env_id: manifest.env_id,
        env_path: manifest.env_path.display().to_string(),
        python_executable: manifest.python_executable,
        installed_packages: manifest.installed_packages,
        capabilities: capabilities(),
        lock_hash: manifest.lock_hash,
        warnings: manifest.warnings,
    })
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    require_nonempty(&request.model_ref, "model_ref")?;
    let recipe = resolve_model_recipe(&request.model_ref);
    let device_policy = request
        .device_policy
        .unwrap_or_else(|| default_device_policy_for_recipe(&recipe));
    Ok(ResolveModelResponse {
        canonical_model_id: recipe.canonical_model_id,
        task: recipe.task.to_owned(),
        source: recipe.source,
        revision: "main".to_owned(),
        loader: recipe.loader.to_owned(),
        trust_remote_code: recipe.trust_remote_code,
        chat_template_mode: recipe.chat_template_mode.to_owned(),
        dtype: recipe.preferred_dtype.to_owned(),
        device_policy,
        estimated_memory: recipe.estimated_memory,
        launch_defaults: json!({
            "host": "127.0.0.1",
            "port": DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        warnings: recipe.warnings,
    })
}

fn launch_service(request: LaunchRequest) -> Result<LaunchResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;

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
    let log_file = fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone log file handle")?;

    let current_exe =
        std::env::current_exe().context("failed to discover current engine binary")?;
    let child = Command::new(current_exe)
        .arg("serve-http")
        .arg(&request.service_id)
        .arg(&request.model_ref)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(
            &request.device_policy.unwrap_or(DevicePolicy::GpuPreferred),
        ))
        .args(optional_arg("--env-id", request.env_id.as_deref()))
        .args(optional_arg("--runtime-id", request.runtime_id.as_deref()))
        .arg("--state-path")
        .arg(&state_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn pytorch serve-http process")?;

    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({
            "engine": ENGINE_NAME,
            "service_id": request.service_id,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "pid": child.id(),
            "status": "starting"
        }))?,
    )
    .with_context(|| format!("failed to write {}", state_path.display()))?;

    Ok(LaunchResponse {
        service_id: request.service_id,
        pid: child.id(),
        endpoint_url: format!("http://{}:{}/v1", request.host, request.port),
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

fn create_or_update_env_manifest(request: &InstallRequest) -> Result<EngineEnvManifest> {
    let paths = AppPaths::discover()?;
    paths.ensure()?;
    fs::create_dir_all(paths.engine_envs_dir(ENGINE_NAME))?;
    fs::create_dir_all(paths.engine_locks_dir(ENGINE_NAME))?;
    fs::create_dir_all(paths.engine_manifests_dir(ENGINE_NAME))?;

    let env_id = managed_env_id(&request.runtime_id, request.python_version.as_deref());
    let env_path = paths.engine_envs_dir(ENGINE_NAME).join(&env_id);
    let lock_path = paths
        .engine_locks_dir(ENGINE_NAME)
        .join(format!("{env_id}.txt"));
    let manifest_path = paths
        .engine_manifests_dir(ENGINE_NAME)
        .join(format!("{env_id}.json"));
    let existing_manifest = if manifest_path.is_file() {
        Some(load_manifest(&manifest_path)?)
    } else {
        None
    };
    let therock_resolution = if std::env::var("ROCM_CLI_PYTORCH_PACKAGE_SPEC").is_ok() {
        None
    } else {
        resolve_therock_torch_resolution(&request.runtime_id)?
    };

    if !request.reinstall {
        if let Some(manifest) = existing_manifest.clone() {
            if manifest.env_path.is_dir()
                && (manifest_has_torch(&manifest) || therock_resolution.is_none())
            {
                return Ok(manifest);
            }
        }
    }

    if env_path.exists() {
        fs::remove_dir_all(&env_path)
            .with_context(|| format!("failed to remove {}", env_path.display()))?;
    }

    let launcher = discover_python_launcher(request.python_version.as_deref())?;
    let env_path_string = env_path.to_string_lossy().to_string();
    run_command(
        &launcher.program,
        launcher
            .args
            .iter()
            .map(String::as_str)
            .chain(["-m", "venv", env_path_string.as_str()]),
        "create managed pytorch venv",
    )?;

    let python_executable = venv_python_path(&env_path);
    let python_executable_string = python_executable.to_string_lossy().to_string();
    ensure_pip_available(&python_executable_string)?;
    run_progress_command(
        &python_executable_string,
        std::iter::once("-m")
            .chain(std::iter::once("pip"))
            .chain(std::iter::once("install"))
            .chain(pip_install_network_args().iter().map(String::as_str))
            .chain(ENGINE_DEPENDENCIES.iter().copied()),
        "install managed pytorch engine dependencies",
    )?;

    let mut warnings = Vec::new();
    let mut therock_channel = None;
    let mut therock_family = None;
    let mut therock_index_url = None;
    let mut therock_packages = Vec::new();
    let maybe_torch_spec = std::env::var("ROCM_CLI_PYTORCH_PACKAGE_SPEC").ok();
    let maybe_extra_index = std::env::var("ROCM_CLI_PYTORCH_EXTRA_INDEX_URL").ok();
    if let Some(torch_spec) = maybe_torch_spec.as_deref() {
        let mut args = vec!["-m".to_owned(), "pip".to_owned(), "install".to_owned()];
        args.extend(pip_install_network_args());
        if let Some(extra_index) = maybe_extra_index.as_deref() {
            args.push("--extra-index-url".to_owned());
            args.push(extra_index.to_owned());
        }
        args.push(torch_spec.to_owned());
        run_progress_command(
            &python_executable_string,
            args.iter().map(String::as_str),
            "install torch package into managed pytorch env",
        )?;
        run_progress_command(
            &python_executable_string,
            std::iter::once("-m")
                .chain(std::iter::once("pip"))
                .chain(std::iter::once("install"))
                .chain(pip_install_network_args().iter().map(String::as_str))
                .chain(TORCH_STACK_DEPENDENCIES.iter().copied()),
            "install pytorch engine runtime dependencies",
        )?;
        warnings.push(
            "using manual torch package override from ROCM_CLI_PYTORCH_PACKAGE_SPEC".to_owned(),
        );
    } else {
        match therock_resolution {
            Some(resolution) => {
                install_therock_torch_packages(&python_executable_string, &resolution)?;
                run_progress_command(
                    &python_executable_string,
                    std::iter::once("-m")
                        .chain(std::iter::once("pip"))
                        .chain(std::iter::once("install"))
                        .chain(pip_install_network_args().iter().map(String::as_str))
                        .chain(TORCH_STACK_DEPENDENCIES.iter().copied()),
                    "install pytorch engine runtime dependencies",
                )?;
                therock_channel = Some(resolution.channel.as_str().to_owned());
                therock_family = Some(resolution.family.clone());
                therock_index_url = Some(resolution.index_url.clone());
                therock_packages = resolution.packages.clone();
                warnings.push(format!(
                    "installed TheRock PyTorch packages from {} ({}, source={})",
                    resolution.index_url, resolution.family, resolution.source
                ));
            }
            None => {
                warnings.push(
                    "torch installation deferred because no supported TheRock GPU family could be resolved; set ROCM_CLI_THEROCK_FAMILY or use a runtime_id like therock-release:gfx950-dcgpu".to_owned(),
                );
            }
        }
    }

    let freeze = capture_command(
        &python_executable_string,
        ["-m", "pip", "freeze"],
        "capture managed pytorch env lockfile",
    )?;
    fs::write(&lock_path, &freeze)
        .with_context(|| format!("failed to write {}", lock_path.display()))?;
    let installed_packages = freeze
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let lock_hash = simple_hash(&freeze);

    let manifest = EngineEnvManifest {
        env_id,
        runtime_id: request.runtime_id.clone(),
        requested_python_version: request.python_version.clone(),
        python_launcher: launcher.display,
        python_executable: python_executable.display().to_string(),
        env_path,
        manifest_path,
        lock_path,
        installed_packages,
        lock_hash,
        therock_channel,
        therock_family,
        therock_index_url,
        therock_packages,
        warnings,
    };
    write_manifest(&manifest)?;
    Ok(manifest)
}

fn latest_env_manifest() -> Result<Option<EngineEnvManifest>> {
    let paths = AppPaths::discover()?;
    let manifests_dir = paths.engine_manifests_dir(ENGINE_NAME);
    if !manifests_dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&manifests_dir)
        .with_context(|| format!("failed to read {}", manifests_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        manifests.push(load_manifest(&path)?);
    }
    manifests.sort_by(|left, right| left.env_id.cmp(&right.env_id));
    Ok(manifests.pop())
}

fn load_manifest(path: &Path) -> Result<EngineEnvManifest> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read engine manifest {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse engine manifest {}", path.display()))
}

fn write_manifest(manifest: &EngineEnvManifest) -> Result<()> {
    fs::write(
        &manifest.manifest_path,
        serde_json::to_vec_pretty(manifest).context("failed to serialize engine manifest")?,
    )
    .with_context(|| format!("failed to write {}", manifest.manifest_path.display()))?;
    Ok(())
}

fn discover_python_launcher(requested_version: Option<&str>) -> Result<PythonLauncher> {
    let mut candidates = Vec::new();
    if cfg!(windows) {
        if let Some(version) = requested_version {
            candidates.push(PythonLauncher {
                program: "py".to_owned(),
                args: vec![format!("-{version}")],
                display: format!("py -{version}"),
            });
        }
        candidates.push(PythonLauncher {
            program: "py".to_owned(),
            args: Vec::new(),
            display: "py".to_owned(),
        });
        candidates.push(PythonLauncher {
            program: "python".to_owned(),
            args: Vec::new(),
            display: "python".to_owned(),
        });
    } else {
        if let Some(version) = requested_version {
            candidates.push(PythonLauncher {
                program: format!("python{version}"),
                args: Vec::new(),
                display: format!("python{version}"),
            });
        }
        candidates.push(PythonLauncher {
            program: "python3".to_owned(),
            args: Vec::new(),
            display: "python3".to_owned(),
        });
        candidates.push(PythonLauncher {
            program: "python".to_owned(),
            args: Vec::new(),
            display: "python".to_owned(),
        });
    }

    for launcher in candidates {
        let status = Command::new(&launcher.program)
            .args(&launcher.args)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(value) if value.success()) {
            return Ok(launcher);
        }
    }

    bail!("unable to locate a usable Python launcher for the pytorch engine")
}

fn venv_python_path(env_path: &Path) -> PathBuf {
    if cfg!(windows) {
        env_path.join("Scripts").join("python.exe")
    } else {
        env_path.join("bin").join("python")
    }
}

fn serve_http(
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    env_id: Option<String>,
    runtime_id: Option<String>,
    state_path: PathBuf,
) -> Result<()> {
    let manifest = ensure_service_env(runtime_id.as_deref(), env_id.as_deref())?;
    let recipe = resolve_model_recipe(&model_ref);
    let worker_script = materialize_python_worker()?;
    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({
            "engine": ENGINE_NAME,
            "service_id": service_id,
            "model_ref": model_ref,
            "status": "starting",
            "device_policy": device_policy_name(&device_policy),
            "env_id": manifest.env_id,
            "runtime_id": manifest.runtime_id,
            "python_executable": manifest.python_executable,
            "preferred_dtype": recipe.preferred_dtype,
            "estimated_memory": recipe.estimated_memory,
            "trust_remote_code": recipe.trust_remote_code,
            "endpoint_url": format!("http://{}:{}/v1", host, port)
        }))?,
    )?;
    let mut child = Command::new(&manifest.python_executable)
        .arg(&worker_script)
        .arg("--service-id")
        .arg(&service_id)
        .arg("--model-ref")
        .arg(&model_ref)
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--device-policy")
        .arg(device_policy_name(&device_policy))
        .arg("--state-path")
        .arg(&state_path)
        .arg("--env-id")
        .arg(&manifest.env_id)
        .arg("--runtime-id")
        .arg(&manifest.runtime_id)
        .arg("--preferred-dtype")
        .arg(recipe.preferred_dtype)
        .args(optional_arg_owned(
            "--min-gpu-mem-gb",
            recipe.min_gpu_mem_gb.map(|value| value.to_string()),
        ))
        .args(flag_arg("--trust-remote-code", recipe.trust_remote_code))
        .env("PYTHONUNBUFFERED", "1")
        .env("TOKENIZERS_PARALLELISM", "false")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to start python worker for pytorch engine")?;

    let status = child
        .wait()
        .context("failed waiting for pytorch python worker")?;
    if status.success() {
        Ok(())
    } else {
        bail!(
            "pytorch worker exited with status {} for service {}",
            status,
            service_id
        )
    }
}

fn canonical_model_id(model_ref: &str) -> String {
    match model_ref.to_ascii_lowercase().as_str() {
        "qwen3.5" | "qwen" => "Qwen/Qwen3.5-4B".to_owned(),
        "qwen32b" | "qwen3.5-32b" | "qwen3-32b" => "Qwen/Qwen3-32B-FP8".to_owned(),
        "glm5" | "glm-5" => "zai-org/GLM-5-FP8".to_owned(),
        "llama3.2" | "llama" => "meta-llama/Llama-3.2-3B-Instruct".to_owned(),
        "tiny-gpt2" | "gpt2tiny" => "sshleifer/tiny-gpt2".to_owned(),
        other if other.contains('/') => model_ref.to_owned(),
        _ => model_ref.to_owned(),
    }
}

fn resolve_model_recipe(model_ref: &str) -> ModelRecipe {
    if let Some(recipe) = known_recipe_for_model(model_ref) {
        return recipe;
    }

    let canonical_model_id = canonical_model_id(model_ref);
    let source = if model_ref.contains('/') {
        "huggingface".to_owned()
    } else {
        "alias".to_owned()
    };
    let family = infer_model_family(&canonical_model_id);
    let trust_remote_code = matches!(family, ModelFamily::Glm);
    let preferred_dtype = preferred_dtype_for_model(&canonical_model_id, family);
    let min_gpu_mem_gb = estimate_gpu_memory_gib(&canonical_model_id);
    let estimated_memory = format_estimated_memory(&canonical_model_id, min_gpu_mem_gb);

    let mut warnings = Vec::new();
    if trust_remote_code {
        warnings.push(
            "this model family is configured with trust_remote_code enabled by recipe".to_owned(),
        );
    }
    if let Some(min_gpu_mem_gb) = min_gpu_mem_gb {
        if min_gpu_mem_gb >= 48 {
            warnings.push(format!(
                "this model looks large (~{min_gpu_mem_gb} GiB GPU memory recommended); automatic CPU fallback is disabled for safety"
            ));
        } else if min_gpu_mem_gb >= 16 {
            warnings.push(format!(
                "this model may need about {min_gpu_mem_gb} GiB of GPU memory for comfortable serving"
            ));
        }
    }

    ModelRecipe {
        canonical_model_id,
        task: "chat",
        source,
        loader: "transformers",
        trust_remote_code,
        chat_template_mode: "auto",
        preferred_dtype,
        estimated_memory,
        min_gpu_mem_gb,
        warnings,
    }
}

fn known_recipe_for_model(model_ref: &str) -> Option<ModelRecipe> {
    let normalized = model_ref.trim().to_ascii_lowercase();
    let source = if model_ref.contains('/') {
        "huggingface".to_owned()
    } else {
        "alias".to_owned()
    };

    match normalized.as_str() {
        "qwen3.5" | "qwen" | "qwen3.5-4b" | "qwen/qwen3.5-4b" => Some(build_known_recipe(
            "Qwen/Qwen3.5-4B",
            source,
            false,
            "bfloat16",
            Some(12),
            Vec::new(),
        )),
        "qwen32b" | "qwen3.5-32b" | "qwen3-32b" | "qwen3-32b-fp8" | "qwen/qwen3-32b-fp8" => {
            Some(build_known_recipe(
                "Qwen/Qwen3-32B-FP8",
                source,
                false,
                "auto",
                Some(48),
                vec![
                    "this recipe prefers ROCm GPU execution and may span multiple accelerators"
                        .to_owned(),
                ],
            ))
        }
        "glm5" | "glm-5" | "glm-5-fp8" | "zai-org/glm-5-fp8" => Some(build_known_recipe(
            "zai-org/GLM-5-FP8",
            source,
            true,
            "auto",
            Some(905),
            vec![
                "this model family is configured with trust_remote_code enabled by recipe"
                    .to_owned(),
            ],
        )),
        "llama3.2"
        | "llama"
        | "llama-3.2-3b"
        | "llama-3.2-3b-instruct"
        | "meta-llama/llama-3.2-3b-instruct" => Some(build_known_recipe(
            "meta-llama/Llama-3.2-3B-Instruct",
            source,
            false,
            "bfloat16",
            Some(8),
            Vec::new(),
        )),
        "tiny-gpt2" | "gpt2tiny" | "sshleifer/tiny-gpt2" => Some(build_known_recipe(
            "sshleifer/tiny-gpt2",
            source,
            false,
            "float32",
            None,
            Vec::new(),
        )),
        _ => None,
    }
}

fn build_known_recipe(
    canonical_model_id: &str,
    source: String,
    trust_remote_code: bool,
    preferred_dtype: &'static str,
    min_gpu_mem_gb: Option<u32>,
    mut warnings: Vec<String>,
) -> ModelRecipe {
    if let Some(min_gpu_mem_gb) = min_gpu_mem_gb {
        if min_gpu_mem_gb >= 48
            && !warnings
                .iter()
                .any(|value| value.contains("automatic CPU fallback"))
        {
            warnings.push(format!(
                "this model looks large (~{min_gpu_mem_gb} GiB GPU memory recommended); automatic CPU fallback is disabled for safety"
            ));
            if !warnings.iter().any(|value| value.contains("visible GPUs")) {
                warnings.push(
                    "startup will attempt auto device_map placement across visible GPUs when aggregate memory is sufficient"
                        .to_owned(),
                );
            }
        } else if min_gpu_mem_gb >= 16 && !warnings.iter().any(|value| value.contains("GPU memory"))
        {
            warnings.push(format!(
                "this model may need about {min_gpu_mem_gb} GiB of GPU memory for comfortable serving"
            ));
        }
    }

    ModelRecipe {
        canonical_model_id: canonical_model_id.to_owned(),
        task: "chat",
        source,
        loader: "transformers",
        trust_remote_code,
        chat_template_mode: "auto",
        preferred_dtype,
        estimated_memory: format_estimated_memory(canonical_model_id, min_gpu_mem_gb),
        min_gpu_mem_gb,
        warnings,
    }
}

fn infer_model_family(model_ref: &str) -> ModelFamily {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("qwen") {
        ModelFamily::Qwen
    } else if lower.contains("glm") {
        ModelFamily::Glm
    } else if lower.contains("llama") {
        ModelFamily::Llama
    } else if lower.contains("gpt2") {
        ModelFamily::Gpt2
    } else {
        ModelFamily::Generic
    }
}

fn preferred_dtype_for_model(model_ref: &str, family: ModelFamily) -> &'static str {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("fp8") || lower.contains("gptq") || lower.contains("awq") {
        "auto"
    } else if matches!(
        family,
        ModelFamily::Qwen | ModelFamily::Glm | ModelFamily::Llama
    ) {
        "bfloat16"
    } else {
        "auto"
    }
}

fn infer_parameter_billions(model_ref: &str) -> Option<f32> {
    let lower = model_ref.to_ascii_lowercase();
    if lower == "zai-org/glm-5-fp8" {
        return Some(754.0);
    }

    lower
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '.'))
        .find_map(|token| {
            token
                .strip_suffix('b')
                .and_then(|value| value.parse::<f32>().ok())
                .or_else(|| {
                    token
                        .strip_suffix('m')
                        .and_then(|value| value.parse::<f32>().ok())
                        .map(|value| value / 1000.0)
                })
        })
}

fn infer_weight_bytes_per_param(model_ref: &str) -> f32 {
    let lower = model_ref.to_ascii_lowercase();
    if lower.contains("int4") || lower.contains("awq") || lower.contains("gptq") {
        0.5
    } else if lower.contains("fp8") || lower.contains("int8") {
        1.0
    } else {
        2.0
    }
}

fn estimate_gpu_memory_gib(model_ref: &str) -> Option<u32> {
    let params = infer_parameter_billions(model_ref)?;
    let bytes_per_param = infer_weight_bytes_per_param(model_ref);
    let overhead = if bytes_per_param <= 1.0 { 1.20 } else { 1.35 };
    Some((params * bytes_per_param * overhead).ceil().max(2.0) as u32)
}

fn format_estimated_memory(model_ref: &str, min_gpu_mem_gb: Option<u32>) -> String {
    match (infer_parameter_billions(model_ref), min_gpu_mem_gb) {
        (Some(params), Some(min_gpu_mem_gb)) => {
            format!(
                "~{min_gpu_mem_gb} GiB GPU memory recommended for ~{}B parameters",
                trim_float(params)
            )
        }
        (Some(params), None) => format!(
            "~{}B parameters; memory estimate unavailable",
            trim_float(params)
        ),
        (None, Some(min_gpu_mem_gb)) => format!("~{min_gpu_mem_gb} GiB GPU memory recommended"),
        (None, None) => "memory estimate unavailable".to_owned(),
    }
}

fn default_device_policy_for_recipe(recipe: &ModelRecipe) -> DevicePolicy {
    if recipe.min_gpu_mem_gb.unwrap_or_default() >= 48 {
        DevicePolicy::GpuRequired
    } else {
        DevicePolicy::GpuPreferred
    }
}

fn trim_float(value: f32) -> String {
    if (value.fract() - 0.0).abs() < f32::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn parse_device_policy(value: &str) -> Result<DevicePolicy> {
    match value {
        "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu_only" => Ok(DevicePolicy::CpuOnly),
        _ => bail!("unknown device policy: {value}"),
    }
}

fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

fn managed_env_id(runtime_id: &str, python_version: Option<&str>) -> String {
    let python = python_version.unwrap_or(default_python_version());
    format!(
        "{}-{}-{}",
        std::env::consts::OS,
        slugify(runtime_id),
        slugify(python)
    )
}

fn default_python_version() -> &'static str {
    if cfg!(windows) { "3.11" } else { "3.10" }
}

fn parse_therock_runtime_request(runtime_id: &str) -> TheRockRuntimeRequest {
    let normalized = runtime_id.trim().to_ascii_lowercase();
    let channel = if normalized.contains("nightly") {
        TheRockChannel::Nightly
    } else {
        TheRockChannel::Release
    };

    let family_override = KNOWN_THEROCK_FAMILIES
        .iter()
        .find(|family| normalized.contains(&family.to_ascii_lowercase()))
        .map(|family| (*family).to_owned())
        .or_else(|| {
            extract_first_gfx_token(&normalized)
                .and_then(|target| normalize_therock_family(&target))
        });

    TheRockRuntimeRequest {
        channel,
        family_override,
    }
}

fn resolve_therock_torch_resolution(runtime_id: &str) -> Result<Option<TheRockTorchResolution>> {
    let runtime_request = parse_therock_runtime_request(runtime_id);

    if let Some(family) = runtime_request.family_override {
        return Ok(Some(TheRockTorchResolution {
            channel: runtime_request.channel,
            family: family.clone(),
            index_url: therock_index_url(&family),
            packages: THEROCK_TORCH_PACKAGES
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            source: "runtime_id".to_owned(),
        }));
    }

    if let Some(value) = std::env::var("ROCM_CLI_THEROCK_FAMILY").ok() {
        if let Some(family) = normalize_therock_family(&value) {
            return Ok(Some(TheRockTorchResolution {
                channel: runtime_request.channel,
                family: family.clone(),
                index_url: therock_index_url(&family),
                packages: THEROCK_TORCH_PACKAGES
                    .iter()
                    .map(|item| (*item).to_owned())
                    .collect(),
                source: "env".to_owned(),
            }));
        }
    }

    if let Some(family) = detect_host_therock_family() {
        return Ok(Some(TheRockTorchResolution {
            channel: runtime_request.channel,
            family: family.clone(),
            index_url: therock_index_url(&family),
            packages: THEROCK_TORCH_PACKAGES
                .iter()
                .map(|item| (*item).to_owned())
                .collect(),
            source: "host".to_owned(),
        }));
    }

    Ok(None)
}

fn install_therock_torch_packages(
    python_executable: &str,
    resolution: &TheRockTorchResolution,
) -> Result<()> {
    let mut args = vec![
        "-m".to_owned(),
        "pip".to_owned(),
        "install".to_owned(),
        "--timeout".to_owned(),
        pip_timeout_secs().to_string(),
        "--retries".to_owned(),
        pip_retries().to_string(),
        "--disable-pip-version-check".to_owned(),
        "--progress-bar".to_owned(),
        "on".to_owned(),
        "--index-url".to_owned(),
        resolution.index_url.clone(),
        "--upgrade-strategy".to_owned(),
        "only-if-needed".to_owned(),
    ];
    if matches!(resolution.channel, TheRockChannel::Nightly) {
        args.push("--pre".to_owned());
    }
    args.extend(resolution.packages.iter().cloned());
    run_progress_command(
        python_executable,
        args.iter().map(String::as_str),
        "install TheRock torch packages into managed pytorch env",
    )
}

fn therock_index_url(family: &str) -> String {
    format!("{THEROCK_SIMPLE_INDEX_BASE}/{family}/")
}

fn simple_hash(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn find_installed_package(packages: &[String], name: &str) -> Option<String> {
    packages.iter().find_map(|entry| {
        entry
            .strip_prefix(&format!("{name}=="))
            .map(ToOwned::to_owned)
    })
}

fn manifest_has_torch(manifest: &EngineEnvManifest) -> bool {
    find_installed_package(&manifest.installed_packages, "torch").is_some()
}

fn ensure_pip_available(python_executable: &str) -> Result<()> {
    if command_succeeds(python_executable, ["-m", "pip", "--version"])? {
        return Ok(());
    }

    run_command(
        python_executable,
        ["-m", "ensurepip", "--upgrade"],
        "bootstrap pip in managed pytorch env",
    )?;
    run_command(
        python_executable,
        ["-m", "pip", "--version"],
        "verify pip in managed pytorch env",
    )
}

fn pip_timeout_secs() -> u64 {
    std::env::var("ROCM_CLI_PIP_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PIP_TIMEOUT_SECS)
}

fn pip_retries() -> u32 {
    std::env::var("ROCM_CLI_PIP_RETRIES")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_PIP_RETRIES)
}

fn pip_install_network_args() -> Vec<String> {
    vec![
        "--timeout".to_owned(),
        pip_timeout_secs().to_string(),
        "--retries".to_owned(),
        pip_retries().to_string(),
        "--disable-pip-version-check".to_owned(),
        "--progress-bar".to_owned(),
        "on".to_owned(),
    ]
}

fn command_succeeds<'a, I>(program: &str, args: I) -> Result<bool>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let output = Command::new(program)
        .args(&args)
        .output()
        .with_context(|| format!("failed to start {}", args.join(" ")))?;
    Ok(output.status.success())
}

fn ensure_service_env(runtime_id: Option<&str>, env_id: Option<&str>) -> Result<EngineEnvManifest> {
    if let Some(env_id) = env_id {
        let manifest = load_manifest_by_env_id(env_id)?;
        if manifest_has_torch(&manifest) {
            return Ok(manifest);
        }
        return create_or_update_env_manifest(&InstallRequest {
            runtime_id: manifest.runtime_id.clone(),
            python_version: manifest.requested_python_version.clone(),
            reinstall: true,
        });
    }

    if let Some(runtime_id) = runtime_id {
        return create_or_update_env_manifest(&InstallRequest {
            runtime_id: runtime_id.to_owned(),
            python_version: None,
            reinstall: false,
        });
    }

    if let Some(manifest) = latest_runnable_env_manifest()? {
        return Ok(manifest);
    }

    create_or_update_env_manifest(&InstallRequest {
        runtime_id: DEFAULT_RUNTIME_ID.to_owned(),
        python_version: None,
        reinstall: false,
    })
}

fn latest_runnable_env_manifest() -> Result<Option<EngineEnvManifest>> {
    let paths = AppPaths::discover()?;
    let manifests_dir = paths.engine_manifests_dir(ENGINE_NAME);
    if !manifests_dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&manifests_dir)
        .with_context(|| format!("failed to read {}", manifests_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let manifest = load_manifest(&path)?;
        if manifest.env_path.is_dir() && manifest_has_torch(&manifest) {
            manifests.push(manifest);
        }
    }
    manifests.sort_by(|left, right| left.env_id.cmp(&right.env_id));
    Ok(manifests.pop())
}

fn load_manifest_by_env_id(env_id: &str) -> Result<EngineEnvManifest> {
    let paths = AppPaths::discover()?;
    let path = paths
        .engine_manifests_dir(ENGINE_NAME)
        .join(format!("{env_id}.json"));
    load_manifest(&path)
}

fn materialize_python_worker() -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    let worker_dir = paths.engine_dir(ENGINE_NAME).join("worker");
    fs::create_dir_all(&worker_dir)
        .with_context(|| format!("failed to create {}", worker_dir.display()))?;

    let worker_path = worker_dir.join("python_worker.py");
    let needs_write = match fs::read_to_string(&worker_path) {
        Ok(current) => current != PYTHON_WORKER_SOURCE,
        Err(_) => true,
    };
    if needs_write {
        fs::write(&worker_path, PYTHON_WORKER_SOURCE)
            .with_context(|| format!("failed to write {}", worker_path.display()))?;
    }
    Ok(worker_path)
}

fn optional_arg(flag: &str, value: Option<&str>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value.to_owned()],
        None => Vec::new(),
    }
}

fn optional_arg_owned(flag: &str, value: Option<String>) -> Vec<String> {
    match value {
        Some(value) => vec![flag.to_owned(), value],
        None => Vec::new(),
    }
}

fn flag_arg(flag: &str, enabled: bool) -> Vec<String> {
    if enabled {
        vec![flag.to_owned()]
    } else {
        Vec::new()
    }
}

fn run_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let output = Command::new(program)
        .args(&args)
        .output()
        .with_context(|| format!("failed to start {context_label}"))?;
    if output.status.success() {
        Ok(())
    } else {
        bail!(
            "{} failed (status {}): {}",
            context_label,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn run_progress_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    if interactive_terminal() {
        let status = Command::new(program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("failed to start {context_label}"))?;
        if status.success() {
            return Ok(());
        }
        bail!("{context_label} failed (status {status})");
    }
    run_command(program, args.iter().map(String::as_str), context_label)
}

fn capture_command<'a, I>(program: &str, args: I, context_label: &str) -> Result<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let args = args.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    let output = Command::new(program)
        .args(&args)
        .output()
        .with_context(|| format!("failed to start {context_label}"))?;
    if !output.status.success() {
        bail!(
            "{} failed (status {}): {}",
            context_label,
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("command output was not valid utf-8")
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

impl From<DevicePolicyArg> for DevicePolicy {
    fn from(value: DevicePolicyArg) -> Self {
        match value {
            DevicePolicyArg::GpuRequired => DevicePolicy::GpuRequired,
            DevicePolicyArg::GpuPreferred => DevicePolicy::GpuPreferred,
            DevicePolicyArg::CpuOnly => DevicePolicy::CpuOnly,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }
}
