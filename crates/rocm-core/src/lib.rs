use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{IsTerminal, stdin, stdout};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_LOCAL_PORT: u16 = 11_435;
pub const DEFAULT_LOCAL_HOST: &str = "127.0.0.1";

#[derive(Debug, Clone, Serialize)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> Result<Self> {
        let project_dirs = ProjectDirs::from("com", "powderluv", "rocm-cli")
            .context("unable to determine project directories for rocm-cli")?;
        Ok(Self {
            config_dir: project_dirs.config_dir().to_path_buf(),
            data_dir: project_dirs.data_dir().to_path_buf(),
            cache_dir: project_dirs.cache_dir().to_path_buf(),
        })
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.cache_dir,
            &self.automations_dir(),
            &self.data_dir.join("engines"),
            &self.data_dir.join("logs"),
            &self.data_dir.join("services"),
            &self.data_dir.join("models"),
            &self.data_dir.join("runtimes"),
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        Ok(())
    }

    pub fn engine_dir(&self, engine: &str) -> PathBuf {
        self.data_dir.join("engines").join(engine)
    }

    pub fn engine_logs_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("logs")
    }

    pub fn engine_envs_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("envs")
    }

    pub fn engine_locks_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("locks")
    }

    pub fn engine_manifests_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("manifests")
    }

    pub fn engine_state_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("state")
    }

    pub fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }

    pub fn services_dir(&self) -> PathBuf {
        self.data_dir.join("services")
    }

    pub fn automations_dir(&self) -> PathBuf {
        self.data_dir.join("automations")
    }

    pub fn automation_state_path(&self) -> PathBuf {
        self.automations_dir().join("runtime-state.json")
    }

    pub fn automation_events_path(&self) -> PathBuf {
        self.automations_dir().join("events.jsonl")
    }

    pub fn service_manifest_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.json"))
    }

    pub fn service_log_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.log"))
    }

    pub fn service_engine_state_path(&self, engine: &str, service_id: &str) -> PathBuf {
        self.engine_state_dir(engine)
            .join(format!("{service_id}.json"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorSummary {
    pub os: String,
    pub arch: String,
    pub interactive_terminal: bool,
    pub default_engine: String,
    pub detected_gfx_target: Option<String>,
    pub detected_therock_family: Option<String>,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl DoctorSummary {
    pub fn gather() -> Result<Self> {
        let paths = AppPaths::discover()?;
        Ok(Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            interactive_terminal: interactive_terminal(),
            default_engine: default_engine_for_platform().to_owned(),
            detected_gfx_target: detect_host_gfx_target(),
            detected_therock_family: detect_host_therock_family(),
            config_dir: paths.config_dir,
            data_dir: paths.data_dir,
            cache_dir: paths.cache_dir,
        })
    }

    pub fn render_text(&self) -> String {
        format!(
            "rocm doctor\n  os: {}\n  arch: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  detected_therock_family: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
            self.os,
            self.arch,
            self.interactive_terminal,
            self.default_engine,
            self.detected_gfx_target.as_deref().unwrap_or("<unknown>"),
            self.detected_therock_family
                .as_deref()
                .unwrap_or("<unknown>"),
            self.config_dir.display(),
            self.data_dir.display(),
            self.cache_dir.display(),
        )
    }
}

pub fn interactive_terminal() -> bool {
    stdin().is_terminal() && stdout().is_terminal()
}

pub fn default_engine_for_platform() -> &'static str {
    if cfg!(target_os = "windows") {
        return "pytorch";
    }

    if sibling_binary_exists("rocm-engine-vllm") {
        "vllm"
    } else {
        "pytorch"
    }
}

pub fn require_nonempty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
}

pub fn detect_host_therock_family() -> Option<String> {
    detect_host_gfx_target().and_then(|target| normalize_therock_family(&target))
}

pub fn detect_host_gfx_target() -> Option<String> {
    capture_optional_command("rocm_agent_enumerator", &[])
        .and_then(|output| extract_first_gfx_token(&output))
        .or_else(|| {
            capture_optional_command("rocminfo", &[])
                .and_then(|output| extract_first_gfx_token(&output))
        })
        .or_else(detect_linux_sysfs_gfx_target)
}

pub fn extract_first_gfx_token(text: &str) -> Option<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .find_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if normalized.starts_with("gfx") {
                Some(normalized)
            } else {
                None
            }
        })
}

pub fn normalize_therock_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let target = extract_first_gfx_token(&normalized).unwrap_or(normalized);
    match target.as_str() {
        value if value.starts_with("gfx101") => Some("gfx101X-dgpu".to_owned()),
        value if value.starts_with("gfx103") => Some("gfx103X-dgpu".to_owned()),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Some("gfx110X-all".to_owned()),
        value if value.starts_with("gfx1150") => Some("gfx1150".to_owned()),
        value if value.starts_with("gfx1151") => Some("gfx1151".to_owned()),
        value if value.starts_with("gfx1152") => Some("gfx1152".to_owned()),
        value if value.starts_with("gfx1153") => Some("gfx1153".to_owned()),
        "gfx1200" | "gfx1201" => Some("gfx120X-all".to_owned()),
        value if value.starts_with("gfx900") => Some("gfx900".to_owned()),
        value if value.starts_with("gfx906") => Some("gfx906".to_owned()),
        value if value.starts_with("gfx908") => Some("gfx908".to_owned()),
        value if value.starts_with("gfx90a") => Some("gfx90a".to_owned()),
        value if value.starts_with("gfx950") => Some("gfx950-dcgpu".to_owned()),
        value
            if value.starts_with("gfx942")
                || value.starts_with("gfx94")
                || value.starts_with("gfx9-4") =>
        {
            Some("gfx94X-dcgpu".to_owned())
        }
        value if value.starts_with("gfx90") => Some("gfx90X-dcgpu".to_owned()),
        _ => None,
    }
}

fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(target_os = "linux")]
fn detect_linux_sysfs_gfx_target() -> Option<String> {
    detect_linux_kfd_gfx_target().or_else(detect_linux_drm_ip_discovery_gfx_target)
}

#[cfg(not(target_os = "linux"))]
fn detect_linux_sysfs_gfx_target() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn detect_linux_kfd_gfx_target() -> Option<String> {
    let nodes_dir = Path::new("/sys/class/kfd/kfd/topology/nodes");
    let entries = fs::read_dir(nodes_dir).ok()?;
    for entry in entries.flatten() {
        let Some(value) = fs::read_to_string(entry.path().join("gfx_target_version")).ok() else {
            continue;
        };
        let Some(token) = parse_linux_kfd_gfx_target(value.trim()) else {
            continue;
        };
        return Some(token);
    }
    None
}

#[cfg(target_os = "linux")]
fn parse_linux_kfd_gfx_target(value: &str) -> Option<String> {
    if let Some(token) = extract_first_gfx_token(value) {
        return Some(token);
    }
    let digits = value.trim();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    match digits.len() {
        3 | 4 => Some(format!("gfx{digits}")),
        5 | 6 => {
            let raw: u32 = digits.parse().ok()?;
            let major = raw / 10_000;
            let minor = (raw / 100) % 100;
            let revision = raw % 100;
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
            Some(format!("gfx{digits}"))
        }
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_ip_discovery_gfx_target() -> Option<String> {
    let drm_dir = Path::new("/sys/class/drm");
    let entries = fs::read_dir(drm_dir).ok()?;
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(card_name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !card_name.starts_with("card") || card_name.contains('-') {
            continue;
        }
        let device_dir = card_path.join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        let gc_root = device_dir.join("ip_discovery");
        let token = detect_ip_discovery_gc_target(&gc_root);
        if token.is_some() {
            return token;
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn is_amdgpu_device(device_dir: &Path) -> bool {
    if let Ok(vendor) = fs::read_to_string(device_dir.join("vendor"))
        && vendor.trim().eq_ignore_ascii_case("0x1002")
    {
        return true;
    }
    if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
        return uevent.lines().any(|line| line.trim() == "DRIVER=amdgpu");
    }
    false
}

#[cfg(target_os = "linux")]
fn detect_ip_discovery_gc_target(ip_discovery_dir: &Path) -> Option<String> {
    let die_entries = fs::read_dir(ip_discovery_dir.join("die")).ok()?;
    for die in die_entries.flatten() {
        let Some(gc_entries) = fs::read_dir(die.path().join("GC")).ok() else {
            continue;
        };
        for gc in gc_entries.flatten() {
            let block = gc.path();
            let Some(major) = fs::read_to_string(block.join("major"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(minor) = fs::read_to_string(block.join("minor"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(revision) = fs::read_to_string(block.join("revision"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
        }
    }
    None
}

fn gfx_target_from_gc_version(major: u32, minor: u32, revision: u32) -> Option<String> {
    if major == 0 {
        return None;
    }
    Some(format!("gfx{major}{minor}{revision}"))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WatcherMode {
    Observe,
    Propose,
    Contained,
}

impl WatcherMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Propose => "propose",
            Self::Contained => "contained",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinWatcherSpec {
    pub id: &'static str,
    pub summary: &'static str,
    pub trigger: &'static str,
    pub default_mode: WatcherMode,
    pub actions: &'static [&'static str],
}

const BUILTIN_WATCHERS: &[BuiltinWatcherSpec] = &[
    BuiltinWatcherSpec {
        id: "therock-update",
        summary: "Emit scheduled TheRock update reminders and proposals.",
        trigger: "schedule: every 6h",
        default_mode: WatcherMode::Observe,
        actions: &["remind_update_check", "queue_update_proposal"],
    },
    BuiltinWatcherSpec {
        id: "server-recover",
        summary: "Observe or restart failed managed services when restart metadata exists.",
        trigger: "event: managed_service_failed",
        default_mode: WatcherMode::Contained,
        actions: &["collect_failure_snapshot", "restart_managed_service"],
    },
];

pub fn builtin_watchers() -> &'static [BuiltinWatcherSpec] {
    BUILTIN_WATCHERS
}

pub fn builtin_watcher(id: &str) -> Option<&'static BuiltinWatcherSpec> {
    builtin_watchers().iter().find(|watcher| watcher.id == id)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineUserConfig {
    #[serde(default)]
    pub preferred_runtime_id: Option<String>,
    #[serde(default)]
    pub preferred_env_id: Option<String>,
    #[serde(default)]
    pub last_installed_runtime_id: Option<String>,
    #[serde(default)]
    pub last_installed_env_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mode: Option<WatcherMode>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutomationsConfig {
    #[serde(default)]
    pub daemon_enabled: bool,
    #[serde(default)]
    pub watchers: BTreeMap<String, WatcherUserConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RocmCliConfig {
    #[serde(default)]
    pub default_engine: Option<String>,
    #[serde(default)]
    pub engines: BTreeMap<String, EngineUserConfig>,
    #[serde(default)]
    pub automations: AutomationsConfig,
}

impl RocmCliConfig {
    pub fn load(paths: &AppPaths) -> Result<Self> {
        let path = paths.config_path();
        if !path.is_file() {
            return Ok(Self::default());
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        let path = paths.config_path();
        fs::write(
            &path,
            serde_json::to_vec_pretty(self).context("failed to serialize rocm-cli config")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn engine_config(&self, engine: &str) -> Option<&EngineUserConfig> {
        self.engines.get(engine)
    }

    pub fn engine_config_mut(&mut self, engine: &str) -> &mut EngineUserConfig {
        self.engines.entry(engine.to_owned()).or_default()
    }

    pub fn watcher_config(&self, watcher: &str) -> Option<&WatcherUserConfig> {
        self.automations.watchers.get(watcher)
    }

    pub fn watcher_config_mut(&mut self, watcher: &str) -> &mut WatcherUserConfig {
        self.automations
            .watchers
            .entry(watcher.to_owned())
            .or_default()
    }

    pub fn automation_daemon_enabled(&self) -> bool {
        self.automations.daemon_enabled || self.automations.watchers.values().any(|cfg| cfg.enabled)
    }

    pub fn watcher_enabled(&self, watcher: &BuiltinWatcherSpec) -> bool {
        self.watcher_config(watcher.id)
            .map(|cfg| cfg.enabled)
            .unwrap_or(false)
    }

    pub fn effective_watcher_mode(&self, watcher: &BuiltinWatcherSpec) -> WatcherMode {
        self.watcher_config(watcher.id)
            .and_then(|cfg| cfg.mode)
            .unwrap_or(watcher.default_mode)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherRuntimeSnapshot {
    pub id: String,
    pub enabled: bool,
    pub mode: WatcherMode,
    pub summary: String,
    #[serde(default)]
    pub last_event: Option<String>,
    #[serde(default)]
    pub last_event_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRuntimeState {
    pub running: bool,
    pub automations_enabled: bool,
    pub daemon_pid: u32,
    pub started_at_unix_ms: u128,
    pub last_tick_unix_ms: u128,
    pub active_watchers: Vec<WatcherRuntimeSnapshot>,
}

impl AutomationRuntimeState {
    pub fn load(paths: &AppPaths) -> Result<Option<Self>> {
        let path = paths.automation_state_path();
        if !path.is_file() {
            return Ok(None);
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let state = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(state))
    }

    pub fn write(&self, paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        let path = paths.automation_state_path();
        fs::write(
            &path,
            serde_json::to_vec_pretty(self)
                .context("failed to serialize automation runtime state")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn watcher_mut(&mut self, watcher_id: &str) -> Option<&mut WatcherRuntimeSnapshot> {
        self.active_watchers
            .iter_mut()
            .find(|watcher| watcher.id == watcher_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEventRecord {
    pub at_unix_ms: u128,
    pub watcher_id: String,
    pub level: String,
    pub action: String,
    pub message: String,
    #[serde(default)]
    pub service_id: Option<String>,
}

pub fn append_automation_event(paths: &AppPaths, event: &AutomationEventRecord) -> Result<()> {
    paths.ensure()?;
    let path = paths.automation_events_path();
    let mut line =
        serde_json::to_string(event).context("failed to serialize automation event record")?;
    line.push('\n');
    let mut existing = if path.is_file() {
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
    existing.push_str(&line);
    fs::write(&path, existing).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn load_recent_automation_events(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationEventRecord>> {
    let path = paths.automation_events_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }

    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let text =
        String::from_utf8(bytes).with_context(|| format!("failed to decode {}", path.display()))?;
    let mut events = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<AutomationEventRecord>(line)
            .with_context(|| format!("failed to parse event in {}", path.display()))?;
        events.push(event);
    }
    if events.len() > limit {
        events.drain(0..events.len() - limit);
    }
    Ok(events)
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ManagedServiceRecord {
    pub service_id: String,
    pub engine: String,
    pub model_ref: String,
    pub canonical_model_id: String,
    pub host: String,
    pub port: u16,
    pub endpoint_url: String,
    pub mode: String,
    pub status: String,
    pub supervisor_pid: u32,
    pub engine_pid: Option<u32>,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub env_id: Option<String>,
    #[serde(default)]
    pub device_policy: Option<String>,
    #[serde(default)]
    pub restart_count: u32,
    #[serde(default)]
    pub last_restart_unix_ms: Option<u128>,
    pub manifest_path: PathBuf,
    pub log_path: PathBuf,
    pub engine_state_path: PathBuf,
    pub created_at_unix_ms: u128,
}

impl ManagedServiceRecord {
    pub fn new(
        paths: &AppPaths,
        service_id: impl Into<String>,
        engine: impl Into<String>,
        model_ref: impl Into<String>,
        canonical_model_id: impl Into<String>,
        host: impl Into<String>,
        port: u16,
        mode: impl Into<String>,
        supervisor_pid: u32,
        runtime_id: Option<String>,
        env_id: Option<String>,
        device_policy: Option<String>,
    ) -> Self {
        let service_id = service_id.into();
        let engine = engine.into();
        let host = host.into();
        let manifest_path = paths.service_manifest_path(&service_id);
        let log_path = paths.service_log_path(&service_id);
        let engine_state_path = paths.service_engine_state_path(&engine, &service_id);
        Self {
            endpoint_url: format!("http://{host}:{port}/v1"),
            service_id,
            engine,
            model_ref: model_ref.into(),
            canonical_model_id: canonical_model_id.into(),
            host,
            port,
            mode: mode.into(),
            status: "starting".to_owned(),
            supervisor_pid,
            engine_pid: None,
            runtime_id,
            env_id,
            device_policy,
            restart_count: 0,
            last_restart_unix_ms: None,
            manifest_path,
            log_path,
            engine_state_path,
            created_at_unix_ms: unix_time_millis(),
        }
    }

    pub fn write(&self) -> Result<()> {
        let parent = self
            .manifest_path
            .parent()
            .context("service manifest path must have a parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        fs::write(
            &self.manifest_path,
            serde_json::to_vec_pretty(self).context("failed to serialize service record")?,
        )
        .with_context(|| format!("failed to write {}", self.manifest_path.display()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeSnapshot {
    pub protocol: String,
    pub generated_at_unix_ms: u128,
    pub doctor: DoctorSummary,
    pub gpu: CodexBridgeGpuSnapshot,
    pub config: RocmCliConfig,
    #[serde(default)]
    pub automation_runtime: Option<AutomationRuntimeState>,
    #[serde(default)]
    pub recent_automation_events: Vec<AutomationEventRecord>,
    #[serde(default)]
    pub engines: Vec<CodexBridgeEngine>,
    #[serde(default)]
    pub services: Vec<ManagedServiceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeGpuSnapshot {
    pub amd_smi_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_snapshot: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_snapshot: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeEngine {
    pub id: String,
    pub summary: String,
    pub default_for_platform: bool,
    pub installed_binary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
}

pub fn sibling_binary_path(binary_name: &str) -> Result<PathBuf> {
    require_nonempty(binary_name, "binary_name")?;
    let current_exe = std::env::current_exe().context("failed to discover current executable")?;
    let binary_dir = current_exe
        .parent()
        .context("current executable has no parent directory")?;
    let candidate = binary_dir.join(platform_binary_name(binary_name));
    if candidate.is_file() {
        Ok(candidate)
    } else {
        bail!(
            "unable to locate sibling binary {} next to {}",
            candidate.display(),
            current_exe.display()
        )
    }
}

pub fn sibling_binary_exists(binary_name: &str) -> bool {
    let Ok(current_exe) = std::env::current_exe() else {
        return false;
    };
    let Some(binary_dir) = current_exe.parent() else {
        return false;
    };
    binary_dir.join(platform_binary_name(binary_name)).is_file()
}

pub fn engine_binary_path(engine: &str) -> Result<PathBuf> {
    sibling_binary_path(&format!("rocm-engine-{engine}"))
}

pub fn daemon_binary_path() -> Result<PathBuf> {
    sibling_binary_path("rocmd")
}

pub fn platform_binary_name(binary_name: &str) -> String {
    if cfg!(windows) {
        format!("{binary_name}.exe")
    } else {
        binary_name.to_owned()
    }
}

pub fn generate_service_id(engine: &str, model_ref: &str) -> String {
    let model_slug = sanitize_component(model_ref)
        .trim_matches('-')
        .chars()
        .take(24)
        .collect::<String>();
    format!(
        "{}-{}-{}",
        sanitize_component(engine),
        model_slug,
        unix_time_millis()
    )
}

pub fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

pub fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_binary_name_adds_windows_suffix_only_on_windows() {
        let name = platform_binary_name("rocm");
        if cfg!(windows) {
            assert_eq!(name, "rocm.exe");
        } else {
            assert_eq!(name, "rocm");
        }
    }

    #[test]
    fn default_engine_is_always_usable_on_windows() {
        if cfg!(windows) {
            assert_eq!(default_engine_for_platform(), "pytorch");
        }
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn gc_version_converts_to_gfx_target() {
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 1),
            Some("gfx1101".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 3),
            Some("gfx1103".to_owned())
        );
    }
}
