use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{IsTerminal, stdin, stdout};
use std::path::PathBuf;
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

#[derive(Debug, Clone, Serialize)]
pub struct DoctorSummary {
    pub os: String,
    pub arch: String,
    pub interactive_terminal: bool,
    pub default_engine: &'static str,
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
            default_engine: default_engine_for_platform(),
            config_dir: paths.config_dir,
            data_dir: paths.data_dir,
            cache_dir: paths.cache_dir,
        })
    }

    pub fn render_text(&self) -> String {
        format!(
            "rocm doctor\n  os: {}\n  arch: {}\n  interactive_terminal: {}\n  default_engine: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
            self.os,
            self.arch,
            self.interactive_terminal,
            self.default_engine,
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
        "pytorch"
    } else {
        "vllm"
    }
}

pub fn require_nonempty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
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
