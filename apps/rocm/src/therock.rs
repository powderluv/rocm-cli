use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, detect_host_therock_family, interactive_terminal, normalize_therock_family,
    unix_time_millis,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const THEROCK_PIP_INDEX_BASE: &str = "https://rocm.nightlies.amd.com/v2";
const THEROCK_RELEASE_TARBALL_BASE: &str = "https://repo.amd.com/rocm/tarball/";
const THEROCK_NIGHTLY_TARBALL_BASE: &str = "https://rocm.nightlies.amd.com/tarball/";
const THEROCK_ROCM_PACKAGE_SPEC: &str = "rocm[libraries,devel]";
const DEFAULT_PIP_TIMEOUT_SECS: u64 = 600;
const DEFAULT_PIP_RETRIES: u32 = 8;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TheRockChannel {
    Release,
    Nightly,
}

impl TheRockChannel {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "release" => Ok(Self::Release),
            "nightly" => Ok(Self::Nightly),
            other => bail!("unsupported TheRock channel: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }

    fn tarball_base_url(self) -> &'static str {
        match self {
            Self::Release => THEROCK_RELEASE_TARBALL_BASE,
            Self::Nightly => THEROCK_NIGHTLY_TARBALL_BASE,
        }
    }
}

#[derive(Debug, Clone)]
struct FamilyResolution {
    family: String,
    source: String,
}

#[derive(Debug, Clone)]
struct PipRuntimeResolution {
    family: String,
    family_source: String,
    index_url: String,
    latest_version: String,
}

#[derive(Debug, Clone)]
struct TarballArtifact {
    family: String,
    family_source: String,
    file_name: String,
    version: String,
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InstalledRuntimeManifest {
    pub runtime_key: String,
    pub runtime_id: String,
    pub channel: String,
    pub format: String,
    pub family: String,
    pub family_source: String,
    pub version: String,
    pub install_root: PathBuf,
    pub selected_artifact_url: String,
    #[serde(default)]
    pub index_url: Option<String>,
    #[serde(default)]
    pub tarball_file_name: Option<String>,
    #[serde(default)]
    pub python_launcher: Option<String>,
    #[serde(default)]
    pub python_executable: Option<String>,
    pub installed_at_unix_ms: u128,
}

#[derive(Debug, Clone, Deserialize)]
struct TarballIndexFile {
    name: String,
    mtime: f64,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct ParsedVersion {
    major: u32,
    minor: u32,
    patch: u32,
    stage: VersionStage,
    stage_number: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum VersionStage {
    Alpha,
    Rc,
    Stable,
}

pub(crate) fn install_sdk(
    paths: &AppPaths,
    channel: &str,
    format: &str,
    prefix: Option<PathBuf>,
    dry_run: bool,
) -> Result<String> {
    let channel = TheRockChannel::parse(channel)?;
    match format {
        "pip" => install_pip_runtime(paths, channel, prefix, dry_run),
        "tarball" => install_tarball_runtime(paths, channel, prefix, dry_run),
        other => bail!("unsupported install format: {other}"),
    }
}

pub(crate) fn render_update_report(paths: &AppPaths) -> Result<String> {
    let manifests = load_runtime_manifests(paths)?;
    let mut output = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(output, "update");
    let _ = writeln!(
        output,
        "  policy: check every run, prompt before mutating state."
    );

    if manifests.is_empty() {
        let _ = writeln!(output, "  managed runtimes: none");
        let _ = writeln!(
            output,
            "  next step: run `rocm install sdk --channel release --dry-run` to resolve a TheRock runtime"
        );
        return Ok(output);
    }

    for manifest in manifests {
        let channel = TheRockChannel::parse(&manifest.channel)?;
        let latest = match manifest.format.as_str() {
            "pip" => match resolve_pip_runtime(channel, Some(manifest.family.as_str())) {
                Ok(resolution) => Some((
                    resolution.latest_version,
                    resolution.index_url,
                    "pip".to_owned(),
                )),
                Err(error) => {
                    let _ = writeln!(
                        output,
                        "  runtime {} format=pip status=error message={}",
                        manifest.runtime_key, error
                    );
                    None
                }
            },
            "tarball" => match resolve_tarball_artifact(channel, Some(manifest.family.as_str())) {
                Ok(artifact) => Some((artifact.version, artifact.url, "tarball".to_owned())),
                Err(error) => {
                    let _ = writeln!(
                        output,
                        "  runtime {} format=tarball status=error message={}",
                        manifest.runtime_key, error
                    );
                    None
                }
            },
            other => {
                let _ = writeln!(
                    output,
                    "  runtime {} format={} status=error message=unknown manifest format",
                    manifest.runtime_key, other
                );
                None
            }
        };

        let Some((latest_version, latest_source, kind)) = latest else {
            continue;
        };
        let status = match compare_version_strings(&manifest.version, &latest_version) {
            Ordering::Less => "update_available",
            Ordering::Equal => "up_to_date",
            Ordering::Greater => "ahead_of_index",
        };
        let _ = writeln!(
            output,
            "  runtime {} format={} channel={} family={} installed={} latest={} status={}",
            manifest.runtime_key,
            kind,
            manifest.channel,
            manifest.family,
            manifest.version,
            latest_version,
            status
        );
        let _ = writeln!(
            output,
            "    install_root: {}",
            manifest.install_root.display()
        );
        let _ = writeln!(output, "    source: {latest_source}");
        if status == "update_available" {
            let _ = writeln!(
                output,
                "    next step: rerun `rocm install sdk --channel {} --format {}` to install the newer runtime",
                manifest.channel, manifest.format
            );
        }
    }

    Ok(output)
}

fn install_pip_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    dry_run: bool,
) -> Result<String> {
    let resolution = resolve_pip_runtime(channel, None)?;
    let runtime_key = runtime_key(channel, "pip", &resolution.family);
    let install_root = prefix.unwrap_or_else(|| managed_runtime_root(paths, "pip", &runtime_key));
    let manifest_path = runtime_manifest_path(paths, &runtime_key);
    let python_launcher = resolve_python_launcher()?;

    let mut output = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(output, "  channel: {}", channel.as_str());
    let _ = writeln!(output, "  format: pip");
    let _ = writeln!(output, "  family: {}", resolution.family);
    let _ = writeln!(output, "  family_source: {}", resolution.family_source);
    let _ = writeln!(output, "  index_url: {}", resolution.index_url);
    let _ = writeln!(output, "  latest_version: {}", resolution.latest_version);
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    let _ = writeln!(output, "  python_launcher: {python_launcher}");
    if dry_run {
        let mut install_args = vec![
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
        if matches!(channel, TheRockChannel::Nightly) {
            install_args.push("--pre".to_owned());
        }
        install_args.push(THEROCK_ROCM_PACKAGE_SPEC.to_owned());
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(
            output,
            "  command: {} -m venv {} && {}/python -m pip install {}",
            python_launcher,
            install_root.display(),
            venv_bin_dir(&install_root).display(),
            install_args.join(" ")
        );
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(output);
    }

    fs::create_dir_all(
        install_root
            .parent()
            .context("runtime install root has no parent directory")?,
    )?;
    ensure_python_venv(&python_launcher, &install_root)?;
    let env_python = venv_python_path(&install_root);
    run_command(
        &env_python,
        &["-m", "ensurepip", "--upgrade"],
        "bootstrap pip in managed TheRock runtime",
    )?;

    let mut install_args = vec![
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
        THEROCK_ROCM_PACKAGE_SPEC.to_owned(),
    ];
    if matches!(channel, TheRockChannel::Nightly) {
        install_args.insert(4, "--pre".to_owned());
    }
    run_progress_command(
        &env_python,
        install_args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        "install TheRock ROCm runtime packages",
    )?;

    let installed_version = resolve_installed_rocm_version(&env_python)?
        .unwrap_or_else(|| resolution.latest_version.clone());
    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), resolution.family),
        channel: channel.as_str().to_owned(),
        format: "pip".to_owned(),
        family: resolution.family.clone(),
        family_source: resolution.family_source.clone(),
        version: installed_version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: resolution.index_url.clone(),
        index_url: Some(resolution.index_url.clone()),
        tarball_file_name: None,
        python_launcher: Some(python_launcher),
        python_executable: Some(env_python.display().to_string()),
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(output, "  installed_version: {installed_version}");
    let _ = writeln!(output, "  python_executable: {}", env_python.display());
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(output)
}

fn install_tarball_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    dry_run: bool,
) -> Result<String> {
    let artifact = resolve_tarball_artifact(channel, None)?;
    let runtime_key = runtime_key(channel, "tarball", &artifact.family);
    let install_root =
        prefix.unwrap_or_else(|| managed_runtime_root(paths, "tarball", &runtime_key));
    let manifest_path = runtime_manifest_path(paths, &runtime_key);
    let cache_path = paths.cache_dir.join("therock").join(&artifact.file_name);

    let mut output = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(output, "  channel: {}", channel.as_str());
    let _ = writeln!(output, "  format: tarball");
    let _ = writeln!(output, "  family: {}", artifact.family);
    let _ = writeln!(output, "  family_source: {}", artifact.family_source);
    let _ = writeln!(output, "  tarball: {}", artifact.file_name);
    let _ = writeln!(output, "  tarball_url: {}", artifact.url);
    let _ = writeln!(output, "  latest_version: {}", artifact.version);
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  cache_path: {}", cache_path.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(output);
    }

    fs::create_dir_all(paths.cache_dir.join("therock"))?;
    fs::create_dir_all(&install_root)?;
    if has_nontrivial_directory_contents(&install_root)? {
        bail!(
            "tarball install target {} is not empty; choose a clean prefix or remove the old extraction first",
            install_root.display()
        );
    }

    download_file(&artifact.url, &cache_path)?;
    extract_tarball(&cache_path, &install_root)?;

    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), artifact.family),
        channel: channel.as_str().to_owned(),
        format: "tarball".to_owned(),
        family: artifact.family.clone(),
        family_source: artifact.family_source.clone(),
        version: artifact.version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: artifact.url.clone(),
        index_url: None,
        tarball_file_name: Some(artifact.file_name.clone()),
        python_launcher: None,
        python_executable: None,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(output, "  extracted: {}", install_root.display());
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(output)
}

fn resolve_pip_runtime(
    channel: TheRockChannel,
    family_override: Option<&str>,
) -> Result<PipRuntimeResolution> {
    let family_resolution = resolve_family(family_override)?;
    let index_url = therock_index_url(&family_resolution.family);
    let versions = load_simple_index_versions(&index_url, "rocm")?;
    let latest_version = select_latest_version(&versions, channel)
        .context("no TheRock ROCm wheel versions were discovered in the package index")?;
    Ok(PipRuntimeResolution {
        family: family_resolution.family,
        family_source: family_resolution.source,
        index_url,
        latest_version,
    })
}

fn resolve_tarball_artifact(
    channel: TheRockChannel,
    family_override: Option<&str>,
) -> Result<TarballArtifact> {
    let family_resolution = resolve_family(family_override)?;
    let html = download_text(channel.tarball_base_url())?;
    let files = parse_tarball_index_html(&html)?;
    let prefix = format!(
        "therock-dist-{}-{}-",
        platform_tarball_token(),
        family_resolution.family
    );
    let mut candidates = files
        .into_iter()
        .filter_map(|file| {
            let version = file
                .name
                .strip_prefix(&prefix)?
                .strip_suffix(".tar.gz")?
                .to_owned();
            Some((file, version))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.0
            .mtime
            .partial_cmp(&right.0.mtime)
            .unwrap_or(Ordering::Equal)
            .then_with(|| compare_version_strings(&left.1, &right.1))
    });
    let (file, version) = candidates
        .pop()
        .context("no matching TheRock tarball artifact was found for the resolved GPU family")?;
    Ok(TarballArtifact {
        family: family_resolution.family,
        family_source: family_resolution.source,
        url: format!(
            "{}/{}",
            channel.tarball_base_url().trim_end_matches('/'),
            file.name
        ),
        file_name: file.name,
        version,
    })
}

fn resolve_family(family_override: Option<&str>) -> Result<FamilyResolution> {
    if let Some(value) = family_override
        && let Some(family) = normalize_therock_family(value)
    {
        return Ok(FamilyResolution {
            family,
            source: "manifest".to_owned(),
        });
    }

    if let Some(value) = std::env::var("ROCM_CLI_THEROCK_FAMILY").ok()
        && let Some(family) = normalize_therock_family(&value)
    {
        return Ok(FamilyResolution {
            family,
            source: "env".to_owned(),
        });
    }

    if let Some(family) = detect_host_therock_family() {
        return Ok(FamilyResolution {
            family,
            source: "host".to_owned(),
        });
    }

    bail!("unable to resolve a supported TheRock GPU family for this host")
}

fn load_simple_index_versions(index_url: &str, package_name: &str) -> Result<Vec<String>> {
    let url = format!("{}/{package_name}/", index_url.trim_end_matches('/'));
    let html = download_text(&url)?;
    let marker = format!("{package_name}-");
    let mut versions = Vec::new();
    for line in html.lines() {
        let Some(start) = line.find(&marker) else {
            continue;
        };
        let version_start = start + marker.len();
        let Some(rest) = line.get(version_start..) else {
            continue;
        };
        let Some(version_end) = rest.find(".tar.gz").or_else(|| rest.find(".whl")) else {
            continue;
        };
        versions.push(rest[..version_end].to_owned());
    }
    versions.sort_by(|left, right| compare_version_strings(left, right));
    versions.dedup();
    Ok(versions)
}

fn select_latest_version(versions: &[String], channel: TheRockChannel) -> Option<String> {
    let mut stable = Vec::new();
    let mut all = versions.to_vec();
    all.sort_by(|left, right| compare_version_strings(left, right));
    for version in versions {
        if parse_version(version)
            .map(|parsed| parsed.stage == VersionStage::Stable)
            .unwrap_or(false)
        {
            stable.push(version.clone());
        }
    }
    stable.sort_by(|left, right| compare_version_strings(left, right));
    match channel {
        TheRockChannel::Release => stable.pop().or_else(|| all.pop()),
        TheRockChannel::Nightly => all.pop(),
    }
}

fn download_text(url: &str) -> Result<String> {
    let output = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .with_context(|| format!("failed to launch curl for {url}"))?;
    if !output.status.success() {
        bail!(
            "curl failed for {}: {}",
            url,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout)
        .with_context(|| format!("failed to decode response from {url}"))
}

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("download destination has no parent directory")?;
    fs::create_dir_all(parent)?;
    run_progress_command(
        Path::new("curl"),
        &[
            "-fL",
            "--progress-bar",
            "-o",
            destination.to_string_lossy().as_ref(),
            url,
        ],
        "download TheRock tarball artifact",
    )
}

fn extract_tarball(archive_path: &Path, target_dir: &Path) -> Result<()> {
    run_command(
        Path::new("tar"),
        &[
            "-xf",
            archive_path.to_string_lossy().as_ref(),
            "-C",
            target_dir.to_string_lossy().as_ref(),
        ],
        "extract TheRock tarball artifact",
    )
}

fn ensure_python_venv(python_launcher: &str, install_root: &Path) -> Result<()> {
    let env_python = venv_python_path(install_root);
    if env_python.is_file() {
        return Ok(());
    }
    run_command(
        Path::new(python_launcher),
        &["-m", "venv", install_root.to_string_lossy().as_ref()],
        "create managed TheRock runtime virtual environment",
    )
}

fn resolve_installed_rocm_version(python_executable: &Path) -> Result<Option<String>> {
    let output = Command::new(python_executable)
        .args(["-m", "pip", "show", "rocm"])
        .output()
        .with_context(|| {
            format!(
                "failed to inspect installed ROCm version via {}",
                python_executable.display()
            )
        })?;
    if !output.status.success() {
        return Ok(None);
    }
    let text = String::from_utf8(output.stdout)
        .context("failed to decode `pip show rocm` output for managed runtime")?;
    for line in text.lines() {
        if let Some(version) = line.strip_prefix("Version:") {
            return Ok(Some(version.trim().to_owned()));
        }
    }
    Ok(None)
}

fn run_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    if output.status.success() {
        return Ok(());
    }
    bail!(
        "{}: {}",
        context_text,
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn run_progress_command(program: &Path, args: &[&str], context_text: &str) -> Result<()> {
    if interactive_terminal() {
        let status = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| format!("failed to launch {}", program.display()))?;
        if status.success() {
            return Ok(());
        }
        bail!("{context_text}: command exited with status {status}");
    }
    run_command(program, args, context_text)
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

fn resolve_python_launcher() -> Result<String> {
    if let Some(value) = std::env::var("ROCM_CLI_PYTHON").ok()
        && command_succeeds(&value, &["--version"])
    {
        return Ok(value);
    }

    let candidates: &[&str] = if cfg!(windows) {
        &["python", "py"]
    } else {
        &["python3", "python"]
    };
    for candidate in candidates {
        if command_succeeds(candidate, &["--version"]) {
            return Ok((*candidate).to_owned());
        }
    }
    bail!("unable to locate a Python launcher; set ROCM_CLI_PYTHON or install python3/python")
}

fn command_succeeds(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn parse_tarball_index_html(html: &str) -> Result<Vec<TarballIndexFile>> {
    let start = html
        .find("const files = ")
        .context("tarball index did not contain the embedded file list")?;
    let json_start = start + "const files = ".len();
    let rest = &html[json_start..];
    let end = rest
        .find("];")
        .context("tarball index did not contain the end of the embedded file list")?;
    let json = format!("{}]", &rest[..end]);
    serde_json::from_str(&json).context("failed to parse TheRock tarball index file list")
}

fn compare_version_strings(left: &str, right: &str) -> Ordering {
    match (parse_version(left), parse_version(right)) {
        (Some(left), Some(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

fn parse_version(value: &str) -> Option<ParsedVersion> {
    let mut parts = value.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_rest = parts.next()?;

    let patch_len = patch_and_rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .count();
    if patch_len == 0 {
        return None;
    }
    let patch = patch_and_rest[..patch_len].parse().ok()?;
    let suffix = &patch_and_rest[patch_len..];

    let (stage, stage_number) = if suffix.is_empty() {
        (VersionStage::Stable, 0)
    } else if let Some(rest) = suffix.strip_prefix("rc") {
        (VersionStage::Rc, rest.parse().ok()?)
    } else if let Some(rest) = suffix.strip_prefix('a') {
        (VersionStage::Alpha, rest.parse().ok()?)
    } else {
        return None;
    };

    Some(ParsedVersion {
        major,
        minor,
        patch,
        stage,
        stage_number,
    })
}

fn therock_index_url(family: &str) -> String {
    format!("{THEROCK_PIP_INDEX_BASE}/{family}")
}

fn platform_tarball_token() -> &'static str {
    if cfg!(windows) { "windows" } else { "linux" }
}

fn runtime_key(channel: TheRockChannel, format: &str, family: &str) -> String {
    slugify(&format!("{}-{format}-{family}", channel.as_str()))
}

fn managed_runtime_root(paths: &AppPaths, format: &str, runtime_key: &str) -> PathBuf {
    paths
        .data_dir
        .join("runtimes")
        .join(format)
        .join(runtime_key)
}

fn runtime_registry_dir(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("runtimes").join("registry")
}

fn runtime_manifest_path(paths: &AppPaths, runtime_key: &str) -> PathBuf {
    runtime_registry_dir(paths).join(format!("{runtime_key}.json"))
}

fn save_runtime_manifest(paths: &AppPaths, manifest: &InstalledRuntimeManifest) -> Result<()> {
    let registry_path = runtime_manifest_path(paths, &manifest.runtime_key);
    fs::create_dir_all(
        registry_path
            .parent()
            .context("runtime manifest registry path has no parent directory")?,
    )?;
    fs::write(
        &registry_path,
        serde_json::to_vec_pretty(manifest).context("failed to serialize runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", registry_path.display()))?;

    let local_manifest_path = manifest.install_root.join(".rocm-cli-runtime.json");
    fs::write(
        &local_manifest_path,
        serde_json::to_vec_pretty(manifest)
            .context("failed to serialize local runtime manifest")?,
    )
    .with_context(|| format!("failed to write {}", local_manifest_path.display()))?;
    Ok(())
}

fn load_runtime_manifests(paths: &AppPaths) -> Result<Vec<InstalledRuntimeManifest>> {
    let registry_dir = runtime_registry_dir(paths);
    if !registry_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut manifests = Vec::new();
    for entry in fs::read_dir(&registry_dir)
        .with_context(|| format!("failed to read {}", registry_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if let Ok(manifest) = serde_json::from_slice::<InstalledRuntimeManifest>(&bytes) {
            manifests.push(manifest);
        }
    }
    manifests.sort_by(|left, right| right.installed_at_unix_ms.cmp(&left.installed_at_unix_ms));
    Ok(manifests)
}

fn has_nontrivial_directory_contents(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut entries =
        fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?;
    while let Some(entry) = entries.next() {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn venv_python_path(install_root: &Path) -> PathBuf {
    if cfg!(windows) {
        install_root.join("Scripts").join("python.exe")
    } else {
        install_root.join("bin").join("python")
    }
}

fn venv_bin_dir(install_root: &Path) -> PathBuf {
    if cfg!(windows) {
        install_root.join("Scripts")
    } else {
        install_root.join("bin")
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

    #[test]
    fn release_channel_prefers_stable_versions() {
        let versions = vec![
            "7.11.0".to_owned(),
            "7.12.0".to_owned(),
            "7.13.0a20260326".to_owned(),
        ];
        assert_eq!(
            select_latest_version(&versions, TheRockChannel::Release),
            Some("7.12.0".to_owned())
        );
    }
}
