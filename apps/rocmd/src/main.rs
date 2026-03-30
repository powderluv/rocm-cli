use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, AutomationEventRecord, AutomationRuntimeState, CodexBridgeEngine,
    CodexBridgeGpuSnapshot, CodexBridgeSnapshot, DEFAULT_LOCAL_HOST, DoctorSummary,
    ManagedServiceRecord, RocmCliConfig, WatcherMode, WatcherRuntimeSnapshot,
    append_automation_event, builtin_watchers, daemon_binary_path, default_engine_for_platform,
    engine_binary_path, load_recent_automation_events, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use tokio::time::{self, MissedTickBehavior};

const WATCHER_TICK_INTERVAL: Duration = Duration::from_secs(5);
const SERVER_RECOVER_BACKOFF_MS: u128 = 30_000;
const THEROCK_UPDATE_INTERVAL_MS: u128 = 6 * 60 * 60 * 1000;

#[derive(Parser, Debug)]
#[command(name = "rocmd", about = "rocm-cli local supervisor", version)]
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
    BridgeSnapshot {
        #[arg(long)]
        pretty: bool,
    },
    McpServer,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::discover()?;

    match cli.command.unwrap_or(Command::Status) {
        Command::Run {
            automations_enabled,
        } => run_daemon(&paths, automations_enabled).await?,
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
        Command::BridgeSnapshot { pretty } => {
            print_bridge_snapshot(&paths, pretty)?;
        }
        Command::McpServer => {
            run_mcp_server(&paths)?;
        }
    }

    Ok(())
}

fn print_bridge_snapshot(paths: &AppPaths, pretty: bool) -> Result<()> {
    let snapshot = build_bridge_snapshot(paths)?;

    if pretty {
        println!(
            "{}",
            serde_json::to_string_pretty(&snapshot)
                .context("failed to serialize bridge snapshot")?
        );
    } else {
        println!(
            "{}",
            serde_json::to_string(&snapshot).context("failed to serialize bridge snapshot")?
        );
    }

    Ok(())
}

fn build_bridge_snapshot(paths: &AppPaths) -> Result<CodexBridgeSnapshot> {
    Ok(CodexBridgeSnapshot {
        protocol: "rocmd-codex-bridge-v0".to_owned(),
        generated_at_unix_ms: unix_time_millis(),
        doctor: DoctorSummary::gather()?,
        gpu: gather_gpu_snapshot(),
        config: RocmCliConfig::load(paths).unwrap_or_default(),
        automation_runtime: AutomationRuntimeState::load(paths)?,
        recent_automation_events: load_recent_automation_events(paths, 32)?,
        engines: bridge_engine_inventory(),
        services: load_managed_services(paths)?,
    })
}

fn gather_gpu_snapshot() -> CodexBridgeGpuSnapshot {
    let static_snapshot = match capture_amd_smi_json(&["static", "-a", "-g", "all", "--json"]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: false,
                static_snapshot: None,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    let monitor_snapshot = match capture_amd_smi_json(&[
        "monitor", "-p", "-t", "-u", "-m", "-v", "-g", "all", "--json",
    ]) {
        Ok(value) => Some(value),
        Err(error) => {
            return CodexBridgeGpuSnapshot {
                amd_smi_available: true,
                static_snapshot,
                monitor_snapshot: None,
                note: Some(error.to_string()),
            };
        }
    };

    CodexBridgeGpuSnapshot {
        amd_smi_available: true,
        static_snapshot,
        monitor_snapshot,
        note: None,
    }
}

fn capture_amd_smi_json(args: &[&str]) -> Result<Value> {
    let output = ProcessCommand::new("amd-smi")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        anyhow::bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
}

fn bridge_engine_inventory() -> Vec<CodexBridgeEngine> {
    let default_engine = default_engine_for_platform();
    rocmd_engine_inventory()
        .iter()
        .map(|(id, summary)| {
            let binary_path = engine_binary_path(id).ok();
            CodexBridgeEngine {
                id: (*id).to_owned(),
                summary: (*summary).to_owned(),
                default_for_platform: *id == default_engine,
                installed_binary: binary_path.is_some(),
                binary_path: binary_path.map(|path| path.display().to_string()),
            }
        })
        .collect()
}

fn rocmd_engine_inventory() -> &'static [(&'static str, &'static str)] {
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

fn run_mcp_server(paths: &AppPaths) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let message: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {error}"),
                        }
                    }),
                )?;
                continue;
            }
        };

        let Some(method) = message.get("method").and_then(Value::as_str) else {
            if message.get("id").is_some() {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": message.get("id").cloned().unwrap_or(Value::Null),
                        "error": {
                            "code": -32600,
                            "message": "invalid request: missing method",
                        }
                    }),
                )?;
            }
            continue;
        };

        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "initialize" => {
                let protocol_version = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2025-03-26");
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "protocolVersion": protocol_version,
                                "capabilities": {
                                    "tools": {
                                        "listChanged": true,
                                    }
                                },
                                "serverInfo": {
                                    "name": "rocmd-mcp-server",
                                    "title": "ROCm AI Command Center",
                                    "version": env!("CARGO_PKG_VERSION"),
                                }
                            }
                        }),
                    )?;
                }
            }
            "ping" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {}
                        }),
                    )?;
                }
            }
            "notifications/initialized" => {}
            "tools/list" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "tools": rocm_mcp_tools(),
                                "nextCursor": Value::Null,
                            }
                        }),
                    )?;
                }
            }
            "tools/call" => {
                if let Some(id) = id {
                    let result = match handle_mcp_tool_call(paths, &params) {
                        Ok(result) => result,
                        Err(error) => tool_error(
                            format!("ROCm MCP tool call failed: {error:#}"),
                            json!({
                                "tool": params.get("name").cloned().unwrap_or(Value::Null),
                            }),
                        ),
                    };
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": result,
                        }),
                    )?;
                }
            }
            notification if notification.starts_with("notifications/") => {}
            other => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": format!("method not found: {other}"),
                            }
                        }),
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn write_json_line(writer: &mut impl Write, value: &Value) -> Result<()> {
    writer.write_all(serde_json::to_string(value)?.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn rocm_mcp_tools() -> Vec<Value> {
    vec![
        rocm_mcp_tool(
            "doctor",
            "Read the current ROCm AI Command Center host summary.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "bridge_snapshot",
            "Read the full ROCm bridge snapshot including doctor data, engines, services, automations, and gpu telemetry.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "gpu_snapshot",
            "Read the current amd-smi GPU telemetry snapshot if available.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "engines",
            "List available ROCm serving engines and whether each one is installed.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "services",
            "List managed model services and their current status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "service_logs",
            "Read the tail of a managed service log file.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "automations",
            "List automation runtime status and recent watcher events.",
            json!({
                "type": "object",
                "properties": {
                    "event_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "natural_language_plan",
            "Ask `rocm` to translate a natural-language ROCm request into a visible plan without executing privileged work.",
            json!({
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string"
                    }
                },
                "required": ["request"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "update_check",
            "Run `rocm update` and return the current TheRock update status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk_dry_run",
            "Run a dry-run TheRock SDK install plan.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pip", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk",
            "Install a TheRock SDK into the managed runtime area or an explicitly approved prefix.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pip", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "allow_system_prefix": {
                        "type": "boolean"
                    }
                },
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "install_engine",
            "Install or refresh a managed serving engine environment.",
            json!({
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "python_version": {
                        "type": "string"
                    },
                    "reinstall": {
                        "type": "boolean"
                    }
                },
                "required": ["engine"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "launch_server",
            "Launch a managed local model server through `rocm serve --managed`.",
            json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string"
                    },
                    "engine": {
                        "type": "string"
                    },
                    "device": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "env_id": {
                        "type": "string"
                    },
                    "host": {
                        "type": "string"
                    },
                    "port": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 65535
                    },
                    "allow_public_bind": {
                        "type": "boolean"
                    }
                },
                "required": ["model"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "stop_server",
            "Stop a managed service by service id and update its manifest status.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "watcher_enable",
            "Enable a watcher and optionally set its mode.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["observe", "propose", "contained"]
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "watcher_disable",
            "Disable a watcher.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
    ]
}

fn rocm_mcp_tool(
    name: &str,
    description: &str,
    input_schema: Value,
    read_only: bool,
    destructive: bool,
) -> Value {
    json!({
        "name": name,
        "title": name.replace('_', " "),
        "description": description,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "openWorldHint": false,
        },
        "inputSchema": input_schema,
    })
}

fn handle_mcp_tool_call(paths: &AppPaths, params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = params
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    match name {
        "doctor" => {
            let doctor = DoctorSummary::gather()?;
            Ok(tool_success(doctor.render_text(), json!(doctor)))
        }
        "bridge_snapshot" => {
            let snapshot = build_bridge_snapshot(paths)?;
            Ok(tool_success(
                format!(
                    "Captured bridge snapshot for {} / {} with default engine `{}`.",
                    snapshot.doctor.os, snapshot.doctor.arch, snapshot.doctor.default_engine
                ),
                json!(snapshot),
            ))
        }
        "gpu_snapshot" => {
            let gpu = gather_gpu_snapshot();
            let status = if gpu.amd_smi_available {
                "Captured amd-smi GPU snapshot."
            } else {
                "amd-smi is unavailable on this host."
            };
            Ok(tool_success(status.to_owned(), json!(gpu)))
        }
        "engines" => {
            let engines = bridge_engine_inventory();
            Ok(tool_success(
                format!("Found {} engine entries.", engines.len()),
                json!({ "engines": engines }),
            ))
        }
        "services" => {
            let services = load_managed_services(paths)?;
            Ok(tool_success(
                format!("Found {} managed services.", services.len()),
                json!({ "services": services }),
            ))
        }
        "service_logs" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("service_logs requires `service_id`")?;
            let lines = arguments
                .get("lines")
                .and_then(Value::as_u64)
                .unwrap_or(80)
                .clamp(1, 500) as usize;
            let record = load_managed_services(paths)?
                .into_iter()
                .find(|service| service.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            let tail = read_tail_lines(&record.log_path, lines)?;
            Ok(tool_success(
                format!(
                    "Read the last {} line(s) from service `{}`.",
                    lines, record.service_id
                ),
                json!({
                    "service": record,
                    "lines": lines,
                    "tail": tail,
                }),
            ))
        }
        "automations" => {
            let event_limit = arguments
                .get("event_limit")
                .and_then(Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 64) as usize;
            let runtime = AutomationRuntimeState::load(paths)?;
            let events = load_recent_automation_events(paths, event_limit)?;
            Ok(tool_success(
                format!(
                    "Loaded automation runtime and {} recent events.",
                    events.len()
                ),
                json!({
                    "runtime": runtime,
                    "recent_events": events,
                }),
            ))
        }
        "natural_language_plan" => {
            let request = arguments
                .get("request")
                .and_then(Value::as_str)
                .context("natural_language_plan requires `request`")?;
            let output = run_rocm_capture(&[request])?;
            Ok(tool_result_from_command(
                "Ran natural-language planning through `rocm`.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output = run_rocm_capture(&["update"])?;
            Ok(tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let argv = build_install_sdk_args(&arguments, true)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "install_sdk" => {
            let argv = build_install_sdk_args(&arguments, false)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk`.",
                output,
                false,
            ))
        }
        "install_engine" => {
            let argv = build_install_engine_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm engines install`.",
                output,
                false,
            ))
        }
        "launch_server" => {
            let argv = build_launch_server_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm serve --managed`.",
                output,
                false,
            ))
        }
        "stop_server" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("stop_server requires `service_id`")?;
            let stopped = stop_managed_service(paths, service_id)?;
            Ok(tool_success(
                format!("Stopped managed service `{service_id}`."),
                stopped,
            ))
        }
        "watcher_enable" => {
            let argv = build_watcher_enable_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations enable`.",
                output,
                false,
            ))
        }
        "watcher_disable" => {
            let watcher = arguments
                .get("watcher")
                .and_then(Value::as_str)
                .context("watcher_disable requires `watcher`")?;
            let argv = vec![
                "automations".to_owned(),
                "disable".to_owned(),
                watcher.to_owned(),
            ];
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations disable`.",
                output,
                false,
            ))
        }
        other => Ok(tool_error(
            format!("Unknown ROCm MCP tool `{other}`."),
            json!({ "tool": other }),
        )),
    }
}

fn tool_success(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": false,
    })
}

fn tool_error(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": true,
    })
}

fn tool_result_from_command(prefix: &str, output: CommandCapture, is_error: bool) -> Value {
    let text = if output.stderr.trim().is_empty() {
        format!("{prefix}\n\n{}", output.stdout.trim())
    } else if output.stdout.trim().is_empty() {
        format!("{prefix}\n\nstderr:\n{}", output.stderr.trim())
    } else {
        format!(
            "{prefix}\n\nstdout:\n{}\n\nstderr:\n{}",
            output.stdout.trim(),
            output.stderr.trim()
        )
    };
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": {
            "argv": output.argv,
            "exit_status": output.exit_status,
            "stdout": output.stdout,
            "stderr": output.stderr,
        },
        "isError": is_error || output.exit_status != 0,
    })
}

#[derive(Debug)]
struct CommandCapture {
    argv: Vec<String>,
    exit_status: i32,
    stdout: String,
    stderr: String,
}

fn run_rocm_capture(args: &[&str]) -> Result<CommandCapture> {
    let rocm_binary = rocm_core::sibling_binary_path("rocm")?;
    let output = ProcessCommand::new(&rocm_binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to launch {}", rocm_binary.display()))?;
    Ok(CommandCapture {
        argv: std::iter::once(rocm_binary.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        exit_status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn read_tail_lines(path: &std::path::Path, limit: usize) -> Result<String> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = VecDeque::with_capacity(limit);
    for line in content.lines() {
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line.to_owned());
    }
    Ok(lines.into_iter().collect::<Vec<_>>().join("\n"))
}

fn build_install_sdk_args(
    arguments: &serde_json::Map<String, Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = arguments
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or("release");
    let format = arguments
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("pip");
    let prefix = arguments.get("prefix").and_then(Value::as_str);
    let allow_system_prefix = arguments
        .get("allow_system_prefix")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel.to_owned(),
        "--format".to_owned(),
        format.to_owned(),
    ];
    if let Some(prefix) = prefix {
        let prefix_path = std::path::Path::new(prefix);
        if system_prefix_requires_ack(prefix_path) && !allow_system_prefix {
            bail!(
                "install_sdk prefix `{}` is outside the user home; require `allow_system_prefix=true` before using system paths",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix.to_owned());
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    }
    Ok(argv)
}

fn build_install_engine_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let engine = arguments
        .get("engine")
        .and_then(Value::as_str)
        .context("install_engine requires `engine`")?;
    let runtime_id = arguments
        .get("runtime_id")
        .and_then(Value::as_str)
        .unwrap_or("therock-release");
    let python_version = arguments.get("python_version").and_then(Value::as_str);
    let reinstall = arguments
        .get("reinstall")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut argv = vec![
        "engines".to_owned(),
        "install".to_owned(),
        engine.to_owned(),
        "--runtime-id".to_owned(),
        runtime_id.to_owned(),
    ];
    if let Some(python_version) = python_version {
        argv.push("--python-version".to_owned());
        argv.push(python_version.to_owned());
    }
    if reinstall {
        argv.push("--reinstall".to_owned());
    }
    Ok(argv)
}

fn build_launch_server_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let model = arguments
        .get("model")
        .and_then(Value::as_str)
        .context("launch_server requires `model`")?;
    let host = arguments
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_LOCAL_HOST);
    let allow_public_bind = arguments
        .get("allow_public_bind")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "launch_server host `{host}` is not loopback; require `allow_public_bind=true` before binding a non-local interface"
        );
    }

    let mut argv = vec!["serve".to_owned(), model.to_owned(), "--managed".to_owned()];
    if let Some(engine) = arguments.get("engine").and_then(Value::as_str) {
        argv.push("--engine".to_owned());
        argv.push(engine.to_owned());
    }
    if let Some(device) = arguments.get("device").and_then(Value::as_str) {
        argv.push("--device".to_owned());
        argv.push(device.to_owned());
    }
    if let Some(runtime_id) = arguments.get("runtime_id").and_then(Value::as_str) {
        argv.push("--runtime-id".to_owned());
        argv.push(runtime_id.to_owned());
    }
    if let Some(env_id) = arguments.get("env_id").and_then(Value::as_str) {
        argv.push("--env-id".to_owned());
        argv.push(env_id.to_owned());
    }
    argv.push("--host".to_owned());
    argv.push(host.to_owned());
    if let Some(port) = arguments.get("port").and_then(Value::as_u64) {
        argv.push("--port".to_owned());
        argv.push(port.to_string());
    }
    Ok(argv)
}

fn build_watcher_enable_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let watcher = arguments
        .get("watcher")
        .and_then(Value::as_str)
        .context("watcher_enable requires `watcher`")?;
    let mut argv = vec![
        "automations".to_owned(),
        "enable".to_owned(),
        watcher.to_owned(),
    ];
    if let Some(mode) = arguments.get("mode").and_then(Value::as_str) {
        argv.push("--mode".to_owned());
        argv.push(mode.to_owned());
    }
    Ok(argv)
}

fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn system_prefix_requires_ack(prefix: &std::path::Path) -> bool {
    let home = if cfg!(windows) {
        std::env::var_os("USERPROFILE")
    } else {
        std::env::var_os("HOME")
    }
    .map(std::path::PathBuf::from);
    match home {
        Some(home) => !prefix.starts_with(home),
        None => true,
    }
}

fn stop_managed_service(paths: &AppPaths, service_id: &str) -> Result<Value> {
    let mut record = load_managed_services(paths)?
        .into_iter()
        .find(|record| record.service_id == service_id)
        .with_context(|| format!("managed service `{service_id}` not found"))?;
    let mut signaled_pids = Vec::new();
    let mut skipped_pids = Vec::new();
    if let Some(engine_pid) = record.engine_pid {
        match terminate_process(engine_pid)? {
            true => signaled_pids.push(engine_pid),
            false => skipped_pids.push(engine_pid),
        }
    }
    if record.supervisor_pid != 0
        && record.supervisor_pid != std::process::id()
        && Some(record.supervisor_pid) != record.engine_pid
    {
        match terminate_process(record.supervisor_pid)? {
            true => signaled_pids.push(record.supervisor_pid),
            false => skipped_pids.push(record.supervisor_pid),
        }
    }
    record.status = "stopped".to_owned();
    record.write()?;
    Ok(json!({
        "service": record,
        "signaled_pids": signaled_pids,
        "skipped_pids": skipped_pids,
    }))
}

fn terminate_process(pid: u32) -> Result<bool> {
    if pid == std::process::id() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        let output = ProcessCommand::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch kill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "failed to signal pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
    #[cfg(windows)]
    {
        let output = ProcessCommand::new("taskkill")
            .arg("/PID")
            .arg(pid.to_string())
            .arg("/T")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to launch taskkill for pid {pid}"))?;
        if output.status.success() {
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "failed to stop pid {pid}: {}",
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    format!("exit status {}", output.status)
                }
            )
        }
    }
}

async fn run_daemon(paths: &AppPaths, automations_enabled: bool) -> Result<()> {
    let config = RocmCliConfig::load(paths)?;
    let mut state = build_runtime_state(&config, automations_enabled);

    println!("rocmd run");
    println!("  automations enabled: {automations_enabled}");
    println!(
        "  lifecycle: {}",
        if automations_enabled {
            "persistent"
        } else {
            "on-demand"
        }
    );
    println!("  config: {}", paths.config_path().display());
    println!("  state: {}", paths.automation_state_path().display());
    let enabled_count = state
        .active_watchers
        .iter()
        .filter(|watcher| watcher.enabled)
        .count();
    println!("  enabled watchers: {enabled_count}");

    if !automations_enabled {
        println!(
            "  note: rerun with --automations-enabled to keep rocmd alive for watcher execution"
        );
        return Ok(());
    }

    paths.ensure()?;
    state.write(paths)?;
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_start",
        "rocmd automation supervisor started",
        None,
    )?;
    state.write(paths)?;

    evaluate_watchers(paths, &config, &mut state)?;
    state.last_tick_unix_ms = unix_time_millis();
    state.write(paths)?;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    let mut ticker = time::interval(WATCHER_TICK_INTERVAL);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let config = RocmCliConfig::load(paths)?;
                reconcile_watcher_snapshots(&config, &mut state);
                evaluate_watchers(paths, &config, &mut state)?;
                state.last_tick_unix_ms = unix_time_millis();
                state.write(paths)?;
            }
            _ = &mut shutdown => {
                break;
            }
        }
    }

    state.running = false;
    state.last_tick_unix_ms = unix_time_millis();
    record_event(
        paths,
        &mut state,
        "rocmd",
        "info",
        "daemon_stop",
        "rocmd automation supervisor stopped",
        None,
    )?;
    state.write(paths)?;
    Ok(())
}

fn print_status(paths: &AppPaths) -> Result<()> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    println!("rocmd status");
    println!("  config dir: {}", paths.config_dir.display());
    println!("  data dir: {}", paths.data_dir.display());
    println!("  policy: on-demand by default, persistent only with background features");
    println!(
        "  automations desired: {}",
        if config.automation_daemon_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    match AutomationRuntimeState::load(paths)? {
        Some(state) => {
            println!(
                "  automations runtime: {} pid={} last_tick_unix_ms={}",
                if state.running { "running" } else { "stopped" },
                state.daemon_pid,
                state.last_tick_unix_ms
            );
            for watcher in state
                .active_watchers
                .into_iter()
                .filter(|watcher| watcher.enabled)
            {
                println!(
                    "  watcher {} mode={} last_event={}",
                    watcher.id,
                    watcher.mode.as_str(),
                    watcher.last_event.as_deref().unwrap_or("<none>")
                );
            }
        }
        None => println!("  automations runtime: inactive"),
    }
    println!(
        "  automation events: {}",
        paths.automation_events_path().display()
    );

    let records = load_managed_services(paths)?;
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
        runtime_id.clone(),
        env_id.clone(),
        Some(device_policy.clone()),
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

fn build_runtime_state(
    config: &RocmCliConfig,
    automations_enabled: bool,
) -> AutomationRuntimeState {
    let now = unix_time_millis();
    let active_watchers = builtin_watchers()
        .iter()
        .map(|watcher| WatcherRuntimeSnapshot {
            id: watcher.id.to_owned(),
            enabled: config.watcher_enabled(watcher),
            mode: config.effective_watcher_mode(watcher),
            summary: watcher.summary.to_owned(),
            last_event: None,
            last_event_unix_ms: None,
        })
        .collect();
    AutomationRuntimeState {
        running: automations_enabled,
        automations_enabled,
        daemon_pid: std::process::id(),
        started_at_unix_ms: now,
        last_tick_unix_ms: now,
        active_watchers,
    }
}

fn reconcile_watcher_snapshots(config: &RocmCliConfig, state: &mut AutomationRuntimeState) {
    for watcher in builtin_watchers() {
        match state.watcher_mut(watcher.id) {
            Some(snapshot) => {
                snapshot.enabled = config.watcher_enabled(watcher);
                snapshot.mode = config.effective_watcher_mode(watcher);
                snapshot.summary = watcher.summary.to_owned();
            }
            None => state.active_watchers.push(WatcherRuntimeSnapshot {
                id: watcher.id.to_owned(),
                enabled: config.watcher_enabled(watcher),
                mode: config.effective_watcher_mode(watcher),
                summary: watcher.summary.to_owned(),
                last_event: None,
                last_event_unix_ms: None,
            }),
        }
    }
}

fn evaluate_watchers(
    paths: &AppPaths,
    config: &RocmCliConfig,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    for watcher in builtin_watchers() {
        if !config.watcher_enabled(watcher) {
            continue;
        }
        match watcher.id {
            "therock-update" => {
                evaluate_therock_update(paths, config.effective_watcher_mode(watcher), state)?
            }
            "server-recover" => {
                evaluate_server_recover(paths, config.effective_watcher_mode(watcher), state)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn evaluate_therock_update(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let last_event_unix_ms = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "therock-update")
        .and_then(|watcher| watcher.last_event_unix_ms);
    let now = unix_time_millis();
    if let Some(last_event) = last_event_unix_ms
        && now.saturating_sub(last_event) < THEROCK_UPDATE_INTERVAL_MS
    {
        return Ok(());
    }

    let action = match mode {
        WatcherMode::Observe => "observe_schedule",
        WatcherMode::Propose => "queue_update_proposal",
        WatcherMode::Contained => "queue_update_proposal",
    };
    let message = if mode == WatcherMode::Contained {
        "scheduled TheRock update check reminder emitted; contained mode falls back to proposal until SDK updates are fully wired"
    } else {
        "scheduled TheRock update check reminder emitted; run `rocm update` to inspect the selected channel"
    };
    record_event(
        paths,
        state,
        "therock-update",
        "info",
        action,
        message,
        None,
    )
}

fn evaluate_server_recover(
    paths: &AppPaths,
    mode: WatcherMode,
    state: &mut AutomationRuntimeState,
) -> Result<()> {
    let last_event_unix_ms = state
        .active_watchers
        .iter()
        .find(|watcher| watcher.id == "server-recover")
        .and_then(|watcher| watcher.last_event_unix_ms);
    let now = unix_time_millis();
    if let Some(last_event) = last_event_unix_ms
        && now.saturating_sub(last_event) < SERVER_RECOVER_BACKOFF_MS
    {
        return Ok(());
    }

    let Some(mut record) = load_managed_services(paths)?
        .into_iter()
        .find(|record| record.mode == "managed" && record.status == "failed")
    else {
        return Ok(());
    };

    match mode {
        WatcherMode::Observe => record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "observe_failure",
            &format!(
                "observed failed managed service {}; restart not attempted in observe mode",
                record.service_id
            ),
            Some(record.service_id.clone()),
        ),
        WatcherMode::Propose => record_event(
            paths,
            state,
            "server-recover",
            "warn",
            "queue_restart_proposal",
            &format!(
                "managed service {} failed; queueing restart proposal",
                record.service_id
            ),
            Some(record.service_id.clone()),
        ),
        WatcherMode::Contained => {
            if let Some(last_restart) = record.last_restart_unix_ms
                && now.saturating_sub(last_restart) < SERVER_RECOVER_BACKOFF_MS
            {
                return Ok(());
            }
            restart_managed_service(paths, &mut record)?;
            record_event(
                paths,
                state,
                "server-recover",
                "info",
                "restart_managed_service",
                &format!(
                    "restarted failed managed service {} on {}:{}",
                    record.service_id, record.host, record.port
                ),
                Some(record.service_id.clone()),
            )
        }
    }
}

fn restart_managed_service(_paths: &AppPaths, record: &mut ManagedServiceRecord) -> Result<()> {
    let rocmd_binary =
        std::env::current_exe().context("failed to resolve current rocmd executable path")?;
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    let log_file_err = log_file
        .try_clone()
        .context("failed to clone service log file handle")?;

    record.status = "recovering".to_owned();
    record.restart_count = record.restart_count.saturating_add(1);
    record.last_restart_unix_ms = Some(unix_time_millis());
    record.supervisor_pid = std::process::id();
    record.write()?;

    let mut child = detached_rocmd_command(&rocmd_binary)
        .arg("supervise")
        .arg(&record.service_id)
        .arg("--engine")
        .arg(&record.engine)
        .arg("--model-ref")
        .arg(&record.model_ref)
        .arg("--canonical-model-id")
        .arg(&record.canonical_model_id)
        .arg("--host")
        .arg(&record.host)
        .arg("--port")
        .arg(record.port.to_string())
        .arg("--device-policy")
        .arg(record.device_policy.as_deref().unwrap_or("gpu_preferred"))
        .args(optional_arg("--runtime-id", record.runtime_id.as_deref()))
        .args(optional_arg("--env-id", record.env_id.as_deref()))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .context("failed to spawn recovery supervisor")?;

    record.supervisor_pid = child.id();
    record.write()?;

    thread::sleep(Duration::from_millis(200));
    if let Some(status) = child
        .try_wait()
        .context("failed to check recovery supervisor startup state")?
    {
        record.status = "failed".to_owned();
        record.write()?;
        anyhow::bail!(
            "recovery supervisor exited immediately with status {status}; inspect {}",
            record.log_path.display()
        );
    }

    Ok(())
}

fn record_event(
    paths: &AppPaths,
    state: &mut AutomationRuntimeState,
    watcher_id: &str,
    level: &str,
    action: &str,
    message: &str,
    service_id: Option<String>,
) -> Result<()> {
    let now = unix_time_millis();
    if let Some(snapshot) = state.watcher_mut(watcher_id) {
        snapshot.last_event = Some(message.to_owned());
        snapshot.last_event_unix_ms = Some(now);
    }
    append_automation_event(
        paths,
        &AutomationEventRecord {
            at_unix_ms: now,
            watcher_id: watcher_id.to_owned(),
            level: level.to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            service_id,
        },
    )
}

fn load_managed_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_mcp_tools_include_bridge_gaps() {
        let names = rocm_mcp_tools()
            .into_iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
            .collect::<Vec<_>>();
        assert!(names.contains(&"gpu_snapshot".to_owned()));
        assert!(names.contains(&"service_logs".to_owned()));
        assert!(names.contains(&"natural_language_plan".to_owned()));
        assert!(names.contains(&"install_sdk".to_owned()));
        assert!(names.contains(&"install_engine".to_owned()));
        assert!(names.contains(&"launch_server".to_owned()));
        assert!(names.contains(&"stop_server".to_owned()));
        assert!(names.contains(&"watcher_enable".to_owned()));
        assert!(names.contains(&"watcher_disable".to_owned()));
    }

    #[test]
    fn read_tail_lines_returns_last_lines_only() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocmd-tail-test-{}-{}.log",
            std::process::id(),
            unix_time_millis()
        ));
        fs::write(&path, "line1\nline2\nline3\nline4\n")?;
        let tail = read_tail_lines(&path, 2)?;
        fs::remove_file(&path)?;
        assert_eq!(tail, "line3\nline4");
        Ok(())
    }

    #[test]
    fn launch_server_rejects_public_bind_without_ack() {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
        ]);
        let error = build_launch_server_args(&arguments).unwrap_err();
        assert!(
            error.to_string().contains("allow_public_bind=true"),
            "{error:#}"
        );
    }

    #[test]
    fn install_sdk_rejects_system_prefix_without_ack() {
        let arguments = serde_json::Map::from_iter([(
            "prefix".to_owned(),
            Value::String("/opt/rocm".to_owned()),
        )]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    #[test]
    fn watcher_enable_builds_mode_args() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            (
                "watcher".to_owned(),
                Value::String("server-recover".to_owned()),
            ),
            ("mode".to_owned(), Value::String("contained".to_owned())),
        ]);
        let argv = build_watcher_enable_args(&arguments)?;
        assert_eq!(
            argv,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "contained".to_owned()
            ]
        );
        Ok(())
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
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
