use crate::{
    render_automations_text, render_chat_text, render_config_text, render_daemon_text,
    render_doctor_text, render_engine_inventory_text, render_freeform_plan, render_logs_text,
    render_services_text, render_sidebar_text, render_uninstall_dry_run, render_update_text,
    tui_help_text,
};
use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use rocm_core::{AppPaths, RocmCliConfig};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{self, Stdout};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant};

const GPU_MONITOR_REFRESH_INTERVAL: Duration = Duration::from_secs(1);
const GPU_STATIC_REFRESH_INTERVAL: Duration = Duration::from_secs(300);
const GPU_ERROR_RETRY_INTERVAL: Duration = Duration::from_secs(5);

pub fn run(initial_provider: Option<String>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut app = App::new(paths, initial_provider);

    enable_raw_mode().context("failed to enable raw mode for rocm TUI")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to initialize terminal backend")?;

    let result = run_loop(&mut terminal, &mut app);

    disable_raw_mode().context("failed to disable raw mode for rocm TUI")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen")?;
    terminal.show_cursor().context("failed to restore cursor")?;

    result
}

struct App {
    paths: AppPaths,
    config: RocmCliConfig,
    provider: String,
    gpu_telemetry: GpuTelemetry,
    transcript: Vec<String>,
    transcript_scroll: u16,
    input: String,
    history: Vec<String>,
    history_index: Option<usize>,
    status: String,
    should_quit: bool,
}

struct CommandOutput {
    ok: bool,
    rendered: String,
}

#[derive(Default)]
struct GpuTelemetry {
    static_cards: BTreeMap<u32, GpuStaticCard>,
    monitor_cards: BTreeMap<u32, GpuMonitorCard>,
    last_static_refresh: Option<Instant>,
    last_monitor_refresh: Option<Instant>,
    last_error: Option<String>,
    last_error_at: Option<Instant>,
}

#[derive(Debug, Clone)]
struct GpuStaticCard {
    market_name: Option<String>,
    gfx_target: Option<String>,
    compute_units: Option<u32>,
}

#[derive(Debug, Clone)]
struct GpuMonitorCard {
    power_watts: Option<f64>,
    max_power_watts: Option<f64>,
    hotspot_celsius: Option<f64>,
    memory_celsius: Option<f64>,
    gfx_mhz: Option<f64>,
    gfx_percent: Option<f64>,
    mem_percent: Option<f64>,
    mem_mhz: Option<f64>,
    vram_used_mb: Option<f64>,
    vram_total_mb: Option<f64>,
    vram_percent: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct AmdSmiStaticResponse {
    #[serde(default)]
    gpu_data: Vec<AmdSmiStaticEntry>,
}

#[derive(Debug, Deserialize)]
struct AmdSmiStaticEntry {
    gpu: u32,
    #[serde(default)]
    asic: Option<AmdSmiAsic>,
}

#[derive(Debug, Deserialize)]
struct AmdSmiAsic {
    #[serde(default)]
    market_name: Option<String>,
    #[serde(default)]
    num_compute_units: Option<u32>,
    #[serde(default)]
    target_graphics_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AmdSmiMetricValue {
    #[serde(default)]
    value: Value,
}

#[derive(Debug, Deserialize)]
struct AmdSmiMonitorEntry {
    gpu: u32,
    #[serde(default)]
    power_usage: Option<AmdSmiMetricValue>,
    #[serde(default)]
    max_power: Option<AmdSmiMetricValue>,
    #[serde(default)]
    hotspot_temperature: Option<AmdSmiMetricValue>,
    #[serde(default)]
    memory_temperature: Option<AmdSmiMetricValue>,
    #[serde(default)]
    gfx_clk: Option<AmdSmiMetricValue>,
    #[serde(default)]
    gfx: Option<AmdSmiMetricValue>,
    #[serde(default)]
    mem: Option<AmdSmiMetricValue>,
    #[serde(default)]
    mem_clock: Option<AmdSmiMetricValue>,
    #[serde(default)]
    vram_used: Option<AmdSmiMetricValue>,
    #[serde(default)]
    vram_total: Option<AmdSmiMetricValue>,
    #[serde(default)]
    vram_percent: Option<AmdSmiMetricValue>,
}

impl AmdSmiMetricValue {
    fn as_f64(&self) -> Option<f64> {
        match &self.value {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => text.trim().parse::<f64>().ok(),
            _ => None,
        }
    }
}

impl GpuTelemetry {
    fn maybe_refresh(&mut self) {
        let now = Instant::now();
        if let Some(last_error_at) = self.last_error_at
            && now.duration_since(last_error_at) < GPU_ERROR_RETRY_INTERVAL
        {
            return;
        }

        let static_due = self.static_cards.is_empty()
            || self
                .last_static_refresh
                .is_none_or(|last| now.duration_since(last) >= GPU_STATIC_REFRESH_INTERVAL);
        if static_due {
            match load_gpu_static_cards() {
                Ok(cards) => {
                    self.static_cards = cards;
                    self.last_static_refresh = Some(now);
                    self.clear_error();
                }
                Err(error) => self.record_error(format!("static: {error}")),
            }
        }

        let monitor_due = self.monitor_cards.is_empty()
            || self
                .last_monitor_refresh
                .is_none_or(|last| now.duration_since(last) >= GPU_MONITOR_REFRESH_INTERVAL);
        if monitor_due {
            match load_gpu_monitor_cards() {
                Ok(cards) => {
                    self.monitor_cards = cards;
                    self.last_monitor_refresh = Some(now);
                    self.clear_error();
                }
                Err(error) => self.record_error(format!("monitor: {error}")),
            }
        }
    }

    fn force_refresh(&mut self) {
        self.last_static_refresh = None;
        self.last_monitor_refresh = None;
        self.last_error_at = None;
        self.maybe_refresh();
    }

    fn sidebar_text(&self) -> String {
        let mut output = String::new();
        use std::fmt::Write as _;

        let _ = writeln!(output, "gpu telemetry:");
        if self.static_cards.is_empty() && self.monitor_cards.is_empty() {
            let _ = writeln!(output, "  status: probing amd-smi");
            if let Some(error) = &self.last_error {
                let _ = writeln!(output, "  last error: {error}");
            }
            return output.trim_end().to_owned();
        }

        let _ = writeln!(
            output,
            "  gpus: {}",
            self.monitor_cards.len().max(self.static_cards.len())
        );
        if let Some(model) = self.primary_model_name() {
            let _ = writeln!(output, "  model: {model}");
        }
        if let Some(gfx) = self.primary_gfx_target() {
            let _ = writeln!(output, "  gfx: {gfx}");
        }
        if let Some(age) = self.last_monitor_refresh.map(|value| value.elapsed()) {
            let _ = writeln!(output, "  updated: {}", format_elapsed(age));
        }
        if let Some(summary) = self.aggregate_line() {
            let _ = writeln!(output, "  {summary}");
        }
        for line in self.per_gpu_lines() {
            let _ = writeln!(output, "  {line}");
        }
        if let Some(error) = &self.last_error {
            let _ = writeln!(output, "  note: {error}");
        }
        output.trim_end().to_owned()
    }

    fn detail_text(&self) -> String {
        let mut output = String::new();
        use std::fmt::Write as _;

        let _ = writeln!(output, "amd-smi telemetry");
        if self.static_cards.is_empty() && self.monitor_cards.is_empty() {
            let _ = writeln!(output, "  status: no telemetry yet");
            if let Some(error) = &self.last_error {
                let _ = writeln!(output, "  last error: {error}");
            }
            return output;
        }

        if let Some(age) = self.last_monitor_refresh.map(|value| value.elapsed()) {
            let _ = writeln!(output, "  updated: {}", format_elapsed(age));
        }
        if let Some(summary) = self.aggregate_line() {
            let _ = writeln!(output, "  summary: {summary}");
        }
        for gpu in self.all_gpu_ids() {
            let static_card = self.static_cards.get(&gpu);
            let monitor_card = self.monitor_cards.get(&gpu);
            let _ = writeln!(
                output,
                "  gpu {} model={} gfx={} cu={} gfx_util={} mem_util={} power={} hot={} mem_temp={} gfx_clk={} mem_clk={} vram={}",
                gpu,
                static_card
                    .and_then(|card| card.market_name.as_deref())
                    .unwrap_or("<unknown>"),
                static_card
                    .and_then(|card| card.gfx_target.as_deref())
                    .unwrap_or("<unknown>"),
                static_card
                    .and_then(|card| card.compute_units)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "<unknown>".to_owned()),
                format_optional_percent(monitor_card.and_then(|card| card.gfx_percent)),
                format_optional_percent(monitor_card.and_then(|card| card.mem_percent)),
                format_optional_watts(monitor_card.and_then(|card| card.power_watts)),
                format_optional_celsius(monitor_card.and_then(|card| card.hotspot_celsius)),
                format_optional_celsius(monitor_card.and_then(|card| card.memory_celsius)),
                format_optional_mhz(monitor_card.and_then(|card| card.gfx_mhz)),
                format_optional_mhz(monitor_card.and_then(|card| card.mem_mhz)),
                format_vram_line(monitor_card),
            );
        }
        if let Some(error) = &self.last_error {
            let _ = writeln!(output, "  note: {error}");
        }
        output
    }

    fn clear_error(&mut self) {
        self.last_error = None;
        self.last_error_at = None;
    }

    fn record_error(&mut self, message: String) {
        self.last_error = Some(message);
        self.last_error_at = Some(Instant::now());
    }

    fn primary_model_name(&self) -> Option<&str> {
        let first = self
            .static_cards
            .values()
            .find_map(|card| card.market_name.as_deref())?;
        if self
            .static_cards
            .values()
            .all(|card| card.market_name.as_deref() == Some(first))
        {
            Some(first)
        } else {
            Some("mixed")
        }
    }

    fn primary_gfx_target(&self) -> Option<&str> {
        let first = self
            .static_cards
            .values()
            .find_map(|card| card.gfx_target.as_deref())?;
        if self
            .static_cards
            .values()
            .all(|card| card.gfx_target.as_deref() == Some(first))
        {
            Some(first)
        } else {
            Some("mixed")
        }
    }

    fn aggregate_line(&self) -> Option<String> {
        if self.monitor_cards.is_empty() {
            return None;
        }

        let avg_gfx = average(
            self.monitor_cards
                .values()
                .filter_map(|card| card.gfx_percent),
        );
        let avg_mem = average(
            self.monitor_cards
                .values()
                .filter_map(|card| card.mem_percent),
        );
        let avg_hot = average(
            self.monitor_cards
                .values()
                .filter_map(|card| card.hotspot_celsius),
        );
        let total_power = self
            .monitor_cards
            .values()
            .filter_map(|card| card.power_watts)
            .sum::<f64>();
        let total_max_power = self
            .monitor_cards
            .values()
            .filter_map(|card| card.max_power_watts)
            .sum::<f64>();
        let total_vram_used = self
            .monitor_cards
            .values()
            .filter_map(|card| card.vram_used_mb)
            .sum::<f64>();
        let total_vram_total = self
            .monitor_cards
            .values()
            .filter_map(|card| card.vram_total_mb)
            .sum::<f64>();
        let total_vram_percent = if total_vram_total > 0.0 {
            Some((total_vram_used / total_vram_total) * 100.0)
        } else {
            None
        };

        Some(format!(
            "avg gfx {:.0}% mem {:.0}% hot {:.0}C pwr {:.0}/{:.0}W vr {:.1}%",
            avg_gfx.unwrap_or(0.0),
            avg_mem.unwrap_or(0.0),
            avg_hot.unwrap_or(0.0),
            total_power,
            total_max_power,
            total_vram_percent.unwrap_or(0.0)
        ))
    }

    fn per_gpu_lines(&self) -> Vec<String> {
        self.all_gpu_ids()
            .into_iter()
            .map(|gpu| {
                let card = self.monitor_cards.get(&gpu);
                let gfx_target = self
                    .static_cards
                    .get(&gpu)
                    .and_then(|item| item.gfx_target.as_deref())
                    .unwrap_or("");
                format!(
                    "g{gpu:02} {:>4} {:>5} {:>4} {:>5}{}{}",
                    format_optional_percent(card.and_then(|item| item.gfx_percent)),
                    format_optional_watts(card.and_then(|item| item.power_watts)),
                    format_optional_celsius(card.and_then(|item| item.hotspot_celsius)),
                    format_optional_percent(card.and_then(|item| item.vram_percent)),
                    if gfx_target.is_empty() { "" } else { " " },
                    gfx_target,
                )
                .trim_end()
                .to_owned()
            })
            .collect()
    }

    fn all_gpu_ids(&self) -> Vec<u32> {
        let mut ids = self.static_cards.keys().copied().collect::<Vec<_>>();
        for gpu in self.monitor_cards.keys().copied() {
            if !ids.contains(&gpu) {
                ids.push(gpu);
            }
        }
        ids.sort_unstable();
        ids
    }
}

impl App {
    fn new(paths: AppPaths, initial_provider: Option<String>) -> Self {
        let config = RocmCliConfig::load(&paths).unwrap_or_default();
        let provider = initial_provider.unwrap_or_else(|| "local".to_owned());
        let mut app = Self {
            paths,
            config,
            provider,
            gpu_telemetry: GpuTelemetry::default(),
            transcript: Vec::new(),
            transcript_scroll: 0,
            input: String::new(),
            history: Vec::new(),
            history_index: None,
            status: "Ready.".to_owned(),
            should_quit: false,
        };
        app.push_block(
            "Welcome",
            &format!(
                "ROCm AI Command Center CLI\nprovider: {}\n\nThe status pane actively polls amd-smi for live GPU telemetry when available. Most inspection and setup commands execute inline. Foreground serving and uninstall apply still redirect to the CLI.",
                app.provider
            ),
        );
        app.push_block("Help", &tui_help_text());
        app.transcript_scroll = 0;
        app.status = "Ready. Enter a command, /help, or a request like `serve Qwen3.5 with vllm`."
            .to_owned();
        app
    }

    fn refresh_config(&mut self) {
        self.config = RocmCliConfig::load(&self.paths).unwrap_or_default();
    }

    fn refresh_sidebar_state(&mut self) {
        self.refresh_config();
        self.gpu_telemetry.maybe_refresh();
    }

    fn sidebar_text(&self) -> String {
        let mut output = render_sidebar_text(&self.paths, &self.config, &self.provider);
        let gpu_text = self.gpu_telemetry.sidebar_text();
        if !gpu_text.is_empty() {
            output.push_str("\n\n");
            output.push_str(&gpu_text);
        }
        output
    }

    fn paste_text(&mut self, text: &str) {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        for ch in normalized.chars() {
            if ch == '\n' {
                self.submit();
            } else {
                self.input.push(ch);
                self.history_index = None;
            }
        }
    }

    fn push_user_input(&mut self, input: &str) {
        self.transcript.push(format!("> {input}"));
        self.transcript.push(String::new());
        self.trim_transcript();
        self.scroll_to_bottom();
    }

    fn push_block(&mut self, title: &str, body: &str) {
        self.transcript.push(format!("[{title}]"));
        self.transcript
            .extend(body.lines().map(std::string::ToString::to_string));
        self.transcript.push(String::new());
        self.trim_transcript();
        self.scroll_to_bottom();
    }

    fn trim_transcript(&mut self) {
        const MAX_LINES: usize = 800;
        if self.transcript.len() > MAX_LINES {
            let drain = self.transcript.len() - MAX_LINES;
            self.transcript.drain(0..drain);
            self.transcript_scroll = self.transcript_scroll.saturating_sub(drain as u16);
        }
    }

    fn transcript_text(&self) -> String {
        if self.transcript.is_empty() {
            "No activity yet.".to_owned()
        } else {
            self.transcript.join("\n")
        }
    }

    fn max_scroll(&self) -> u16 {
        self.transcript
            .len()
            .saturating_sub(1)
            .min(u16::MAX as usize) as u16
    }

    fn scroll_to_bottom(&mut self) {
        self.transcript_scroll = self.max_scroll();
    }

    fn scroll_up(&mut self, lines: u16) {
        self.transcript_scroll = self.transcript_scroll.saturating_sub(lines);
    }

    fn scroll_down(&mut self, lines: u16) {
        self.transcript_scroll = self
            .transcript_scroll
            .saturating_add(lines)
            .min(self.max_scroll());
    }

    fn previous_history(&mut self) {
        if self.history.is_empty() {
            self.status = "History is empty.".to_owned();
            return;
        }
        let next_index = match self.history_index {
            Some(index) if index > 0 => index - 1,
            Some(index) => index,
            None => self.history.len() - 1,
        };
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.status = "History recall.".to_owned();
    }

    fn next_history(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 >= self.history.len() {
            self.history_index = None;
            self.input.clear();
            self.status = "History recall cleared.".to_owned();
            return;
        }
        let next_index = index + 1;
        self.history_index = Some(next_index);
        self.input = self.history[next_index].clone();
        self.status = "History recall.".to_owned();
    }

    fn submit(&mut self) {
        let input = self.input.trim().to_owned();
        self.input.clear();
        self.history_index = None;
        if input.is_empty() {
            return;
        }

        if self.history.last() != Some(&input) {
            self.history.push(input.clone());
        }
        self.push_user_input(&input);

        if let Some(command) = input.strip_prefix('/') {
            if !self.handle_command(command.trim()) {
                self.push_block(
                    "Error",
                    &format!(
                        "Unknown slash command: /{}\n\nUse /help to inspect the available commands.",
                        command.trim()
                    ),
                );
                self.status = "Unknown slash command.".to_owned();
            }
            return;
        }

        if self.handle_command(&input) {
            return;
        }

        self.refresh_config();
        let plan = render_freeform_plan(&input, &self.paths, &self.config);
        self.push_block("Plan", &plan);
        self.status =
            "Plan rendered. Inline execution covers inspection plus selected setup commands; serve and uninstall apply still require the CLI."
                .to_owned();
    }

    fn handle_command(&mut self, command: &str) -> bool {
        let mut parts = command.split_whitespace();
        let head = parts.next().unwrap_or_default();
        let args = parts.collect::<Vec<_>>();

        match head {
            "" => true,
            "help" => {
                self.push_block("Help", &tui_help_text());
                self.status = "Help updated.".to_owned();
                true
            }
            "doctor" => {
                match render_doctor_text() {
                    Ok(text) => {
                        self.push_block("Doctor", &text);
                        self.status = "Doctor summary refreshed.".to_owned();
                    }
                    Err(error) => {
                        self.push_block("Error", &error.to_string());
                        self.status = "Doctor collection failed.".to_owned();
                    }
                }
                true
            }
            "engines" => {
                if args.is_empty() || args == ["list"] {
                    self.push_block("Engines", &render_engine_inventory_text());
                    self.status = "Engine inventory refreshed.".to_owned();
                } else {
                    self.status = "Running engine command...".to_owned();
                    let mut cli_args = vec!["engines".to_owned()];
                    cli_args.extend(args.iter().map(|value| (*value).to_owned()));
                    self.run_cli_command("Engines", &cli_args);
                }
                true
            }
            "config" => {
                if args.is_empty() || args == ["show"] {
                    self.refresh_config();
                    self.push_block("Config", &render_config_text(&self.paths, &self.config));
                    self.status = "Config loaded.".to_owned();
                } else {
                    self.status = "Running config command...".to_owned();
                    let mut cli_args = vec!["config".to_owned()];
                    cli_args.extend(args.iter().map(|value| (*value).to_owned()));
                    self.run_cli_command("Config", &cli_args);
                    self.refresh_config();
                }
                true
            }
            "automations" => {
                if args.is_empty() || args == ["list"] {
                    self.refresh_config();
                    match render_automations_text(&self.paths, &self.config) {
                        Ok(text) => {
                            self.push_block("Automations", &text);
                            self.status = "Automation watcher state loaded.".to_owned();
                        }
                        Err(error) => {
                            self.push_block("Error", &error.to_string());
                            self.status = "Automation watcher load failed.".to_owned();
                        }
                    }
                } else {
                    self.status = "Running automations command...".to_owned();
                    let mut cli_args = vec!["automations".to_owned()];
                    cli_args.extend(args.iter().map(|value| (*value).to_owned()));
                    self.run_cli_command("Automations", &cli_args);
                    self.refresh_config();
                }
                true
            }
            "install" => {
                self.status = "Running install command...".to_owned();
                let mut cli_args = vec!["install".to_owned()];
                cli_args.extend(args.iter().map(|value| (*value).to_owned()));
                self.run_cli_command("Install", &cli_args);
                true
            }
            "services" => {
                match render_services_text(&self.paths) {
                    Ok(text) => {
                        self.push_block("Services", &text);
                        self.status = "Managed service state loaded.".to_owned();
                    }
                    Err(error) => {
                        self.push_block("Error", &error.to_string());
                        self.status = "Service inspection failed.".to_owned();
                    }
                }
                true
            }
            "logs" => {
                self.push_block("Logs", &render_logs_text(&self.paths));
                self.status = "Log directories refreshed.".to_owned();
                true
            }
            "gpu" => {
                if args.first() == Some(&"refresh") {
                    self.gpu_telemetry.force_refresh();
                    self.status = "GPU telemetry refreshed.".to_owned();
                } else {
                    self.gpu_telemetry.maybe_refresh();
                    self.status = "GPU telemetry snapshot loaded.".to_owned();
                }
                self.push_block("GPU", &self.gpu_telemetry.detail_text());
                true
            }
            "update" => {
                match render_update_text(&self.paths) {
                    Ok(text) => {
                        self.push_block("Update", &text);
                        self.status = "Update state refreshed.".to_owned();
                    }
                    Err(error) => {
                        self.push_block("Error", &error.to_string());
                        self.status = "Update inspection failed.".to_owned();
                    }
                }
                true
            }
            "daemon" => {
                self.refresh_config();
                self.push_block("Daemon", &render_daemon_text(&self.paths, &self.config));
                self.status = "Daemon policy shown.".to_owned();
                true
            }
            "chat" => {
                self.push_block("Chat", &render_chat_text(&self.provider));
                self.status = "Chat provider shown.".to_owned();
                true
            }
            "provider" => {
                match args.first().copied() {
                    None => {
                        self.push_block(
                            "Provider",
                            &format!("current provider: {}", self.provider),
                        );
                        self.status = "Provider shown.".to_owned();
                    }
                    Some("local" | "anthropic" | "openai") => {
                        self.provider = args[0].to_owned();
                        self.push_block("Provider", &format!("provider set to {}", self.provider));
                        self.status = "Provider updated for this TUI session.".to_owned();
                    }
                    Some(other) => {
                        self.push_block(
                            "Error",
                            &format!(
                                "Unsupported provider: {other}\n\nExpected one of: local, anthropic, openai."
                            ),
                        );
                        self.status = "Provider update failed.".to_owned();
                    }
                }
                true
            }
            "uninstall" => {
                if args
                    .iter()
                    .any(|value| *value == "--yes" || *value == "apply")
                {
                    self.push_block(
                        "Plan",
                        "The TUI only renders uninstall dry-runs right now.\n\nUse `rocm uninstall --yes` from the CLI to apply removals.",
                    );
                    self.status = "Uninstall apply redirected to CLI.".to_owned();
                } else {
                    match render_uninstall_dry_run(&self.paths) {
                        Ok(text) => {
                            self.push_block("Uninstall", &text);
                            self.status = "Rendered uninstall dry-run.".to_owned();
                        }
                        Err(error) => {
                            self.push_block("Error", &error.to_string());
                            self.status = "Uninstall planning failed.".to_owned();
                        }
                    }
                }
                true
            }
            "serve" => {
                self.push_block(
                    "Plan",
                    "Serving is not executed inline in the TUI yet because foreground servers would take over the session.\n\nUse `rocm serve ...` from another shell, or use `--managed` once you want rocmd to supervise it.",
                );
                self.status = "Serve redirected to CLI.".to_owned();
                true
            }
            "clear" => {
                self.transcript.clear();
                self.transcript_scroll = 0;
                self.status = "Transcript cleared.".to_owned();
                true
            }
            "quit" | "exit" => {
                self.should_quit = true;
                true
            }
            _ => false,
        }
    }

    fn run_cli_command(&mut self, title: &str, args: &[String]) {
        match run_cli_command(args) {
            Ok(output) => {
                self.push_block(title, &output.rendered);
                self.status = if output.ok {
                    format!("{title} command completed.")
                } else {
                    format!("{title} command failed.")
                };
            }
            Err(error) => {
                self.push_block("Error", &error.to_string());
                self.status = format!("{title} command failed to launch.");
            }
        }
    }
}

fn run_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    while !app.should_quit {
        app.refresh_sidebar_state();
        terminal
            .draw(|frame| draw(frame, app))
            .context("failed to render rocm TUI")?;

        if !event::poll(Duration::from_millis(250)).context("failed to poll terminal events")? {
            continue;
        }

        match event::read().context("failed to read terminal event")? {
            Event::Key(key) => handle_key(app, key),
            Event::Paste(text) => app.paste_text(&text),
            _ => {}
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) {
    if key.kind == KeyEventKind::Release {
        return;
    }

    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
            app.should_quit = true;
        }
        (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
            app.input.clear();
            app.history_index = None;
            app.status = "Input cleared.".to_owned();
        }
        (KeyModifiers::CONTROL, KeyCode::Char('j' | 'm'))
        | (_, KeyCode::Char('\n' | '\r'))
        | (_, KeyCode::Enter) => app.submit(),
        (_, KeyCode::Backspace) => {
            app.input.pop();
            app.history_index = None;
        }
        (_, KeyCode::Esc) => {
            if app.input.is_empty() {
                app.status = "Input already empty. Use /quit or Ctrl-C to exit.".to_owned();
            } else {
                app.input.clear();
                app.history_index = None;
                app.status = "Input cleared.".to_owned();
            }
        }
        (_, KeyCode::Up) => app.previous_history(),
        (_, KeyCode::Down) => app.next_history(),
        (_, KeyCode::PageUp) => {
            app.scroll_up(10);
            app.status = "Transcript scrolled up.".to_owned();
        }
        (_, KeyCode::PageDown) => {
            app.scroll_down(10);
            app.status = "Transcript scrolled down.".to_owned();
        }
        (_, KeyCode::Home) => {
            app.transcript_scroll = 0;
            app.status = "Transcript moved to top.".to_owned();
        }
        (_, KeyCode::End) => {
            app.scroll_to_bottom();
            app.status = "Transcript moved to bottom.".to_owned();
        }
        (modifiers, KeyCode::Char(ch))
            if !modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) =>
        {
            app.input.push(ch);
            app.history_index = None;
        }
        _ => {}
    }
}

fn draw(frame: &mut Frame<'_>, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(60), Constraint::Length(42)])
        .split(layout[0]);

    let transcript = Paragraph::new(app.transcript_text())
        .block(Block::default().borders(Borders::ALL).title("Transcript"))
        .wrap(Wrap { trim: false })
        .scroll((app.transcript_scroll, 0));
    frame.render_widget(transcript, body[0]);

    let sidebar = Paragraph::new(app.sidebar_text())
        .block(Block::default().borders(Borders::ALL).title("Status"))
        .wrap(Wrap { trim: false });
    frame.render_widget(sidebar, body[1]);

    let input = Paragraph::new(app.input.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Prompt")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(input, layout[1]);

    let footer = Paragraph::new(format!(
        "{} | Enter submit | Up/Down history | PgUp/PgDn scroll | /help | Ctrl-C quit",
        app.status
    ))
    .style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    );
    frame.render_widget(footer, layout[2]);

    let cursor_x = layout[1]
        .x
        .saturating_add(1)
        .saturating_add(u16::try_from(app.input.chars().count()).unwrap_or(u16::MAX))
        .min(layout[1].right().saturating_sub(2));
    let cursor_y = layout[1].y.saturating_add(1);
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn run_cli_command(args: &[String]) -> Result<CommandOutput> {
    let current_exe =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    let output = ProcessCommand::new(&current_exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| {
            format!(
                "failed to execute `{}`",
                format_command_for_display(
                    current_exe.as_os_str().to_string_lossy().as_ref(),
                    args
                )
            )
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let mut rendered = String::new();
    rendered.push_str("command: ");
    rendered.push_str(&format_command_for_display("rocm", args));
    rendered.push('\n');
    rendered.push_str("status: ");
    rendered.push_str(if output.status.success() {
        "ok"
    } else {
        "failed"
    });
    rendered.push('\n');

    if !stdout.is_empty() {
        rendered.push('\n');
        rendered.push_str(stdout.as_str());
        if !stdout.ends_with('\n') {
            rendered.push('\n');
        }
    }
    if !stderr.is_empty() {
        rendered.push('\n');
        rendered.push_str("stderr:\n");
        rendered.push_str(stderr.as_str());
        if !stderr.ends_with('\n') {
            rendered.push('\n');
        }
    }

    Ok(CommandOutput {
        ok: output.status.success(),
        rendered: rendered.trim_end().to_owned(),
    })
}

fn format_command_for_display(binary: &str, args: &[String]) -> String {
    let mut rendered = String::from(binary);
    for arg in args {
        rendered.push(' ');
        if arg.contains(' ') {
            rendered.push('"');
            rendered.push_str(arg);
            rendered.push('"');
        } else {
            rendered.push_str(arg);
        }
    }
    rendered
}

fn load_gpu_static_cards() -> Result<BTreeMap<u32, GpuStaticCard>> {
    let output = run_amd_smi_json(&["static", "-a", "-g", "all", "--json"])?;
    load_gpu_static_cards_from_json(&output)
}

fn load_gpu_static_cards_from_json(output: &str) -> Result<BTreeMap<u32, GpuStaticCard>> {
    let payload = serde_json::from_str::<AmdSmiStaticResponse>(output)
        .context("failed to parse amd-smi static JSON")?;
    let mut cards = BTreeMap::new();
    for entry in payload.gpu_data {
        cards.insert(
            entry.gpu,
            GpuStaticCard {
                market_name: entry
                    .asic
                    .as_ref()
                    .and_then(|asic| asic.market_name.clone()),
                gfx_target: entry
                    .asic
                    .as_ref()
                    .and_then(|asic| asic.target_graphics_version.clone()),
                compute_units: entry.asic.as_ref().and_then(|asic| asic.num_compute_units),
            },
        );
    }
    Ok(cards)
}

fn load_gpu_monitor_cards() -> Result<BTreeMap<u32, GpuMonitorCard>> {
    let output = run_amd_smi_json(&[
        "monitor", "-p", "-t", "-u", "-m", "-v", "-g", "all", "--json",
    ])?;
    load_gpu_monitor_cards_from_json(&output)
}

fn load_gpu_monitor_cards_from_json(output: &str) -> Result<BTreeMap<u32, GpuMonitorCard>> {
    let payload = serde_json::from_str::<Vec<AmdSmiMonitorEntry>>(output)
        .context("failed to parse amd-smi monitor JSON")?;
    let mut cards = BTreeMap::new();
    for entry in payload {
        cards.insert(
            entry.gpu,
            GpuMonitorCard {
                power_watts: entry
                    .power_usage
                    .as_ref()
                    .and_then(AmdSmiMetricValue::as_f64),
                max_power_watts: entry.max_power.as_ref().and_then(AmdSmiMetricValue::as_f64),
                hotspot_celsius: entry
                    .hotspot_temperature
                    .as_ref()
                    .and_then(AmdSmiMetricValue::as_f64),
                memory_celsius: entry
                    .memory_temperature
                    .as_ref()
                    .and_then(AmdSmiMetricValue::as_f64),
                gfx_mhz: entry.gfx_clk.as_ref().and_then(AmdSmiMetricValue::as_f64),
                gfx_percent: entry.gfx.as_ref().and_then(AmdSmiMetricValue::as_f64),
                mem_percent: entry.mem.as_ref().and_then(AmdSmiMetricValue::as_f64),
                mem_mhz: entry.mem_clock.as_ref().and_then(AmdSmiMetricValue::as_f64),
                vram_used_mb: entry.vram_used.as_ref().and_then(AmdSmiMetricValue::as_f64),
                vram_total_mb: entry
                    .vram_total
                    .as_ref()
                    .and_then(AmdSmiMetricValue::as_f64),
                vram_percent: entry
                    .vram_percent
                    .as_ref()
                    .and_then(AmdSmiMetricValue::as_f64),
            },
        );
    }
    Ok(cards)
}

fn run_amd_smi_json(args: &[&str]) -> Result<String> {
    let output = ProcessCommand::new("amd-smi")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to launch amd-smi")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("exit status {}", output.status)
        };
        anyhow::bail!("amd-smi {} failed: {}", args.join(" "), detail);
    }
    String::from_utf8(output.stdout).context("amd-smi output was not valid utf-8")
}

fn average<I>(values: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn format_elapsed(duration: Duration) -> String {
    if duration.as_secs() >= 60 {
        format!(
            "{}m {}s ago",
            duration.as_secs() / 60,
            duration.as_secs() % 60
        )
    } else if duration.as_secs() > 0 {
        format!("{}s ago", duration.as_secs())
    } else {
        format!("{}ms ago", duration.as_millis())
    }
}

fn format_optional_percent(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.0}%"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_optional_watts(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.0}W"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_optional_celsius(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.0}C"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_optional_mhz(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.0}MHz"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_vram_line(card: Option<&GpuMonitorCard>) -> String {
    let Some(card) = card else {
        return "<unknown>".to_owned();
    };
    match (card.vram_used_mb, card.vram_total_mb, card.vram_percent) {
        (Some(used), Some(total), Some(percent)) => {
            format!("{used:.0}/{total:.0}MB ({percent:.1}%)")
        }
        _ => "<unknown>".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        App, handle_key, load_gpu_monitor_cards_from_json, load_gpu_static_cards_from_json,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
    use rocm_core::{AppPaths, RocmCliConfig};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_amd_smi_static_json() {
        let payload = r#"{
          "gpu_data": [
            {
              "gpu": 0,
              "asic": {
                "market_name": "Radeon RX",
                "num_compute_units": 40,
                "target_graphics_version": "gfx1103"
              }
            }
          ]
        }"#;
        let cards = load_gpu_static_cards_from_json(payload).expect("static json should parse");
        let card = cards.get(&0).expect("gpu 0 should exist");
        assert_eq!(card.market_name.as_deref(), Some("Radeon RX"));
        assert_eq!(card.gfx_target.as_deref(), Some("gfx1103"));
        assert_eq!(card.compute_units, Some(40));
    }

    #[test]
    fn parses_amd_smi_monitor_json() {
        let payload = r#"[
          {
            "gpu": 0,
            "power_usage": { "value": 245, "unit": "W" },
            "hotspot_temperature": { "value": 63, "unit": "C" },
            "gfx": { "value": 98, "unit": "%" },
            "vram_total": { "value": 16384, "unit": "MB" },
            "vram_used": { "value": 8192, "unit": "MB" },
            "vram_percent": { "value": 50.0, "unit": "%" }
          }
        ]"#;
        let cards = load_gpu_monitor_cards_from_json(payload).expect("monitor json should parse");
        let card = cards.get(&0).expect("gpu 0 should exist");
        assert_eq!(card.power_watts, Some(245.0));
        assert_eq!(card.hotspot_celsius, Some(63.0));
        assert_eq!(card.gfx_percent, Some(98.0));
        assert_eq!(card.vram_percent, Some(50.0));
    }

    #[test]
    fn ctrl_j_submits_prompt() {
        let mut app = test_app();
        app.input = "/quit".to_owned();
        handle_key(
            &mut app,
            key_event(KeyCode::Char('j'), KeyModifiers::CONTROL),
        );
        assert!(app.should_quit);
    }

    #[test]
    fn enter_submits_prompt() {
        let mut app = test_app();
        app.input = "/quit".to_owned();
        handle_key(&mut app, key_event(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.should_quit);
    }

    #[test]
    fn paste_with_newline_submits_prompt() {
        let mut app = test_app();
        app.paste_text("/quit\n");
        assert!(app.should_quit);
    }

    fn test_app() -> App {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rocm-cli-tui-test-{unique}"));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        App {
            paths,
            config: RocmCliConfig::default(),
            provider: "local".to_owned(),
            gpu_telemetry: Default::default(),
            transcript: Vec::new(),
            transcript_scroll: 0,
            input: String::new(),
            history: Vec::new(),
            history_index: None,
            status: String::new(),
            should_quit: false,
        }
    }

    fn key_event(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }
    }
}
