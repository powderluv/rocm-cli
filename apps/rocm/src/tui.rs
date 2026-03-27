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
use std::io::{self, Stdout};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

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

impl App {
    fn new(paths: AppPaths, initial_provider: Option<String>) -> Self {
        let config = RocmCliConfig::load(&paths).unwrap_or_default();
        let provider = initial_provider.unwrap_or_else(|| "local".to_owned());
        let mut app = Self {
            paths,
            config,
            provider,
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
                "ROCm AI Command Center CLI\nprovider: {}\n\nMost inspection and setup commands execute inline. Foreground serving and uninstall apply still redirect to the CLI.",
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
        app.refresh_config();
        terminal
            .draw(|frame| draw(frame, app))
            .context("failed to render rocm TUI")?;

        if !event::poll(Duration::from_millis(250)).context("failed to poll terminal events")? {
            continue;
        }

        let Event::Key(key) = event::read().context("failed to read terminal event")? else {
            continue;
        };
        handle_key(app, key);
    }
    Ok(())
}

fn handle_key(app: &mut App, key: KeyEvent) {
    if key.kind != KeyEventKind::Press {
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
        (_, KeyCode::Enter) => app.submit(),
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
        (_, KeyCode::Char(ch)) => {
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

    let sidebar = Paragraph::new(render_sidebar_text(&app.paths, &app.config, &app.provider))
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
