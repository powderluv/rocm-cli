use crate::{
    render_config_text, render_doctor_text, render_engine_inventory_text, render_freeform_plan,
    render_services_text, render_sidebar_text, render_uninstall_dry_run, tui_help_text,
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
use std::time::Duration;

pub fn run() -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut app = App::new(paths)?;

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
    transcript: Vec<String>,
    input: String,
    status: String,
    should_quit: bool,
}

impl App {
    fn new(paths: AppPaths) -> Result<Self> {
        let config = RocmCliConfig::load(&paths)?;
        let mut app = Self {
            paths,
            config,
            transcript: Vec::new(),
            input: String::new(),
            status: "Enter a slash command or a natural-language request.".to_owned(),
            should_quit: false,
        };
        app.push_block(
            "Welcome",
            "ROCm AI Command Center CLI\n\nUse /help for commands or type a request like `serve Qwen3.5 with vllm`.",
        );
        app.push_block("Help", &tui_help_text());
        Ok(app)
    }

    fn refresh_config(&mut self) {
        self.config = RocmCliConfig::load(&self.paths).unwrap_or_default();
    }

    fn push_user_input(&mut self, input: &str) {
        self.transcript.push(format!("> {input}"));
        self.transcript.push(String::new());
        self.trim_transcript();
    }

    fn push_block(&mut self, title: &str, body: &str) {
        self.transcript.push(format!("[{title}]"));
        self.transcript
            .extend(body.lines().map(std::string::ToString::to_string));
        self.transcript.push(String::new());
        self.trim_transcript();
    }

    fn trim_transcript(&mut self) {
        const MAX_LINES: usize = 500;
        if self.transcript.len() > MAX_LINES {
            let drain = self.transcript.len() - MAX_LINES;
            self.transcript.drain(0..drain);
        }
    }

    fn transcript_text(&self) -> String {
        if self.transcript.is_empty() {
            "No activity yet.".to_owned()
        } else {
            self.transcript.join("\n")
        }
    }

    fn submit(&mut self) {
        let input = self.input.trim().to_owned();
        self.input.clear();
        if input.is_empty() {
            return;
        }

        self.push_user_input(&input);
        if let Some(command) = input.strip_prefix('/') {
            self.handle_command(command.trim());
        } else {
            self.refresh_config();
            let plan = render_freeform_plan(&input, &self.paths, &self.config);
            self.push_block("Plan", &plan);
            self.status = "Plan rendered. Use the explicit CLI command to execute it.".to_owned();
        }
    }

    fn handle_command(&mut self, command: &str) {
        let mut parts = command.split_whitespace();
        let head = parts.next().unwrap_or_default();
        match head {
            "" => {}
            "help" => {
                self.push_block("Help", &tui_help_text());
                self.status = "Help updated.".to_owned();
            }
            "doctor" => match render_doctor_text() {
                Ok(text) => {
                    self.push_block("Doctor", &text);
                    self.status = "Doctor summary refreshed.".to_owned();
                }
                Err(error) => {
                    self.push_block("Error", &error.to_string());
                    self.status = "Doctor collection failed.".to_owned();
                }
            },
            "engines" => {
                self.push_block("Engines", &render_engine_inventory_text());
                self.status = "Engine inventory refreshed.".to_owned();
            }
            "config" => {
                self.refresh_config();
                self.push_block("Config", &render_config_text(&self.paths, &self.config));
                self.status = "Config loaded.".to_owned();
            }
            "services" => match render_services_text(&self.paths) {
                Ok(text) => {
                    self.push_block("Services", &text);
                    self.status = "Managed service state loaded.".to_owned();
                }
                Err(error) => {
                    self.push_block("Error", &error.to_string());
                    self.status = "Service inspection failed.".to_owned();
                }
            },
            "uninstall" => match render_uninstall_dry_run(&self.paths) {
                Ok(text) => {
                    self.push_block("Uninstall", &text);
                    self.status = "Rendered uninstall dry-run.".to_owned();
                }
                Err(error) => {
                    self.push_block("Error", &error.to_string());
                    self.status = "Uninstall planning failed.".to_owned();
                }
            },
            "clear" => {
                self.transcript.clear();
                self.status = "Transcript cleared.".to_owned();
            }
            "quit" | "exit" => {
                self.should_quit = true;
            }
            other => {
                self.push_block(
                    "Error",
                    &format!("Unknown slash command: /{other}\n\nUse /help to inspect the available commands."),
                );
                self.status = "Unknown slash command.".to_owned();
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
            app.status = "Input cleared.".to_owned();
        }
        (_, KeyCode::Enter) => app.submit(),
        (_, KeyCode::Backspace) => {
            app.input.pop();
        }
        (_, KeyCode::Esc) => {
            if app.input.is_empty() {
                app.status = "Input already empty. Use /quit or Ctrl-C to exit.".to_owned();
            } else {
                app.input.clear();
                app.status = "Input cleared.".to_owned();
            }
        }
        (_, KeyCode::Char(ch)) => {
            app.input.push(ch);
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
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript, body[0]);

    let sidebar = Paragraph::new(render_sidebar_text(&app.paths, &app.config))
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
        "{} | Enter submit | /help commands | Esc clear input | Ctrl-C quit",
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
