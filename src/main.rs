use std::env;
use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{Validator, ValidationContext, ValidationResult};
use rustyline::{CompletionType, Config as RLConfig, Context, Editor, Helper};
use rustyline::completion::Completer;
use std::borrow::Cow;

const DEFAULT_MODEL: &str = "gemma-4-26b-a4b-it";
const DEFAULT_BRIDGE: &str = "scripts/gemma_stream.py";

enum StreamKind {
    Answer,
    Thought,
}

struct StreamRenderer<'a> {
    config: &'a Config,
    kind: StreamKind,
    has_prefix: bool,
    in_bold: bool,
    pending_stars: usize,
    leading_spaces: String,
    line_token: String,
}

#[derive(Clone)]
struct Message {
    role: String,
    content: String,
}

struct Config {
    model: String,
    provider: String,
    ollama_url: String,
    python: String,
    bridge: PathBuf,
    temperature: f32,
    thinking: bool,
    plain: bool,
    demo: bool,
    prompt: Option<String>,
}

#[derive(Helper)]
struct RLHelper {
    completer: RLCompleter,
    hinter: HistoryHinter,
}

impl Completer for RLHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        self.completer.complete(line, pos, ctx)
    }
}

impl Hinter for RLHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for RLHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(prompt.to_string())
    }

    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        Cow::Borrowed(line)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        false
    }
}

impl Validator for RLHelper {
    fn validate(&self, _ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(ValidationResult::Valid(None))
    }
}

struct RLCompleter {
    commands: Vec<String>,
}

impl Completer for RLCompleter {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        _pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        if line.starts_with('/') {
            let matches: Vec<String> = self
                .commands
                .iter()
                .filter(|c| c.starts_with(line))
                .cloned()
                .collect();
            return Ok((0, matches));
        }
        Ok((0, vec![]))
    }
}

fn main() {
    let mut config = match parse_args() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            print_help();
            std::process::exit(2);
        }
    };

    if !config.plain {
        print_banner(&config);
    }

    if config.demo {
        run_demo(&config);
        return;
    }

    if let Some(prompt) = config.prompt.clone() {
        let mut history = vec![Message {
            role: "user".to_string(),
            content: prompt,
        }];
        match ask_model(&config, &history) {
            Ok(answer) => {
                if !answer.trim().is_empty() {
                    history.push(Message {
                        role: "model".to_string(),
                        content: answer,
                    });
                }
            }
            Err(err) => exit_with_error(&err),
        }
        return;
    }

    chat_loop(&mut config);
}

fn chat_loop(config: &mut Config) {
    let mut history_msgs: Vec<Message> = Vec::new();

    let rl_config = RLConfig::builder()
        .completion_type(CompletionType::List)
        .build();
    
    let commands = vec![
        "/exit".into(),
        "/quit".into(),
        "/clear".into(),
        "/help".into(),
        "/model".into(),
        "/provider".into(),
    ];

    let helper = RLHelper {
        completer: RLCompleter {
            commands,
        },
        hinter: HistoryHinter {},
    };
    
    let mut rl = Editor::with_config(rl_config).expect("failed to init rustyline");
    rl.set_helper(Some(helper));

    let history_path = env::temp_dir().join(".edge_glow_history");
    let _ = rl.load_history(&history_path);

    loop {
        let label = "you";
        let color = "\x1b[1m\x1b[38;5;82m";
        let prompt = paint(config, &format!("{label} ❯ "), color);

        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(input);

                if input.starts_with('/') {
                    let parts: Vec<&str> = input.split_whitespace().collect();
                    match parts[0] {
                        "/exit" | "/quit" => break,
                        "/clear" => {
                            history_msgs.clear();
                            print!("\x1b[2J\x1b[H");
                            if !config.plain {
                                print_banner(config);
                            }
                            continue;
                        }
                        "/models" | "/model" => {
                            if parts.len() > 1 {
                                let mut new_model = parts[1..].join(" ");
                                if new_model.starts_with("ollama:") {
                                    config.provider = "ollama".to_string();
                                    new_model = new_model.trim_start_matches("ollama:").to_string();
                                    print_notice(config, "provider switched to: ollama");
                                }
                                config.model = new_model;
                                print_notice(config, &format!("model switched to: {}", config.model));
                            } else {
                                list_models(config);
                            }
                            continue;
                        }
                        "/help" => {
                            print_chat_help(config);
                            continue;
                        }
                        "/providers" | "/provider" => {
                            if parts.len() > 1 {
                                let new_provider = parts[1].to_lowercase();
                                if new_provider == "google" || new_provider == "ollama" {
                                    let old_provider = config.provider.clone();
                                    config.provider = new_provider;
                                    print_notice(config, &format!("provider switched to: {}", config.provider));
                                    
                                    if config.provider != old_provider {
                                        if config.provider == "ollama" {
                                            if let Ok(models) = fetch_models(config) {
                                                let first_installed = models.iter().find(|m| m.ends_with("- installed"));
                                                if let Some(m) = first_installed {
                                                    let id = if let Some(sp) = m.find(' ') { &m[..sp] } else { m };
                                                    config.model = id.to_string();
                                                    print_notice(config, &format!("auto-selected model: {}", config.model));
                                                }
                                            }
                                        } else if config.provider == "google" {
                                            config.model = env::var("GEMMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
                                            print_notice(config, &format!("reset to default google model: {}", config.model));
                                        }
                                    }
                                } else {
                                    print_error(config, "unknown provider. use 'google' or 'ollama'");
                                }
                            } else {
                                list_providers(config);
                            }
                            continue;
                        }
                        _ => {
                            print_error(config, &format!("unknown command: '{}'. type /help for commands", parts[0]));
                            continue;
                        }
                    }
                }

                history_msgs.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                });

                match ask_model(config, &history_msgs) {
                    Ok(answer) => {
                        if !answer.trim().is_empty() {
                            history_msgs.push(Message {
                                role: "model".to_string(),
                                content: answer,
                            });
                        }
                    }
                    Err(err) => {
                        print_error(config, &err);
                        history_msgs.pop();
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    let _ = rl.save_history(&history_path);
}

fn list_models(config: &Config) {
    match fetch_models(config) {
        Ok(models) => {
            if models.is_empty() {
                print_notice(config, "no models found or provider unreachable");
            } else {
                println!("{}", paint(config, "╭─ Available Models ──────────────────────────╮", "\x1b[38;5;141m"));
                for entry in models {
                    let mut display_name = entry.clone();
                    let mut indicator = " ".to_string();

                    if config.provider == "ollama" {
                        let model_id = if let Some(first_space) = entry.find(' ') {
                            &entry[..first_space]
                        } else {
                            &entry
                        };

                        let is_current = model_id == config.model 
                            || format!("{}:latest", model_id) == config.model
                            || config.model.starts_with(model_id);

                        if is_current {
                            indicator = paint(config, "●", "\x1b[38;5;220m");
                            display_name = entry.trim_end_matches("- installed")
                                               .trim_end_matches("- available")
                                               .to_string();
                        } else if entry.ends_with("- installed") {
                            indicator = paint(config, "●", "\x1b[38;5;39m");
                            display_name = entry.trim_end_matches("- installed").to_string();
                        } else if entry.ends_with("- available") {
                            indicator = " ".to_string();
                            display_name = entry.trim_end_matches("- available").to_string();
                        }
                    } else {
                        if entry == config.model {
                            indicator = paint(config, "●", "\x1b[38;5;220m");
                        }
                    }

                    println!("{} {} {}", paint(config, "│", "\x1b[38;5;141m"), indicator, display_name);
                }
                println!("{}", paint(config, "╰────────────────────────────────────────────╯", "\x1b[38;5;141m"));
                print_notice(config, &format!("current model: {} (provider: {})", config.model, config.provider));
                print_notice(config, "use '/model <name>' to switch");
            }
        }
        Err(err) => print_error(config, &err),
    }
}

fn list_providers(config: &Config) {
    println!("{}", paint(config, "╭─ Supported Providers ───────────────────────╮", "\x1b[38;5;141m"));
    let providers = ["google", "ollama"];
    for p in providers {
        let indicator = if p == config.provider {
            paint(config, "●", "\x1b[38;5;82m")
        } else {
            " ".to_string()
        };
        println!("{} {} {}", paint(config, "│", "\x1b[38;5;141m"), indicator, p);
    }
    println!("{}", paint(config, "╰────────────────────────────────────────────╯", "\x1b[38;5;141m"));
    print_notice(config, &format!("current provider: {}", config.provider));
    print_notice(config, "use '/provider <name>' to switch");
}

fn ask_model(config: &Config, history: &[Message]) -> Result<String, String> {
    let payload = build_payload(config, history);
    let mut child = Command::new(&config.python)
        .arg(&config.bridge)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            format!(
                "failed to start Python bridge '{}': {err}",
                config.bridge.display()
            )
        })?;

    let mut child_stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to open Python stdin".to_string())?;
    child_stdin
        .write_all(payload.as_bytes())
        .map_err(|err| format!("failed to send prompt to Python bridge: {err}"))?;
    drop(child_stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to read Python stdout".to_string())?;
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut answer = String::new();
    let mut thought_renderer: Option<StreamRenderer<'_>> = None;
    let mut answer_renderer: Option<StreamRenderer<'_>> = None;
    let mut stream_error: Option<String> = None;

    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|err| format!("failed while reading stream: {err}"))?;
        if bytes == 0 {
            break;
        }

        let event_type = json_field(&line, "type").unwrap_or_default();
        match event_type.as_str() {
            "thought" => {
                let text = json_field(&line, "text").unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                if thought_renderer.is_none() {
                    thought_renderer = Some(StreamRenderer::new(config, StreamKind::Thought));
                }
                if let Some(renderer) = thought_renderer.as_mut() {
                    renderer.feed(&text);
                }
                let _ = io::stdout().flush();
            }
            "answer" => {
                let text = json_field(&line, "text").unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                if answer_renderer.is_none() {
                    if let Some(mut renderer) = thought_renderer.take() {
                        renderer.finish();
                    }
                    answer_renderer = Some(StreamRenderer::new(config, StreamKind::Answer));
                }
                answer.push_str(&text);
                if let Some(renderer) = answer_renderer.as_mut() {
                    renderer.feed(&text);
                }
                let _ = io::stdout().flush();
            }
            "done" => break,
            "error" => {
                let message = json_field(&line, "message")
                    .unwrap_or_else(|| "unknown bridge error".to_string());
                stream_error = Some(message);
                break;
            }
            _ => {}
        }
    }

    let status = child
        .wait()
        .map_err(|err| format!("failed to wait for Python bridge: {err}"))?;

    if let Some(message) = stream_error {
        return Err(message);
    }

    if !status.success() {
        let mut stderr = String::new();
        if let Some(mut child_stderr) = child.stderr.take() {
            let _ = child_stderr.read_to_string(&mut stderr);
        }
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            format!("Python bridge exited with status {status}")
        } else {
            detail.to_string()
        });
    }

    if let Some(renderer) = thought_renderer.as_mut() {
        renderer.finish();
    }
    if let Some(renderer) = answer_renderer.as_mut() {
        renderer.finish();
    }

    Ok(answer)
}

fn fetch_models(config: &Config) -> Result<Vec<String>, String> {
    let mut payload = String::new();
    payload.push('{');
    write!(payload, "\"action\":\"list_models\"").unwrap();
    write!(payload, ",\"provider\":{}", json_string(&config.provider)).unwrap();
    write!(payload, ",\"ollama_url\":{}", json_string(&config.ollama_url)).unwrap();
    payload.push('}');

    let mut child = Command::new(&config.python)
        .arg(&config.bridge)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to start bridge: {err}"))?;

    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(payload.as_bytes()).unwrap();
    drop(stdin);

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut models = Vec::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|err| format!("read error: {err}"))?;
        if bytes == 0 { break; }

        let event_type = json_field(&line, "type").unwrap_or_default();
        if event_type == "error" {
            if let Some(msg) = json_field(&line, "message") {
                return Err(msg);
            }
        } else if event_type == "model_item" {
            if let Some(text) = json_field(&line, "text") {
                models.push(text);
            }
        } else if event_type == "done" {
            break;
        }
    }

    if models.is_empty() {
        return Err("bridge returned no models".to_string());
    }

    Ok(models)
}

fn build_payload(config: &Config, history: &[Message]) -> String {
    let mut out = String::new();
    out.push('{');
    write!(out, "\"model\":{}", json_string(&config.model)).unwrap();
    write!(out, ",\"provider\":{}", json_string(&config.provider)).unwrap();
    write!(out, ",\"ollama_url\":{}", json_string(&config.ollama_url)).unwrap();
    write!(out, ",\"temperature\":{}", config.temperature).unwrap();
    write!(out, ",\"thinking\":{}", config.thinking).unwrap();
    out.push_str(",\"messages\":[");
    for (index, message) in history.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('{');
        write!(out, "\"role\":{}", json_string(&message.role)).unwrap();
        write!(out, ",\"content\":{}", json_string(&message.content)).unwrap();
        out.push('}');
    }
    out.push_str("]}");
    out
}

fn parse_args() -> Result<Config, String> {
    let mut model = env::var("GEMMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let mut provider = "google".to_string();
    let mut ollama_url = env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let mut python = env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut bridge = PathBuf::from(DEFAULT_BRIDGE);
    let mut temperature = 0.7_f32;
    let mut thinking = true;
    let mut plain = false;
    let mut demo = false;
    let mut prompt = None;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-m" | "--model" => {
                index += 1;
                model = args
                    .get(index)
                    .ok_or_else(|| "--model needs a value".to_string())?
                    .to_string();
                if model.starts_with("ollama:") {
                    provider = "ollama".to_string();
                    model = model.trim_start_matches("ollama:").to_string();
                }
            }
            "--provider" => {
                index += 1;
                provider = args
                    .get(index)
                    .ok_or_else(|| "--provider needs a value (google/ollama)".to_string())?
                    .to_lowercase();
            }
            "--ollama" => {
                provider = "ollama".to_string();
            }
            "--ollama-url" => {
                index += 1;
                ollama_url = args
                    .get(index)
                    .ok_or_else(|| "--ollama-url needs a value".to_string())?
                    .to_string();
            }
            "--python" => {
                index += 1;
                python = args
                    .get(index)
                    .ok_or_else(|| "--python needs a value".to_string())?
                    .to_string();
            }
            "--bridge" => {
                index += 1;
                bridge = PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--bridge needs a value".to_string())?,
                );
            }
            "-t" | "--temperature" => {
                index += 1;
                temperature = args
                    .get(index)
                    .ok_or_else(|| "--temperature needs a value".to_string())?
                    .parse::<f32>()
                    .map_err(|_| "--temperature must be a number".to_string())?;
            }
            "-p" | "--prompt" => {
                index += 1;
                prompt = Some(
                    args.get(index)
                        .ok_or_else(|| "--prompt needs a value".to_string())?
                        .to_string(),
                );
            }
            "--no-thinking" => thinking = false,
            "--plain" => plain = true,
            "--demo" => demo = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
        index += 1;
    }

    Ok(Config {
        model,
        provider,
        ollama_url,
        python,
        bridge,
        temperature,
        thinking,
        plain,
        demo,
        prompt,
    })
}

fn print_banner(config: &Config) {
    println!(
        "{}",
        paint(
            config,
            "╭─ Edge Glow ───────────────────────────────╮",
            "\x1b[38;5;141m"
        )
    );
    println!(
        "{} {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, "Google AI Studio / Ollama local chat", "\x1b[1m\x1b[96m")
    );
    println!(
        "{} provider    {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, &config.provider, "\x1b[38;5;80m")
    );
    println!(
        "{} model       {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, &config.model, "\x1b[38;5;220m")
    );
    println!(
        "{} thinking    {}",
        paint(config, "│", "\x1b[38;5;141m"),
        if config.provider == "ollama" {
            paint(config, "off (not supported by Ollama)", "\x1b[38;5;245m")
        } else if config.thinking {
            paint(config, "summaries on", "\x1b[38;5;80m")
        } else {
            paint(config, "off", "\x1b[38;5;245m")
        }
    );
    println!(
        "{} commands    {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, "/help  /model  /provider  /exit", "\x1b[38;5;250m")
    );
    println!(
        "{}",
        paint(
            config,
            "╰────────────────────────────────────────────╯",
            "\x1b[38;5;141m"
        )
    );
}

fn print_help() {
    println!("Edge Glow - streamed Google AI Studio / Ollama chat");
    println!();
    println!("Usage:");
    println!("  cargo run -- [options]");
    println!("  cargo run -- --prompt \"hello\"");
    println!();
    println!("Options:");
    println!("  -m, --model <name>          Model name (prefix with 'ollama:' for local)");
    println!("      --provider <name>      Backend provider (google or ollama)");
    println!("      --ollama               Force Ollama provider");
    println!("      --ollama-url <url>     Ollama API URL (default: http://localhost:11434)");
    println!("  -p, --prompt <text>         Run a single prompt and exit");
    println!("  -t, --temperature <number>  Sampling temperature (default: 0.7)");
    println!("      --no-thinking          Do not request thought summaries");
    println!("      --python <binary>      Python binary (default: python3 or PYTHON)");
    println!("      --bridge <path>        Python bridge path");
    println!("      --plain                Disable colors");
    println!("      --demo                 Preview the polished renderer");
    println!("  -h, --help                 Show help");
}

fn print_chat_help(config: &Config) {
    println!(
        "{}",
        paint(config, "╭─ Commands ─────────────────────────────────╮", "\x1b[38;5;141m")
    );
    println!("{} /help              show commands", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /model [name]      list or switch model", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /provider [name]   list or switch provider", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /clear             clear chat memory", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /exit              quit", paint(config, "│", "\x1b[38;5;141m"));
    println!(
        "{}",
        paint(config, "╰────────────────────────────────────────────╯", "\x1b[38;5;141m")
    );
}

fn print_notice(config: &Config, message: &str) {
    println!(
        "{} {}",
        paint(config, "note", "\x1b[1m\x1b[38;5;80m"),
        message
    );
}

fn print_error(config: &Config, message: &str) {
    eprintln!(
        "\n{} {}",
        paint(config, "error", "\x1b[1m\x1b[38;5;196m"),
        message
    );
}

fn run_demo(config: &Config) {
    let mut thought = StreamRenderer::new(config, StreamKind::Thought);
    thought.feed("I will keep this concise, structured, and easy to scan.");
    thought.finish();

    let mut answer = StreamRenderer::new(config, StreamKind::Answer);
    answer.feed("## Clean CLI Output\n\n");
    answer.feed("Here is how streamed Markdown now looks inside Edge Glow:\n\n");
    answer.feed("1. **Readable sections:** answers live inside a bordered response block.\n");
    answer.feed("2. **Cleaner Markdown:** bold markers disappear and styling is applied.\n");
    answer.feed("* **Compact bullets:** raw `*` list markers become terminal bullets.\n\n");
    answer.feed("Use `/clear` when you want a fresh chat, and `/exit` when you are done.");
    answer.finish();
}

fn paint(config: &Config, text: &str, color: &str) -> String {
    if config.plain {
        text.to_string()
    } else {
        format!("{color}{text}\x1b[0m")
    }
}

impl<'a> StreamRenderer<'a> {
    fn new(config: &'a Config, kind: StreamKind) -> Self {
        match kind {
            StreamKind::Answer => {
                let label = format!("╭─ Glow({}) ", config.model);
                let mut header = label;
                let total_width = 46;
                let current_width = header.chars().count();
                if current_width < total_width - 1 {
                    for _ in 0..(total_width - 1 - current_width) {
                        header.push('─');
                    }
                }
                header.push('╮');
                println!("{}", paint(config, &header, "\x1b[38;5;213m"));
            }
            StreamKind::Thought => {
                println!(
                    "{}",
                    paint(
                        config,
                        "╭─ Thinking Summary ─────────────────────────╮",
                        "\x1b[2m\x1b[38;5;80m"
                    )
                );
            }
        }

        Self {
            config,
            kind,
            has_prefix: false,
            in_bold: false,
            pending_stars: 0,
            leading_spaces: String::new(),
            line_token: String::new(),
        }
    }

    fn feed(&mut self, text: &str) {
        for ch in text.chars() {
            self.feed_char(ch);
        }
    }

    fn finish(&mut self) {
        self.flush_line_token_as_body();
        self.flush_pending_stars();
        if self.in_bold {
            self.in_bold = false;
            print!("{}", self.base_color());
        }
        if self.has_prefix {
            println!();
        }
        let color = match self.kind {
            StreamKind::Answer => "\x1b[38;5;213m",
            StreamKind::Thought => "\x1b[2m\x1b[38;5;80m",
        };
        println!(
            "{}",
            paint(self.config, "╰────────────────────────────────────────────╯", color)
        );
    }

    fn feed_char(&mut self, ch: char) {
        if ch == '\n' {
            self.flush_line_token_as_body();
            self.flush_pending_stars();
            if self.has_prefix {
                println!();
            }
            self.has_prefix = false;
            self.leading_spaces.clear();
            self.line_token.clear();
            return;
        }

        if !self.has_prefix {
            if self.line_token.is_empty() && (ch == ' ' || ch == '\t') {
                self.leading_spaces.push(ch);
                return;
            }

            if self.try_line_marker(ch) {
                return;
            }

            self.emit_prefix();
            self.emit_leading_spaces();
        }

        self.emit_body_char(ch);
    }

    fn try_line_marker(&mut self, ch: char) -> bool {
        if self.line_token.is_empty() {
            if ch == '*' || ch == '-' || ch == '#' || ch.is_ascii_digit() {
                self.line_token.push(ch);
                return true;
            }
            return false;
        }

        if (self.line_token == "*" || self.line_token == "-") && ch.is_whitespace() {
            self.emit_prefix();
            self.emit_leading_spaces();
            print!("{}", paint(self.config, "• ", "\x1b[38;5;80m"));
            self.line_token.clear();
            return true;
        }

        if self.line_token.chars().all(|token_ch| token_ch == '#') {
            if ch == '#' && self.line_token.len() < 6 {
                self.line_token.push(ch);
                return true;
            }
            if ch.is_whitespace() {
                self.emit_prefix();
                self.emit_leading_spaces();
                print!("{}", self.heading_color());
                self.line_token.clear();
                return true;
            }
        }

        if self.line_token.chars().all(|token_ch| token_ch.is_ascii_digit()) {
            if ch.is_ascii_digit() {
                self.line_token.push(ch);
                return true;
            }
            if ch == '.' {
                self.line_token.push(ch);
                return true;
            }
        }

        if self.line_token.ends_with('.')
            && self
                .line_token
                .trim_end_matches('.')
                .chars()
                .all(|token_ch| token_ch.is_ascii_digit())
            && ch.is_whitespace()
        {
            self.emit_prefix();
            self.emit_leading_spaces();
            print!(
                "{}",
                paint(self.config, &format!("{} ", self.line_token), "\x1b[38;5;80m")
            );
            self.line_token.clear();
            return true;
        }

        self.flush_line_token_as_body();
        false
    }

    fn emit_body_char(&mut self, ch: char) {
        if ch == '*' {
            self.pending_stars += 1;
            if self.pending_stars == 2 {
                self.toggle_bold();
                self.pending_stars = 0;
            }
            return;
        }

        self.flush_pending_stars();
        print!("{ch}");
    }

    fn emit_prefix(&mut self) {
        if self.has_prefix {
            return;
        }
        let color = match self.kind {
            StreamKind::Answer => "\x1b[38;5;213m",
            StreamKind::Thought => "\x1b[2m\x1b[38;5;80m",
        };
        print!("{} ", paint(self.config, "│", color));
        self.has_prefix = true;
        print!("{}", self.base_color());
    }

    fn emit_leading_spaces(&mut self) {
        if !self.leading_spaces.is_empty() {
            print!("{}", self.leading_spaces);
            self.leading_spaces.clear();
        }
    }

    fn flush_line_token_as_body(&mut self) {
        if self.line_token.is_empty() {
            return;
        }
        self.emit_prefix();
        self.emit_leading_spaces();
        let token = std::mem::take(&mut self.line_token);
        for ch in token.chars() {
            self.emit_body_char(ch);
        }
    }

    fn flush_pending_stars(&mut self) {
        for _ in 0..self.pending_stars {
            print!("*");
        }
        self.pending_stars = 0;
    }

    fn toggle_bold(&mut self) {
        self.in_bold = !self.in_bold;
        if self.config.plain {
            return;
        }
        if self.in_bold {
            print!("\x1b[1m");
        } else {
            print!("{}", self.base_color());
        }
    }

    fn base_color(&self) -> &'static str {
        if self.config.plain {
            ""
        } else {
            match self.kind {
                StreamKind::Answer => "\x1b[0m",
                StreamKind::Thought => "\x1b[2m\x1b[38;5;250m",
            }
        }
    }

    fn heading_color(&self) -> &'static str {
        if self.config.plain {
            ""
        } else {
            match self.kind {
                StreamKind::Answer => "\x1b[1m\x1b[38;5;220m",
                StreamKind::Thought => "\x1b[1m\x1b[2m\x1b[38;5;80m",
            }
        }
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            ch if ch.is_control() => write!(out, "\\u{:04x}", ch as u32).unwrap(),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn json_field(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = line.find(&needle)?;
    let after_key = &line[start + needle.len()..];
    let colon = after_key.find(':')?;
    let mut rest = after_key[colon + 1..].trim_start().chars().peekable();
    if rest.next()? != '"' {
        return None;
    }

    let mut out = String::new();
    while let Some(ch) = rest.next() {
        match ch {
            '"' => return Some(out),
            '\\' => match rest.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{08}'),
                'f' => out.push('\u{0C}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let mut code = String::new();
                    for _ in 0..4 {
                        code.push(rest.next()?);
                    }
                    let value = u16::from_str_radix(&code, 16).ok()?;
                    if let Some(decoded) = char::from_u32(value as u32) {
                        out.push(decoded);
                    }
                }
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

fn exit_with_error(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}
