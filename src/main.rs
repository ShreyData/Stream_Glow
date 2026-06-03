use std::env;
use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

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
    python: String,
    bridge: PathBuf,
    temperature: f32,
    thinking: bool,
    plain: bool,
    demo: bool,
    prompt: Option<String>,
}

fn main() {
    let config = match parse_args() {
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

    chat_loop(&config);
}

fn chat_loop(config: &Config) {
    let mut history: Vec<Message> = Vec::new();
    let stdin = io::stdin();

    loop {
        print_prompt(config, "you");
        let _ = io::stdout().flush();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                eprintln!("input error: {err}");
                break;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                print!("\x1b[2J\x1b[H");
                if !config.plain {
                    print_banner(config);
                }
                continue;
            }
            "/help" => {
                print_chat_help(config);
                continue;
            }
            "/model" => {
                print_notice(config, &format!("model: {}", config.model));
                continue;
            }
            _ => {}
        }

        history.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        match ask_model(config, &history) {
            Ok(answer) => {
                if !answer.trim().is_empty() {
                    history.push(Message {
                        role: "model".to_string(),
                        content: answer,
                    });
                }
            }
            Err(err) => {
                print_error(config, &err);
                history.pop();
            }
        }
    }
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

fn build_payload(config: &Config, history: &[Message]) -> String {
    let mut out = String::new();
    out.push('{');
    write!(out, "\"model\":{}", json_string(&config.model)).unwrap();
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
            "╭─ Gemma Glow ───────────────────────────────╮",
            "\x1b[38;5;141m"
        )
    );
    println!(
        "{} {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, "Google AI Studio streaming chat", "\x1b[1m\x1b[96m")
    );
    println!(
        "{} model      {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, &config.model, "\x1b[38;5;220m")
    );
    println!(
        "{} thinking   {}",
        paint(config, "│", "\x1b[38;5;141m"),
        if config.thinking {
            paint(config, "summaries on", "\x1b[38;5;80m")
        } else {
            paint(config, "off", "\x1b[38;5;245m")
        }
    );
    println!(
        "{} commands   {}",
        paint(config, "│", "\x1b[38;5;141m"),
        paint(config, "/help  /model  /clear  /exit", "\x1b[38;5;250m")
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
    println!("Gemma Glow - streamed Google AI Studio chat");
    println!();
    println!("Usage:");
    println!("  cargo run -- [options]");
    println!("  cargo run -- --prompt \"hello\"");
    println!();
    println!("Options:");
    println!("  -m, --model <name>          Model name (default: gemma-4-26b-a4b-it)");
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
    println!("{} /help   show commands", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /model  show current model", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /clear  clear chat memory", paint(config, "│", "\x1b[38;5;141m"));
    println!("{} /exit   quit", paint(config, "│", "\x1b[38;5;141m"));
    println!(
        "{}",
        paint(config, "╰────────────────────────────────────────────╯", "\x1b[38;5;141m")
    );
}

fn print_prompt(config: &Config, label: &str) {
    let color = if label == "you" {
        "\x1b[1m\x1b[38;5;82m"
    } else {
        "\x1b[1m\x1b[38;5;213m"
    };
    print!("\n{} ", paint(config, &format!("{label} ❯"), color));
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
                println!(
                    "{}",
                    paint(config, "╭─ Gemma ────────────────────────────────────╮", "\x1b[38;5;213m")
                );
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
    answer.feed("Here is how streamed Markdown now looks inside Gemma Glow:\n\n");
    answer.feed("1. **Readable sections:** answers live inside a bordered response block.\n");
    answer.feed("2. **Cleaner Markdown:** bold markers disappear and styling is applied.\n");
    answer.feed("* **Compact bullets:** raw `*` list markers become terminal bullets.\n\n");
    answer.feed("Use `/clear` when you want a fresh chat, and `/exit` when you are done.");
    answer.finish();
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
