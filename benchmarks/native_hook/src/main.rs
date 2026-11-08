//! Experimental command boundary for the existing resident worker protocol.
//! No retry after any send attempt. Python remains the pre-dispatch fallback.

#![forbid(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

mod timing;

const DESCRIPTOR_ENV: &str = "MNEMOS_HOOK_WORKER_DESCRIPTOR";
const MAX_FRAME: usize = 1_048_576;
const MAX_HEADERS: usize = 16_384;
const DEADLINE: Duration = Duration::from_secs(30);

enum Outcome {
    Fallback,
    Handled(Value),
    Unknown,
}

struct Launch {
    python: PathBuf,
    environment: Option<BTreeMap<String, String>>,
    guarded_launcher: Option<(PathBuf, String)>,
}

fn file_digest(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn profile(path: &Path, digest: &str, host: &str) -> Option<Launch> {
    // The reviewed command pins the whole profile. Do not use a replacement
    // profile to choose either a worker or a fallback executable.
    let mut bytes = Vec::new();
    File::open(path)
        .ok()?
        .take(65_537)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 65_536 || format!("{:x}", Sha256::digest(&bytes)) != digest {
        return None;
    }
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    if value.get("schema_version").and_then(Value::as_u64) != Some(1)
        || value.get("host").and_then(Value::as_str) != Some(host)
    {
        return None;
    }
    let python = PathBuf::from(value.get("python")?.as_str()?);
    let launcher = PathBuf::from(value.get("fallback_launcher")?.as_str()?);
    let expected = value.get("fallback_launcher_sha256")?.as_str()?.to_string();
    let descriptor = PathBuf::from(value.get("worker_descriptor")?.as_str()?);
    if !python.is_absolute()
        || !python.is_file()
        || !launcher.is_absolute()
        || !launcher.is_file()
        || !descriptor.is_absolute()
        || file_digest(&launcher).as_deref() != Some(&expected)
    {
        return None;
    }
    let mut env = environment()?;
    // Exactly the launcher's configuration boundary: discard inherited
    // Mnemos selectors, preserve unrelated environment, then install the
    // frozen host-specific values. Avoid process-global environment mutation.
    env.retain(|key, _| !key.to_uppercase().starts_with("MNEMOS_"));
    for (key, value) in value.get("environment")?.as_object()? {
        let value = value.as_str()?;
        if !key.starts_with("MNEMOS_")
            || key == DESCRIPTOR_ENV
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
            || value.contains('\0')
        {
            return None;
        }
        env.insert(key.to_owned(), value.to_owned());
    }
    env.insert(
        DESCRIPTOR_ENV.to_owned(),
        descriptor.into_os_string().into_string().ok()?,
    );
    Some(Launch {
        python,
        environment: Some(env),
        guarded_launcher: Some((launcher, expected)),
    })
}

// Match json.dumps(mapping, sort_keys=True) for string-only environment data,
// including its spaces, ASCII escaping and UTF-16 surrogate pairs.
fn python_string(value: &str) -> String {
    let mut result = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{08}' => result.push_str("\\b"),
            '\u{0c}' => result.push_str("\\f"),
            ' '..='~' => result.push(c),
            _ => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    result.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    result.push('"');
    result
}

fn fingerprint(values: &BTreeMap<String, String>) -> String {
    let entries: Vec<_> = values
        .iter()
        .filter(|(key, _)| key.to_uppercase() != DESCRIPTOR_ENV)
        .map(|(key, value)| format!("{}: {}", python_string(key), python_string(value)))
        .collect();
    format!(
        "{:x}",
        Sha256::digest(format!("{{{}}}", entries.join(", ")).as_bytes())
    )
}

fn environment() -> Option<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let key = key.into_string().ok()?;
        // Windows keeps hidden drive-directory entries such as "=C:" in
        // the process environment. CPython omits them from os.environ,
        // including when an env=None subprocess still inherits them.
        #[cfg(windows)]
        if key.starts_with('=') {
            continue;
        }
        // CPython's os.environ normalizes Windows keys to uppercase. Keep
        // distinct names on case-sensitive platforms.
        #[cfg(windows)]
        let key = key.to_uppercase();
        values.insert(key, value.into_string().ok()?);
    }
    Some(values)
}

fn descriptor(path: &Path) -> Option<Value> {
    let mut bytes = Vec::new();
    // Rust's Windows File opens with READ | WRITE | DELETE sharing, so a
    // descriptor snapshot does not prevent its owner's atomic replacement.
    File::open(path)
        .ok()?
        .take(65_537)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 65_536 {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}

fn remaining(deadline: Instant) -> Result<Duration, ()> {
    let value = deadline.checked_duration_since(Instant::now()).ok_or(())?;
    if value.is_zero() { Err(()) } else { Ok(value) }
}

fn read_response(stream: &mut TcpStream, deadline: Instant) -> Result<Vec<u8>, ()> {
    let mut header = Vec::new();
    while !header.ends_with(b"\r\n\r\n") {
        if header.len() == MAX_HEADERS {
            return Err(());
        }
        stream
            .set_read_timeout(Some(remaining(deadline)?))
            .map_err(|_| ())?;
        let mut byte = [0];
        stream.read_exact(&mut byte).map_err(|_| ())?;
        if header.is_empty() {
            timing::mark("response_first_byte");
        }
        header.push(byte[0]);
    }
    timing::mark("response_headers_read");
    let text = std::str::from_utf8(&header).map_err(|_| ())?;
    let mut lines = text.split("\r\n");
    let status: Vec<_> = lines.next().ok_or(())?.split_whitespace().collect();
    if status.len() < 2 || !["HTTP/1.0", "HTTP/1.1"].contains(&status[0]) || status[1] != "200" {
        return Err(());
    }
    let mut length = None;
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':').ok_or(())?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(());
        }
        if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(());
        }
        if name.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err(());
            }
            let value = value.trim();
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(());
            }
            length = Some(value.parse::<usize>().map_err(|_| ())?);
        }
    }
    let size = length.ok_or(())?;
    if size > MAX_FRAME {
        return Err(());
    }
    let mut body = vec![0; size];
    let mut offset = 0;
    while offset < size {
        stream
            .set_read_timeout(Some(remaining(deadline)?))
            .map_err(|_| ())?;
        let read = stream.read(&mut body[offset..]).map_err(|_| ())?;
        if read == 0 {
            return Err(());
        }
        offset += read;
    }
    Ok(body)
}

fn request(
    host: &str,
    event: &str,
    payload: Value,
    cwd: &str,
    env: Option<&BTreeMap<String, String>>,
) -> Outcome {
    timing::mark("request_begin");
    let Some(env) = env else {
        return Outcome::Fallback;
    };
    let Some(path) = env.get(DESCRIPTOR_ENV) else {
        return Outcome::Fallback;
    };
    let Some(desc) = descriptor(Path::new(path)) else {
        return Outcome::Fallback;
    };
    let env_hash = fingerprint(env);
    let port = desc.get("port").and_then(Value::as_u64).unwrap_or(0);
    let token = desc.get("token").and_then(Value::as_str).unwrap_or("");
    if desc.get("hook_protocol").and_then(Value::as_u64) != Some(1)
        || desc.get("adapter_host").and_then(Value::as_str) != Some(host)
        || desc.get("host").and_then(Value::as_str) != Some("127.0.0.1")
        || !(1..=65_535).contains(&port)
        || token.is_empty()
        || !token.bytes().all(|byte| (33..=126).contains(&byte))
        || desc.get("environment_sha256").and_then(Value::as_str) != Some(&env_hash)
    {
        return Outcome::Fallback;
    }
    timing::mark("descriptor_validated");
    let identifier = uuid::Uuid::new_v4().to_string();
    let envelope = json!({"hook_protocol":1,"request_id":identifier,"host":host,"event":event,
        "payload":payload,"cwd":cwd,"environment_sha256":env_hash});
    let wire =
        serde_json::to_vec(&json!({"schema_version":1,"operation":"hook","payload":envelope}))
            .unwrap();
    if wire.len() > MAX_FRAME {
        return Outcome::Fallback;
    }
    timing::mark("request_serialized");
    let deadline = Instant::now() + DEADLINE;
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port as u16));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, DEADLINE) else {
        return Outcome::Fallback;
    };
    timing::mark("connected");
    let Ok(timeout) = remaining(deadline) else {
        return Outcome::Fallback;
    };
    if stream.set_write_timeout(Some(timeout)).is_err() {
        return Outcome::Fallback;
    }
    // Fixed loopback destination, one connection, no proxies, redirects,
    // cookies, authentication negotiation or request resend.
    let header = format!(
        "POST /v1/dispatch HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        wire.len()
    );
    let mut outgoing = header.into_bytes();
    outgoing.extend_from_slice(&wire);
    // Every failure after the first send attempt has an unknown outcome,
    // even if the OS reports zero bytes. Never launch Python in this state.
    let mut offset = 0;
    while offset < outgoing.len() {
        let Ok(timeout) = remaining(deadline) else {
            return Outcome::Unknown;
        };
        if stream.set_write_timeout(Some(timeout)).is_err() {
            return Outcome::Unknown;
        }
        match stream.write(&outgoing[offset..]) {
            Ok(0) | Err(_) => return Outcome::Unknown,
            Ok(count) => offset += count,
        }
    }
    timing::mark("request_written");
    let Ok(bytes) = read_response(&mut stream, deadline) else {
        return Outcome::Unknown;
    };
    timing::mark("response_read");
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return Outcome::Unknown;
    };
    if value.get("hook_protocol").and_then(Value::as_u64) != Some(1)
        || value.get("request_id").and_then(Value::as_str) != Some(&identifier)
        || value.get("host").and_then(Value::as_str) != Some(host)
        || value.get("environment_sha256").and_then(Value::as_str) != Some(&env_hash)
    {
        return Outcome::Unknown;
    }
    if value.get("status").and_then(Value::as_str) == Some("declined")
        && value.get("dispatched").and_then(Value::as_bool) == Some(false)
    {
        return Outcome::Fallback;
    }
    if value.get("status").and_then(Value::as_str) != Some("handled")
        || value.get("dispatched").and_then(Value::as_bool) != Some(true)
        || !value.get("output").is_some_and(Value::is_object)
        || !value.get("adapter_status").is_some_and(Value::is_string)
    {
        return Outcome::Unknown;
    }
    timing::mark("response_validated");
    Outcome::Handled(value)
}

fn diagnostic(host: &str, event: &str, status: &str) {
    let event = if [
        "SessionStart",
        "UserPromptSubmit",
        "PreCompact",
        "PostCompact",
        "Stop",
        "SessionEnd",
    ]
    .contains(&event)
    {
        event
    } else {
        "unknown"
    };
    eprintln!("mnemos hook diagnostic host={host} event={event} status={status}");
}

fn fallback(launch: &Launch, host: &str, event: &str, raw: &[u8]) -> i32 {
    timing::mark("fallback_begin");
    let module = if host == "codex" {
        "mnemos.adapters.codex"
    } else {
        "mnemos.adapters.claude_code"
    };
    let mut command = Command::new(&launch.python);
    if let Some((path, expected)) = &launch.guarded_launcher {
        // Recheck after a worker decline, before a separate process reads it.
        if file_digest(path).as_deref() != Some(expected) {
            diagnostic(host, event, "error");
            return 0;
        }
        command.arg("-I").arg(path).args(["hook", host, event]);
        let Some(env) = &launch.environment else {
            return 0;
        };
        command.env_clear().envs(env);
    } else {
        command.args(["-m", module, event]);
    }
    command.stdin(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let Ok(mut child) = command.spawn() else {
        return 1;
    };
    let Some(mut input) = child.stdin.take() else {
        let _ = child.wait();
        return 1;
    };
    let written = input.write_all(raw).is_ok();
    drop(input);
    let status = child.wait();
    if !written {
        return 1;
    }
    status.ok().and_then(|status| status.code()).unwrap_or(1)
}

fn run() -> i32 {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 2 && args[1] == "--fingerprint" {
        // Test input only. Never print environment values.
        let value = serde_json::from_reader::<_, BTreeMap<String, String>>(std::io::stdin());
        if let Ok(value) = value {
            println!("{}", fingerprint(&value));
            return 0;
        }
        return 2;
    }
    if args.len() != 5 && args.len() != 7 {
        return 2;
    }
    let Some(host) = args[1].to_str() else {
        return 2;
    };
    if !["codex", "claude-code"].contains(&host) {
        return 2;
    }
    let Some(event_arg) = args[2].to_str() else {
        return 2;
    };
    let launch = if args.len() == 5 && args[3] == "--python" {
        let python = PathBuf::from(&args[4]);
        if !python.is_absolute() || !python.is_file() {
            return 2;
        }
        Launch {
            python,
            environment: environment(),
            guarded_launcher: None,
        }
    } else if args.len() == 7 && args[3] == "--profile" && args[5] == "--profile-sha256" {
        let Some(digest) = args[6].to_str() else {
            return 2;
        };
        let Some(launch) = profile(Path::new(&args[4]), digest, host) else {
            diagnostic(host, event_arg, "error");
            if host == "codex" && event_arg == "Stop" {
                println!("{{\"continue\":true}}");
            }
            return 0;
        };
        launch
    } else {
        return 2;
    };
    timing::mark("launch_prepared");
    let mut raw = Vec::new();
    if std::io::stdin().read_to_end(&mut raw).is_err() {
        diagnostic(host, event_arg, "error");
        return 0;
    }
    timing::mark("stdin_read");
    let payload: Value = if raw.iter().all(u8::is_ascii_whitespace) {
        json!({})
    } else {
        match serde_json::from_slice(&raw) {
            Ok(value) => value,
            Err(_) => {
                diagnostic(host, event_arg, "error");
                if host == "codex" && event_arg == "Stop" {
                    println!("{{\"continue\":true}}");
                }
                return 0;
            }
        }
    };
    timing::mark("payload_parsed");
    if !payload.is_object() {
        return 0;
    }
    let event = payload
        .get("hook_event_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|event| !event.is_empty())
        .unwrap_or(event_arg)
        .to_string();
    if event.is_empty() {
        return 0;
    }
    let Some(cwd) = std::env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
    else {
        return fallback(&launch, host, &event, &raw);
    };
    match request(host, &event, payload, &cwd, launch.environment.as_ref()) {
        Outcome::Fallback => fallback(&launch, host, &event, &raw),
        Outcome::Handled(value) => {
            if value.get("degraded").and_then(Value::as_bool) == Some(true) {
                diagnostic(host, &event, "degraded");
            }
            if value["output"]
                .as_object()
                .is_some_and(|map| !map.is_empty())
            {
                println!("{}", value["output"]);
            }
            0
        }
        Outcome::Unknown => {
            diagnostic(host, &event, "error");
            if host == "codex" && event == "Stop" {
                println!("{{\"continue\":true}}");
            }
            0
        }
    }
}

fn main() {
    timing::mark("main_begin");
    let code = run();
    timing::mark("run_finished");
    timing::emit();
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn slow_header_cannot_extend_the_overall_deadline() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}" {
                if connection.write_all(&[*byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(15));
            }
        });
        let mut stream = TcpStream::connect(address).unwrap();
        let started = Instant::now();
        assert!(read_response(&mut stream, started + Duration::from_millis(50)).is_err());
        assert!(started.elapsed() < Duration::from_millis(400));
        drop(stream);
        writer.join().unwrap();
    }

    #[test]
    fn slow_body_cannot_extend_the_overall_deadline() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            connection
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n")
                .unwrap();
            for _ in 0..20 {
                if connection.write_all(b"x").is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(15));
            }
        });
        let mut stream = TcpStream::connect(address).unwrap();
        let started = Instant::now();
        assert!(read_response(&mut stream, started + Duration::from_millis(50)).is_err());
        assert!(started.elapsed() < Duration::from_millis(400));
        drop(stream);
        writer.join().unwrap();
    }
}
