use axum::http::HeaderMap;
use rand::RngCore;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub data_dir: PathBuf,
    pub db_file: PathBuf,
    pub index_file: PathBuf,
    pub region: String,
    pub allowed_origin: String,
    pub allowed_host: String,
    pub client_secret: String,
    pub sim_workers: usize,
    pub trust_proxy: bool,
}

impl Config {
    pub fn from_env(base_dir: &Path) -> Self {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(8080);

        let data_dir = std::env::var("DATA_DIR")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| base_dir.join("data"));

        let db_file = data_dir.join("db.json");
        let index_file = base_dir.join("index.html");

        let region = std::env::var("FLY_REGION")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "local".to_string());

        let allowed_url_raw = std::env::var("URL").unwrap_or_default();
        let (allowed_origin, allowed_host) = if allowed_url_raw.trim().is_empty() {
            (String::new(), String::new())
        } else if let Some(parsed) = parse_origin_and_host(allowed_url_raw.trim()) {
            parsed
        } else {
            eprintln!("[CONFIG] Invalid URL env: {}", allowed_url_raw.trim());
            (String::new(), String::new())
        };

        let client_secret = load_or_create_client_secret(
            std::env::var("CLIENT_SECRET").ok().filter(|s| !s.is_empty()),
            &data_dir,
        );

        let sim_workers = detect_worker_count(std::env::var("SIM_WORKERS").ok().as_deref());

        let trust_proxy = std::env::var("FLY_APP_NAME")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
            || std::env::var("TRUST_PROXY")
                .map(|v| v == "1")
                .unwrap_or(false);

        Self {
            port,
            data_dir,
            db_file,
            index_file,
            region,
            allowed_origin,
            allowed_host,
            client_secret,
            sim_workers,
            trust_proxy,
        }
    }

    /// The origin the game page must be loaded from. Uses `URL` env when set,
    /// otherwise the server's own `scheme://host`.
    pub fn expected_origin(&self, headers: &HeaderMap, is_tls: bool) -> String {
        if !self.allowed_origin.is_empty() {
            return self.allowed_origin.clone();
        }
        let host = request_host_with_proxy(headers, self.trust_proxy);
        if host.is_empty() {
            return String::new();
        }
        let proto = if self.trust_proxy {
            headers
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.split(',').next())
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| s == "http" || s == "https")
        } else {
            None
        }
        .unwrap_or_else(|| if is_tls { "https".into() } else { "http".into() });
        format!("{proto}://{host}")
    }

    pub fn is_request_allowed(&self, headers: &HeaderMap, is_tls: bool) -> bool {
        let origin = header_str(headers, "origin").trim().to_ascii_lowercase();
        if !origin.is_empty() {
            return origin == self.expected_origin(headers, is_tls);
        }
        if self.allowed_origin.is_empty() {
            return true;
        }
        let host = request_host_with_proxy(headers, self.trust_proxy);
        if host == self.allowed_host {
            return true;
        }
        let referer = header_str(headers, "referer").trim().to_ascii_lowercase();
        !referer.is_empty() && (referer == self.allowed_origin
                || referer.starts_with(&format!("{}/", self.allowed_origin)))
    }

    pub fn cors_origin_header(&self) -> Option<&str> {
        if self.allowed_origin.is_empty() {
            None
        } else {
            Some(&self.allowed_origin)
        }
    }

    pub fn get_client_ip(&self, headers: &HeaderMap, remote_addr: Option<SocketAddr>) -> String {
        if self.trust_proxy {
            let fwd = headers
                .get("fly-client-ip")
                .or_else(|| headers.get("x-forwarded-for"))
                .and_then(|v| v.to_str().ok());
            if let Some(raw) = fwd
                && let Some(first) = raw.split(',').next() {
                    let trimmed = first.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
        }
        remote_addr
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

pub fn request_host_with_proxy(headers: &HeaderMap, trust_proxy: bool) -> String {
    let raw = if trust_proxy {
        headers
            .get("x-forwarded-host")
            .or_else(|| headers.get("host"))
    } else {
        headers.get("host")
    }
    .and_then(|v| v.to_str().ok())
    .unwrap_or("");
    raw.split(',')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

pub fn request_host(headers: &HeaderMap) -> String {
    request_host_with_proxy(headers, true)
}

pub fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
}

pub fn detect_worker_count(env_raw: Option<&str>) -> usize {
    if let Some(raw) = env_raw
        && let Ok(parsed) = raw.trim().parse::<usize>()
            && parsed >= 1 {
                return parsed.min(64);
            }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

pub fn load_or_create_client_secret(env_secret: Option<String>, data_dir: &Path) -> String {
    if let Some(secret) = env_secret
        && !secret.is_empty() {
            return secret;
        }
    let file = data_dir.join("client_secret");
    if let Ok(existing) = std::fs::read_to_string(&file) {
        let trimmed = existing.trim();
        if trimmed.len() >= 32 {
            return trimmed.to_string();
        }
    }
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let fresh = hex::encode(bytes);
    if let Err(err) = std::fs::create_dir_all(data_dir).and_then(|_| std::fs::write(&file, &fresh))
    {
        eprintln!("[CONFIG] Could not persist client secret: {err}");
    }
    fresh
}

/// Parses an HTTP/HTTPS URL into `(origin, host)`, matching JS `new URL(raw).origin` and `.host`.
pub fn parse_origin_and_host(raw: &str) -> Option<(String, String)> {
    let (scheme, rest) = raw.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.is_empty() {
        return None;
    }
    // Strip userinfo if present
    let host_port = authority.rsplit('@').next()?;
    if host_port.is_empty() {
        return None;
    }
    let host_port_lower = host_port.to_ascii_lowercase();
    let normalized_host = normalize_default_port(&scheme, &host_port_lower)?;
    let origin = format!("{scheme}://{normalized_host}");
    Some((origin, normalized_host))
}

fn normalize_default_port(scheme: &str, host_port: &str) -> Option<String> {
    if host_port.starts_with('[') {
        // IPv6 literal: [::1] or [::1]:8080
        let close = host_port.find(']')?;
        let host = &host_port[..=close];
        let after = &host_port[close + 1..];
        if after.is_empty() {
            return Some(host.to_string());
        }
        let port_str = after.strip_prefix(':')?;
        let port: u16 = port_str.parse().ok()?;
        if (scheme == "http" && port == 80) || (scheme == "https" && port == 443) {
            return Some(host.to_string());
        }
        return Some(format!("{host}:{port}"));
    }

    if let Some((host, port_str)) = host_port.rsplit_once(':') {
        if host.is_empty() {
            return None;
        }
        if port_str.is_empty() {
            return Some(host.to_string());
        }
        let port: u16 = port_str.parse().ok()?;
        if (scheme == "http" && port == 80) || (scheme == "https" && port == 443) {
            return Some(host.to_string());
        }
        return Some(format!("{host}:{port}"));
    }

    Some(host_port.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn parses_origin_and_host_like_js_url() {
        assert_eq!(
            parse_origin_and_host("https://HyperStrike.Fly.dev:443/play?x=1"),
            Some((
                "https://hyperstrike.fly.dev".into(),
                "hyperstrike.fly.dev".into()
            ))
        );
        assert_eq!(
            parse_origin_and_host("http://localhost:8080/"),
            Some(("http://localhost:8080".into(), "localhost:8080".into()))
        );
        assert_eq!(
            parse_origin_and_host("http://example.com:80"),
            Some(("http://example.com".into(), "example.com".into()))
        );
        assert_eq!(parse_origin_and_host("not a url"), None);
    }

    #[test]
    fn request_allowed_checks_origin_host_referer() {
        let cfg = Config {
            port: 8080,
            data_dir: PathBuf::from("data"),
            db_file: PathBuf::from("data/db.json"),
            index_file: PathBuf::from("index.html"),
            region: "local".into(),
            allowed_origin: "https://hyperstrike.fly.dev".into(),
            allowed_host: "hyperstrike.fly.dev".into(),
            client_secret: "secret".into(),
            sim_workers: 1,
            trust_proxy: false,
        };

        let mut h = HeaderMap::new();
        h.insert("origin", HeaderValue::from_static("https://hyperstrike.fly.dev"));
        assert!(cfg.is_request_allowed(&h, false));

        h.insert("origin", HeaderValue::from_static("https://evil.example"));
        assert!(!cfg.is_request_allowed(&h, false));

        h.remove("origin");
        h.insert("host", HeaderValue::from_static("hyperstrike.fly.dev"));
        assert!(cfg.is_request_allowed(&h, false));

        h.insert("host", HeaderValue::from_static("other.host"));
        assert!(!cfg.is_request_allowed(&h, false));

        h.insert(
            "referer",
            HeaderValue::from_static("https://hyperstrike.fly.dev/index.html"),
        );
        assert!(cfg.is_request_allowed(&h, false));
    }

    #[test]
    fn forwarded_headers_ignored_when_trust_proxy_is_false() {
        let mut cfg = Config {
            port: 8080,
            data_dir: PathBuf::from("data"),
            db_file: PathBuf::from("data/db.json"),
            index_file: PathBuf::from("index.html"),
            region: "local".into(),
            allowed_origin: String::new(),
            allowed_host: String::new(),
            client_secret: "secret".into(),
            sim_workers: 1,
            trust_proxy: false,
        };

        let mut h = HeaderMap::new();
        h.insert("host", HeaderValue::from_static("localhost:8080"));
        h.insert("x-forwarded-host", HeaderValue::from_static("evil.example"));
        h.insert("x-forwarded-proto", HeaderValue::from_static("https"));
        h.insert("origin", HeaderValue::from_static("https://evil.example"));

        assert_eq!(cfg.expected_origin(&h, false), "http://localhost:8080");
        assert!(!cfg.is_request_allowed(&h, false));

        cfg.trust_proxy = true;
        assert_eq!(cfg.expected_origin(&h, false), "https://evil.example");
        assert!(cfg.is_request_allowed(&h, false));
    }
}
