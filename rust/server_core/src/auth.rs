use axum::http::HeaderMap;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, Mac};
use rand::Rng;
use rand::RngCore;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

pub const CLIENT_COOKIE: &str = "hs_client";
pub const CLIENT_COOKIE_TTL_MS: i64 = 24 * 60 * 60 * 1000;
pub const SESSION_TTL_MS: i64 = 30 * 24 * 60 * 60 * 1000;

const CODE_CHARS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn to_base36(mut value: u64) -> String {
    if value == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::with_capacity(13);
    while value > 0 {
        buf.push(DIGITS[(value % 36) as usize]);
        value /= 36;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_else(|_| "0".to_string())
}

pub fn from_base36(s: &str) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    i64::from_str_radix(s, 36).ok()
}

pub fn sign_client_token(secret: &str, ts: &str, nonce: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC-SHA256 accepts keys of any size");
    mac.update(ts.as_bytes());
    mac.update(b".");
    mac.update(nonce.as_bytes());
    let digest = mac.finalize().into_bytes();
    URL_SAFE_NO_PAD.encode(digest)
}

pub fn issue_client_cookie(secret: &str, expected_origin: &str) -> String {
    let ts = to_base36(now_ms().max(0) as u64);
    let mut nonce_bytes = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);
    let sig = sign_client_token(secret, &ts, &nonce);
    let secure = if expected_origin.starts_with("https:") {
        "; Secure"
    } else {
        ""
    };
    format!(
        "{CLIENT_COOKIE}={ts}.{nonce}.{sig}; Path=/; HttpOnly; SameSite=Strict; Max-Age={}{secure}",
        CLIENT_COOKIE_TTL_MS / 1000
    )
}

pub fn read_cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    for header_val in headers.get_all("cookie") {
        let Ok(header_str) = header_val.to_str() else {
            continue;
        };
        for part in header_str.split(';') {
            if let Some((k, v)) = part.split_once('=')
                && k.trim() == name {
                    return Some(v.trim().to_string());
                }
        }
    }
    None
}

pub fn verify_client_cookie_value(secret: &str, value: &str, now: i64) -> bool {
    let mut parts = value.split('.');
    let (Some(ts), Some(nonce), Some(sig), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let Some(issued) = from_base36(ts) else {
        return false;
    };
    if now - issued > CLIENT_COOKIE_TTL_MS || issued - now > 60_000 {
        return false;
    }
    let expected = sign_client_token(secret, ts, nonce);
    let expected_bytes = expected.as_bytes();
    let given_bytes = sig.as_bytes();
    expected_bytes.len() == given_bytes.len() && bool::from(expected_bytes.ct_eq(given_bytes))
}

pub fn has_valid_client_cookie(secret: &str, headers: &HeaderMap) -> bool {
    match read_cookie(headers, CLIENT_COOKIE) {
        Some(val) => verify_client_cookie_value(secret, &val, now_ms()),
        None => false,
    }
}

/// Matches Node's `crypto.scrypt(password, salt, 64)` with default parameters:
/// `N = 16384` (`log_n = 14`), `r = 8`, `p = 1`, `dkLen = 64`, `salt` passed as UTF-8 bytes.
pub fn hash_password_sync(password: &str, salt: &str) -> Result<String, String> {
    let params = scrypt::Params::new(14, 8, 1, 64).map_err(|e| e.to_string())?;
    let mut out = [0u8; 64];
    scrypt::scrypt(password.as_bytes(), salt.as_bytes(), &params, &mut out)
        .map_err(|e| e.to_string())?;
    Ok(hex::encode(out))
}

pub async fn hash_password(password: String, salt: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || hash_password_sync(&password, &salt))
        .await
        .map_err(|e| e.to_string())?
}

pub async fn verify_password(password: String, salt: String, expected_hex: String) -> bool {
    tokio::task::spawn_blocking(move || {
        let Ok(actual_hex) = hash_password_sync(&password, &salt) else {
            return false;
        };
        let (Ok(actual), Ok(expected)) = (hex::decode(&actual_hex), hex::decode(&expected_hex))
        else {
            return false;
        };
        actual.len() == expected.len() && bool::from(actual.ct_eq(&expected))
    })
    .await
    .unwrap_or(false)
}

pub fn random_hex(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

pub fn random_code(len: usize, prefix: &str) -> String {
    let mut out = String::with_capacity(prefix.len() + len);
    out.push_str(prefix);
    let mut rng = rand::rngs::OsRng;
    for _ in 0..len {
        let idx = rng.gen_range(0..CODE_CHARS.len());
        out.push(CODE_CHARS[idx] as char);
    }
    out
}

pub fn get_bearer_token(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get("authorization")?.to_str().ok()?;
    let rest = raw.strip_prefix("Bearer ")?;
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrypt_matches_node_crypto_vector() {
        // Generated via:
        // node -e "const c=require('crypto'); console.log(c.scryptSync('secret123', 'c0b619b0a7404ac1ac9427b856382df1', 64).toString('hex'))"
        let expected = "ac038ade350c957914c39c92074ffb0f4c784574e6e2a3923cc2ebad8c07df487055bc1c51d3698731cbf96083e2fc90fe3ffa4aa0f5db82cb531c3a5ebf1263";
        let actual = hash_password_sync("secret123", "c0b619b0a7404ac1ac9427b856382df1").unwrap();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn verify_password_accepts_good_and_rejects_bad() {
        let expected = "ac038ade350c957914c39c92074ffb0f4c784574e6e2a3923cc2ebad8c07df487055bc1c51d3698731cbf96083e2fc90fe3ffa4aa0f5db82cb531c3a5ebf1263";
        assert!(
            verify_password(
                "secret123".into(),
                "c0b619b0a7404ac1ac9427b856382df1".into(),
                expected.into()
            )
            .await
        );
        assert!(
            !verify_password(
                "wrongpass".into(),
                "c0b619b0a7404ac1ac9427b856382df1".into(),
                expected.into()
            )
            .await
        );
    }

    #[test]
    fn client_cookie_roundtrip_and_expiry() {
        let secret = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let cookie_hdr = issue_client_cookie(secret, "https://hyperstrike.fly.dev");
        assert!(cookie_hdr.contains("; Secure"));
        assert!(cookie_hdr.contains("; HttpOnly; SameSite=Strict; Max-Age=86400"));

        let cookie_val = cookie_hdr
            .split(';')
            .next()
            .unwrap()
            .strip_prefix("hs_client=")
            .unwrap();
        assert!(verify_client_cookie_value(secret, cookie_val, now_ms()));
        assert!(!verify_client_cookie_value(
            "wrong_secret",
            cookie_val,
            now_ms()
        ));
        assert!(!verify_client_cookie_value(
            secret,
            cookie_val,
            now_ms() + CLIENT_COOKIE_TTL_MS + 1000
        ));
    }
}
