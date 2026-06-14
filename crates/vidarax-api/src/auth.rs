use axum::http::HeaderMap;
use sha2::{Digest, Sha256};

pub(crate) const HEADER_API_KEY: &str = "x-api-key";
pub(crate) const HEADER_TENANT_ID: &str = "x-tenant-id";

/// Derive the ownership principal from already-authenticated request headers.
///
/// Open mode (`require_api_key=false`) always returns the shared `public`
/// principal. It is development-only and provides no tenant isolation; any
/// supplied `x-api-key` or `x-tenant-id` is ignored unless the API key has
/// already been validated by `SecurityPolicy`.
pub(crate) fn principal_key_from_headers(
    headers: &HeaderMap,
    api_key_authenticated: bool,
) -> String {
    if api_key_authenticated {
        let Some(api_key) = header_value(headers, HEADER_API_KEY) else {
            return "public".to_string();
        };
        // One API key = one tenant; for sub-tenant isolation issue separate keys.
        return format!("api-key:{}", strong_hash_hex(api_key));
    }
    "public".to_string()
}

pub(crate) fn header_value<'a>(headers: &'a HeaderMap, key: &str) -> Option<&'a str> {
    headers.get(key).and_then(|value| value.to_str().ok())
}

pub(crate) fn strong_hash_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String should not fail");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::{principal_key_from_headers, strong_hash_hex, HEADER_API_KEY, HEADER_TENANT_ID};
    use axum::http::HeaderMap;

    #[test]
    fn derives_principal_keys_stably() {
        let mut headers = HeaderMap::new();
        assert_eq!(principal_key_from_headers(&headers, false), "public");

        headers.insert(HEADER_API_KEY, "test-key".parse().unwrap());
        assert_eq!(
            principal_key_from_headers(&headers, true),
            format!("api-key:{}", strong_hash_hex("test-key"))
        );

        headers.insert(HEADER_TENANT_ID, "tenant-a".parse().unwrap());
        assert_eq!(
            principal_key_from_headers(&headers, true),
            format!("api-key:{}", strong_hash_hex("test-key"))
        );
    }

    #[test]
    fn tenant_header_does_not_change_authenticated_principal() {
        let mut key_a = HeaderMap::new();
        key_a.insert(HEADER_API_KEY, "key-a".parse().unwrap());
        key_a.insert(HEADER_TENANT_ID, "tenant-a".parse().unwrap());

        let mut forged = HeaderMap::new();
        forged.insert(HEADER_API_KEY, "key-a".parse().unwrap());
        forged.insert(HEADER_TENANT_ID, "tenant-b".parse().unwrap());

        assert_eq!(
            principal_key_from_headers(&key_a, true),
            principal_key_from_headers(&forged, true),
            "x-tenant-id must not widen or alter ownership"
        );
    }

    #[test]
    fn unauthenticated_headers_do_not_create_public_tenant_principals() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_API_KEY, "unvalidated-key".parse().unwrap());
        headers.insert(HEADER_TENANT_ID, "tenant-a".parse().unwrap());

        assert_eq!(principal_key_from_headers(&headers, false), "public");
    }
}
