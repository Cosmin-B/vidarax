use axum::http::HeaderMap;

pub(crate) const HEADER_API_KEY: &str = "x-api-key";
pub(crate) const HEADER_TENANT_ID: &str = "x-tenant-id";

pub(crate) fn principal_key_from_headers(headers: &HeaderMap) -> String {
    if let Some(tenant_id) = header_value(headers, HEADER_TENANT_ID) {
        return format!("tenant:{tenant_id}");
    }
    if let Some(api_key) = header_value(headers, HEADER_API_KEY) {
        return format!("api-key:{:016x}", fnv1a64(api_key));
    }
    "public".to_string()
}

pub(crate) fn header_value<'a>(headers: &'a HeaderMap, key: &str) -> Option<&'a str> {
    headers.get(key).and_then(|value| value.to_str().ok())
}

pub(crate) fn fnv1a64(value: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for b in value.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{fnv1a64, principal_key_from_headers, HEADER_API_KEY, HEADER_TENANT_ID};
    use axum::http::HeaderMap;

    #[test]
    fn derives_principal_keys_stably() {
        let mut headers = HeaderMap::new();
        assert_eq!(principal_key_from_headers(&headers), "public");

        headers.insert(HEADER_API_KEY, "test-key".parse().unwrap());
        assert_eq!(
            principal_key_from_headers(&headers),
            format!("api-key:{:016x}", fnv1a64("test-key"))
        );

        headers.insert(HEADER_TENANT_ID, "tenant-a".parse().unwrap());
        assert_eq!(principal_key_from_headers(&headers), "tenant:tenant-a");
    }
}
