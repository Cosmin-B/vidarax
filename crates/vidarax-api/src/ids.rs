pub fn random_run_id(fallback_seq: u64) -> String {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_ok() {
        let mut id = String::with_capacity(4 + 32);
        id.push_str("run-");
        for b in &bytes {
            id.push(hex_char(b >> 4));
            id.push(hex_char(b & 0x0f));
        }
        return id;
    }
    // Fallback preserves availability if OS RNG is temporarily unavailable.
    format!("run-{fallback_seq:016x}")
}

pub fn validate_run_id(run_id: &str) -> bool {
    (run_id.len() == 20 || run_id.len() == 36)
        && run_id.starts_with("run-")
        && run_id[4..].chars().all(|ch| ch.is_ascii_hexdigit())
}

pub fn parse_run_sequence(run_id: &str) -> Option<u64> {
    (run_id.len() == 20).then(|| u64::from_str_radix(&run_id[4..], 16).ok())?
}

#[inline]
fn hex_char(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        _ => (b'a' + (value - 10)) as char,
    }
}
