use vidarax_contracts::lifecycle::StreamState;
use vidarax_contracts::models::REQUIRED_MODELS;

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("run-create") => {
            println!(r#"{{"run_id":"run-local-00000001","status":"pending","mode":"balanced"}}"#);
        }
        Some("health") => {
            println!(r#"{{"status":"ok"}}"#);
        }
        Some("models") => {
            for model in REQUIRED_MODELS {
                println!("{model}");
            }
        }
        Some("states") => {
            let states = [
                StreamState::Pending,
                StreamState::Processing,
                StreamState::Completed,
                StreamState::Failed,
                StreamState::Cancelled,
                StreamState::Expired,
            ];
            for state in states {
                println!("{state:?}\tterminal={}", state.is_terminal());
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  vidarax-cli run-create");
            eprintln!("  vidarax-cli health");
            eprintln!("  vidarax-cli models");
            eprintln!("  vidarax-cli states");
        }
    }
}
