use vidarax_api::{run, ServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::from_env().map_err(vidarax_api::invalid_input)?;
    run(config).await
}
