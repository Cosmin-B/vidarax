use std::{future::Future, io};

use axum::Router;

use crate::config::ServerConfig;

pub async fn serve_h1h2(addr: &str, app: Router) -> io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve_h1h2_with_shutdown(listener, app, shutdown_signal()).await
}

async fn serve_h1h2_with_shutdown(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    let addr = listener.local_addr()?;
    tracing::info!(addr = %addr, "vidarax-api h1/h2 listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = match tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        ) {
            Ok(sigterm) => sigterm,
            Err(err) => {
                tracing::warn!(%err, "failed to install SIGTERM handler; waiting for SIGINT only");
                wait_for_ctrl_c().await;
                return;
            }
        };

        tokio::select! {
            _ = wait_for_ctrl_c() => {}
            _ = sigterm.recv() => {
                tracing::info!(signal = "SIGTERM", "vidarax-api shutdown signal received");
            }
        }
    }

    #[cfg(not(unix))]
    {
        wait_for_ctrl_c().await;
    }
}

async fn wait_for_ctrl_c() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::warn!(%err, "failed to wait for SIGINT");
        return;
    }
    tracing::info!(signal = "SIGINT", "vidarax-api shutdown signal received");
}

#[cfg(feature = "h3-experimental")]
use axum::body::Body;
#[cfg(feature = "h3-experimental")]
use axum::http::{HeaderName, HeaderValue, Method, Request, Response};
#[cfg(feature = "h3-experimental")]
use futures_util::{SinkExt, StreamExt};
#[cfg(feature = "h3-experimental")]
use http_body_util::BodyExt;
#[cfg(feature = "h3-experimental")]
use serde_json::json;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::buf_factory::BufFactory;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::http3::driver::{
    H3Event, InboundFrame, InboundFrameStream, IncomingH3Headers, OutboundFrame,
    OutboundFrameSender, ServerEventStream, ServerH3Event,
};
#[cfg(feature = "h3-experimental")]
use tokio_quiche::http3::settings::Http3Settings;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::metrics::DefaultMetrics;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::quiche::h3::Header;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::quiche::h3::NameValue;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::settings::{CertificateKind, Hooks, QuicSettings, TlsCertificatePaths};
#[cfg(feature = "h3-experimental")]
use tokio_quiche::{listen, ConnectionParams, ServerH3Driver};
#[cfg(feature = "h3-experimental")]
use tower::util::ServiceExt;

#[cfg(feature = "h3-experimental")]
const MAX_H3_BODY_BYTES: usize = 4 * 1024 * 1024;

#[cfg(feature = "h3-experimental")]
pub async fn serve_h3_experimental(config: &ServerConfig, app: Router) -> io::Result<()> {
    let h1_addr = config.bind_addr.clone();
    let h1_app = app.clone();
    // Both listeners register the same process-level shutdown signal; h3 stops
    // accepting here while the h1/h2 task drains through axum.
    let _h1_task = tokio::spawn(async move { serve_h1h2(&h1_addr, h1_app).await });

    let socket = tokio::net::UdpSocket::bind(&config.h3_bind_addr).await?;
    let mut listeners = listen(
        [socket],
        ConnectionParams::new_server(
            QuicSettings::default(),
            TlsCertificatePaths {
                cert: &config.h3_tls_cert_path,
                private_key: &config.h3_tls_key_path,
                kind: CertificateKind::X509,
            },
            Hooks::default(),
        ),
        DefaultMetrics,
    )?;

    tracing::info!(addr = %config.h3_bind_addr, "vidarax-api h3 listening");

    let accepted_connection_stream = &mut listeners[0];
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    loop {
        let conn_res = tokio::select! {
            _ = &mut shutdown => break,
            conn_res = accepted_connection_stream.next() => conn_res,
        };

        let Some(conn_res) = conn_res else {
            break;
        };

        let conn = match conn_res {
            Ok(conn) => conn,
            Err(err) => {
                tracing::error!(%err, "vidarax-api h3 accept error");
                continue;
            }
        };

        let (driver, mut controller) = ServerH3Driver::new(Http3Settings::default());
        conn.start(driver);

        let app = app.clone();
        tokio::spawn(async move {
            let event_rx = controller.event_receiver_mut();
            if let Err(err) = serve_h3_connection(app, event_rx).await {
                tracing::error!(%err, "vidarax-api h3 connection error");
            }
        });
    }

    Ok(())
}

#[cfg(feature = "h3-experimental")]
async fn serve_h3_connection(
    app: Router,
    h3_event_receiver: &mut ServerEventStream,
) -> io::Result<()> {
    while let Some(event) = h3_event_receiver.recv().await {
        match event {
            ServerH3Event::Core(H3Event::ConnectionShutdown(_)) => return Ok(()),
            ServerH3Event::Core(H3Event::ConnectionError(err)) => {
                return Err(io::Error::other(err.to_string()))
            }
            ServerH3Event::Headers {
                incoming_headers, ..
            } => {
                let app = app.clone();
                tokio::spawn(async move {
                    handle_h3_headers(app, incoming_headers).await;
                });
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(feature = "h3-experimental")]
async fn handle_h3_headers(app: Router, headers: IncomingH3Headers) {
    let IncomingH3Headers {
        headers: header_list,
        send: mut frame_sender,
        recv,
        read_fin,
        ..
    } = headers;

    let request = match build_http_request_from_h3(header_list, recv, read_fin).await {
        Ok(request) => request,
        Err(message) => {
            send_h3_error_json(
                &mut frame_sender,
                400,
                json!({ "error": { "code": "bad_request", "message": message } }),
            )
            .await;
            return;
        }
    };

    let response = app
        .oneshot(request)
        .await
        .expect("axum router dispatch is infallible");

    send_h3_response(frame_sender, response).await;
}

#[cfg(feature = "h3-experimental")]
async fn build_http_request_from_h3(
    headers: Vec<Header>,
    mut recv: InboundFrameStream,
    read_fin: bool,
) -> Result<Request<Body>, String> {
    let (method, path, header_map) = parse_h3_request_head(&headers)?;
    let body = read_h3_request_body(&mut recv, read_fin).await?;

    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::from(body))
        .map_err(|err| err.to_string())?;

    for (name, value) in header_map {
        request.headers_mut().insert(name, value);
    }

    Ok(request)
}

#[cfg(feature = "h3-experimental")]
fn parse_h3_request_head(
    headers: &[Header],
) -> Result<(Method, String, Vec<(HeaderName, HeaderValue)>), String> {
    let mut method: Option<Method> = None;
    let mut path: Option<String> = None;
    let mut normal_headers: Vec<(HeaderName, HeaderValue)> = Vec::with_capacity(headers.len());

    for header in headers {
        let name = header.name();
        let value = header.value();
        if name.first() == Some(&b':') {
            match name {
                b":method" => {
                    method = Some(Method::from_bytes(value).map_err(|err| err.to_string())?);
                }
                b":path" => {
                    let parsed = std::str::from_utf8(value)
                        .map_err(|_| "invalid :path utf-8".to_string())?;
                    path = Some(parsed.to_string());
                }
                _ => {}
            }
            continue;
        }

        let name = HeaderName::from_bytes(name).map_err(|err| err.to_string())?;
        let value = HeaderValue::from_bytes(value).map_err(|err| err.to_string())?;
        normal_headers.push((name, value));
    }

    let method = method.ok_or_else(|| "missing :method pseudo-header".to_string())?;
    let path = path.ok_or_else(|| "missing :path pseudo-header".to_string())?;

    Ok((method, path, normal_headers))
}

#[cfg(feature = "h3-experimental")]
async fn read_h3_request_body(
    recv: &mut InboundFrameStream,
    read_fin: bool,
) -> Result<Vec<u8>, String> {
    if read_fin {
        return Ok(Vec::new());
    }

    // Reserve a small default and grow only when needed; this keeps hot-path
    // allocations bounded for common tiny JSON payloads.
    let mut body = Vec::with_capacity(1024);
    while let Some(frame) = recv.recv().await {
        if let InboundFrame::Body(chunk, fin) = frame {
            let bytes = chunk.as_ref();
            if body.len().saturating_add(bytes.len()) > MAX_H3_BODY_BYTES {
                return Err(format!("request body exceeds {} bytes", MAX_H3_BODY_BYTES));
            }
            body.extend_from_slice(bytes);
            if fin {
                return Ok(body);
            }
        }
    }

    Ok(body)
}

#[cfg(feature = "h3-experimental")]
async fn send_h3_response(mut frame_sender: OutboundFrameSender, response: Response<Body>) {
    let (parts, body) = response.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            send_h3_error_json(
                &mut frame_sender,
                500,
                json!({
                    "error": {
                        "code": "internal_error",
                        "message": format!("failed to read response body: {err}")
                    }
                }),
            )
            .await;
            return;
        }
    };

    let mut h3_headers = vec![Header::new(b":status", parts.status.as_str().as_bytes())];
    for (name, value) in &parts.headers {
        h3_headers.push(Header::new(name.as_str().as_bytes(), value.as_bytes()));
    }
    if frame_sender
        .send(OutboundFrame::Headers(h3_headers, None))
        .await
        .is_err()
    {
        return;
    }

    let body_frame = if body_bytes.is_empty() {
        OutboundFrame::body(BufFactory::get_empty_buf(), true)
    } else {
        OutboundFrame::body(BufFactory::buf_from_slice(body_bytes.as_ref()), true)
    };
    let _ = frame_sender.send(body_frame).await;
}

#[cfg(feature = "h3-experimental")]
async fn send_h3_error_json(
    frame_sender: &mut OutboundFrameSender,
    status: u16,
    payload: serde_json::Value,
) {
    let response = Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("error response must be constructible");

    let (parts, body) = response.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return,
    };

    let mut h3_headers = vec![Header::new(b":status", parts.status.as_str().as_bytes())];
    for (name, value) in &parts.headers {
        h3_headers.push(Header::new(name.as_str().as_bytes(), value.as_bytes()));
    }
    if frame_sender
        .send(OutboundFrame::Headers(h3_headers, None))
        .await
        .is_err()
    {
        return;
    }

    let body_frame = if body_bytes.is_empty() {
        OutboundFrame::body(BufFactory::get_empty_buf(), true)
    } else {
        OutboundFrame::body(BufFactory::buf_from_slice(body_bytes.as_ref()), true)
    };
    let _ = frame_sender.send(body_frame).await;
}

#[cfg(not(feature = "h3-experimental"))]
pub async fn serve_h3_experimental(_config: &ServerConfig, _app: Router) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "h3 transport requested but binary was built without feature `h3-experimental`",
    ))
}

#[cfg(test)]
mod tests {
    use super::serve_h1h2_with_shutdown;
    use axum::{routing::get, Router};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn injected_shutdown_stops_h1h2_server_promptly() {
        let app = Router::new().route("/v1/health", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test requires loopback bind");
        let addr = listener.local_addr().expect("listener should have local addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let serve_task = tokio::spawn(serve_h1h2_with_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
        ));

        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("server should accept loopback connection");
        stream
            .write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .expect("health request should write");

        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .expect("health response timed out")
            .expect("health response should read");
        let response = String::from_utf8_lossy(&response);
        assert!(response.contains("200 OK"), "unexpected response: {response}");
        assert!(response.contains("ok"), "unexpected response body: {response}");

        shutdown_tx.send(()).expect("server task should still be live");
        let serve_result = tokio::time::timeout(Duration::from_secs(2), serve_task)
            .await
            .expect("server did not stop after injected shutdown")
            .expect("server task panicked");

        assert!(serve_result.is_ok(), "server returned error: {serve_result:?}");
    }

    #[tokio::test]
    async fn injected_shutdown_drains_in_flight_request() {
        let (started_tx, mut started_rx) = tokio::sync::mpsc::channel(1);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let release_rx = Arc::new(Mutex::new(Some(release_rx)));
        let handler_completed = Arc::new(AtomicBool::new(false));

        let release_rx_for_handler = Arc::clone(&release_rx);
        let handler_completed_for_handler = Arc::clone(&handler_completed);
        let app = Router::new().route(
            "/v1/slow",
            get(move || {
                let started_tx = started_tx.clone();
                let release_rx = Arc::clone(&release_rx_for_handler);
                let handler_completed = Arc::clone(&handler_completed_for_handler);

                async move {
                    let release_rx = {
                        let mut release_rx =
                            release_rx.lock().expect("release receiver lock poisoned");
                        release_rx.take().expect("slow handler should run once")
                    };

                    started_tx
                        .send(())
                        .await
                        .expect("test should wait for handler start");
                    let _ = release_rx.await;
                    handler_completed.store(true, Ordering::SeqCst);
                    "slow-ok"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test requires loopback bind");
        let addr = listener.local_addr().expect("listener should have local addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let serve_task = tokio::spawn(serve_h1h2_with_shutdown(
            listener,
            app,
            async move {
                let _ = shutdown_rx.await;
            },
        ));

        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("server should accept loopback connection");
        stream
            .write_all(b"GET /v1/slow HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .expect("slow request should write");

        tokio::time::timeout(Duration::from_secs(2), started_rx.recv())
            .await
            .expect("slow handler did not start")
            .expect("slow handler start channel closed");

        shutdown_tx.send(()).expect("server task should still be live");

        // Graceful shutdown must stop accepting new work without resolving the
        // serve future until the already-running handler has completed.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !serve_task.is_finished(),
            "server returned before in-flight handler completed"
        );
        assert!(
            !handler_completed.load(Ordering::SeqCst),
            "handler completed before the test released it"
        );

        release_tx.send(()).expect("slow handler should still wait");

        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .expect("slow response timed out")
            .expect("slow response should read");
        let response = String::from_utf8_lossy(&response);
        assert!(response.contains("200 OK"), "unexpected response: {response}");
        assert!(
            response.contains("slow-ok"),
            "unexpected response body: {response}"
        );

        let serve_result = tokio::time::timeout(Duration::from_secs(2), serve_task)
            .await
            .expect("server did not stop after in-flight request drained")
            .expect("server task panicked");

        assert!(serve_result.is_ok(), "server returned error: {serve_result:?}");
    }
}
