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

#[cfg(any(test, feature = "h3-experimental"))]
async fn serve_owned_listeners(
    h1: impl Future<Output = io::Result<()>>,
    h3: impl Future<Output = io::Result<()>>,
    shutdown: impl Future<Output = ()>,
    stop: tokio::sync::watch::Sender<bool>,
) -> io::Result<()> {
    tokio::pin!(h1, h3, shutdown);
    let (h1_result, h3_result) = tokio::select! {
        result = &mut h1 => {
            let _ = stop.send(true);
            (result, h3.await)
        }
        result = &mut h3 => {
            let _ = stop.send(true);
            (h1.await, result)
        }
        _ = &mut shutdown => {
            let _ = stop.send(true);
            tokio::join!(h1, h3)
        }
    };
    h1_result.and(h3_result)
}

#[cfg(any(test, feature = "h3-experimental"))]
async fn drain_owned_tasks(tasks: &mut tokio::task::JoinSet<()>) -> io::Result<()> {
    let mut first_error = None;
    while let Some(result) = tasks.join_next().await {
        if let Err(err) = result {
            first_error.get_or_insert_with(|| io::Error::other(err));
        }
    }
    first_error.map_or(Ok(()), Err)
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
    OutboundFrameSender, ServerEventStream, ServerH3Controller, ServerH3Event,
};
#[cfg(feature = "h3-experimental")]
use tokio_quiche::http3::settings::Http3Settings;
#[cfg(feature = "h3-experimental")]
use tokio_quiche::metrics::DefaultMetrics;

#[cfg(feature = "h3-experimental")]
type ParsedRequestHead = (Method, String, Vec<(HeaderName, HeaderValue)>);
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
    // Bind both transports before running either listener so startup errors
    // return to the caller without leaving a detached server behind.
    let h1_listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
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
    let (stop_tx, mut h1_stop) = tokio::sync::watch::channel(false);
    let mut h3_stop = h1_stop.clone();
    let h3_stop_tx = stop_tx.clone();
    let h1 = serve_h1h2_with_shutdown(h1_listener, app.clone(), async move {
        let _ = h1_stop.changed().await;
    });
    let h3 = async move {
        let mut connections = tokio::task::JoinSet::new();
        loop {
            let conn_res = tokio::select! {
                biased;
                _ = h3_stop.changed() => break,
                result = connections.join_next(), if !connections.is_empty() => {
                    if let Some(Err(err)) = result {
                        tracing::error!(%err, "vidarax-api h3 connection task error");
                    }
                    continue;
                }
                conn_res = listeners[0].next() => conn_res,
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
            let (driver, controller) = ServerH3Driver::new(Http3Settings::default());
            conn.start(driver);
            let app = app.clone();
            let stop = h3_stop.clone();
            connections.spawn(async move {
                if let Err(err) = serve_h3_connection(app, controller, stop).await {
                    tracing::error!(%err, "vidarax-api h3 connection error");
                }
            });
        }
        let _ = h3_stop_tx.send(true);
        // Keep the UDP listener alive while existing streams finish sending.
        let result = drain_owned_tasks(&mut connections).await;
        drop(listeners);
        result
    };
    serve_owned_listeners(h1, h3, shutdown_signal(), stop_tx).await
}

#[cfg(feature = "h3-experimental")]
async fn serve_h3_connection(
    app: Router,
    mut controller: ServerH3Controller,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> io::Result<()> {
    let mut event_rx: ServerEventStream = controller.take_event_receiver();
    let mut requests = tokio::task::JoinSet::new();
    let mut connection_result = Ok(());
    loop {
        let event = tokio::select! {
            biased;
            _ = stop.changed() => break,
            result = requests.join_next(), if !requests.is_empty() => {
                if let Some(Err(err)) = result {
                    tracing::error!(%err, "vidarax-api h3 request task error");
                }
                continue;
            }
            event = event_rx.recv() => event,
        };
        let Some(event) = event else {
            break;
        };
        match event {
            ServerH3Event::Core(H3Event::ConnectionShutdown(_)) => break,
            ServerH3Event::Core(H3Event::ConnectionError(err)) => {
                connection_result = Err(io::Error::other(err.to_string()));
                break;
            }
            ServerH3Event::Headers {
                incoming_headers, ..
            } => {
                let app = app.clone();
                requests.spawn(async move {
                    handle_h3_headers(app, incoming_headers).await;
                });
            }
            _ => {}
        }
    }
    // Request tasks retain their stream senders through response completion.
    let drain_result = drain_owned_tasks(&mut requests).await;
    // The driver uses its last seen stream ID for GOAWAY, so send it only
    // after accepted requests finish to avoid rejecting an active stream.
    controller.send_goaway();
    connection_result.and(drain_result)
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
fn parse_h3_request_head(headers: &[Header]) -> Result<ParsedRequestHead, String> {
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
            let bytes: &[u8] = chunk.as_ref();
            if body.len().saturating_add(bytes.len()) > MAX_H3_BODY_BYTES {
                return Err(format!("request body exceeds {MAX_H3_BODY_BYTES} bytes"));
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
    async fn experimental_shutdown_owns_both_listener_drains() {
        let (h1_started_tx, h1_started_rx) = tokio::sync::oneshot::channel();
        let (h1_release_tx, h1_release_rx) = tokio::sync::oneshot::channel();
        let h1_state = Arc::new(Mutex::new(Some((h1_started_tx, h1_release_rx))));
        let app = Router::new().route(
            "/slow",
            get(move || {
                let (started, release) = h1_state.lock().unwrap().take().unwrap();
                async move {
                    started.send(()).unwrap();
                    release.await.unwrap();
                    "done"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop_tx, mut h1_stop) = tokio::sync::watch::channel(false);
        let mut h3_stop = h1_stop.clone();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let (h3_started_tx, h3_started_rx) = tokio::sync::oneshot::channel();
        let (h3_release_tx, h3_release_rx) = tokio::sync::oneshot::channel();
        let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
        let h3 = async move {
            let mut requests = tokio::task::JoinSet::new();
            requests.spawn(async move {
                h3_started_tx.send(()).unwrap();
                h3_release_rx.await.unwrap();
            });
            h3_stop.changed().await.unwrap();
            stopped_tx.send(()).unwrap();
            super::drain_owned_tasks(&mut requests).await
        };
        let mut server = tokio::spawn(super::serve_owned_listeners(
            serve_h1h2_with_shutdown(listener, app, async move {
                h1_stop.changed().await.unwrap();
            }),
            h3,
            async move {
                shutdown_rx.await.unwrap();
            },
            stop_tx,
        ));
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /slow HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), h1_started_rx)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), h3_started_rx)
            .await
            .unwrap()
            .unwrap();
        shutdown_tx.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut server)
                .await
                .is_err(),
            "server returned before pending requests drained"
        );
        tokio::time::timeout(Duration::from_secs(2), stopped_rx)
            .await
            .expect("H3 listener did not observe shutdown")
            .expect("H3 listener was dropped before observing shutdown");
        h1_release_tx.send(()).unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&response).contains("done"));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut server)
                .await
                .is_err(),
            "server returned before the H3 request drained"
        );
        h3_release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn listener_failure_stops_and_drains_other_listener() {
        let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let h3 = async move {
            started_tx.send(()).unwrap();
            stop_rx.changed().await.unwrap();
            stopped_tx.send(()).unwrap();
            release_rx.await.unwrap();
            Ok(())
        };
        let h1 = async move {
            started_rx.await.unwrap();
            Err(std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                "listener failed",
            ))
        };
        let mut server = tokio::spawn(super::serve_owned_listeners(
            h1,
            h3,
            std::future::pending(),
            stop_tx,
        ));
        tokio::time::timeout(Duration::from_secs(2), stopped_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(50), &mut server)
            .await
            .is_err());
        release_tx.send(()).unwrap();
        let err = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
    }

    #[tokio::test]
    async fn injected_shutdown_stops_h1h2_server_promptly() {
        let app = Router::new().route("/v1/health", get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test requires loopback bind");
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let serve_task = tokio::spawn(serve_h1h2_with_shutdown(listener, app, async move {
            let _ = shutdown_rx.await;
        }));

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
        assert!(
            response.contains("200 OK"),
            "unexpected response: {response}"
        );
        assert!(
            response.contains("ok"),
            "unexpected response body: {response}"
        );

        shutdown_tx
            .send(())
            .expect("server task should still be live");
        let serve_result = tokio::time::timeout(Duration::from_secs(2), serve_task)
            .await
            .expect("server did not stop after injected shutdown")
            .expect("server task panicked");

        assert!(
            serve_result.is_ok(),
            "server returned error: {serve_result:?}"
        );
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
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let serve_task = tokio::spawn(serve_h1h2_with_shutdown(listener, app, async move {
            let _ = shutdown_rx.await;
        }));

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

        shutdown_tx
            .send(())
            .expect("server task should still be live");

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
        assert!(
            response.contains("200 OK"),
            "unexpected response: {response}"
        );
        assert!(
            response.contains("slow-ok"),
            "unexpected response body: {response}"
        );

        let serve_result = tokio::time::timeout(Duration::from_secs(2), serve_task)
            .await
            .expect("server did not stop after in-flight request drained")
            .expect("server task panicked");

        assert!(
            serve_result.is_ok(),
            "server returned error: {serve_result:?}"
        );
    }
}
