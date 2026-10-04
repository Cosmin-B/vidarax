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

#[cfg(test)]
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
use futures_util::{FutureExt, SinkExt, StreamExt};
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
struct OwnedH3Driver {
    inner: ServerH3Driver,
    commands: tokio_quiche::http3::driver::RequestSender<
        tokio_quiche::http3::driver::ServerH3Command,
        tokio_quiche::http3::driver::H3Command,
    >,
    cancellations: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    finished: Option<tokio::sync::oneshot::Sender<io::Result<()>>>,
    close_result: Option<io::Result<()>>,
}

#[cfg(feature = "h3-experimental")]
impl OwnedH3Driver {
    fn cancel_stopped_streams(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        use tokio_quiche::http3::driver::{H3Command, StreamShutdown};
        use tokio_quiche::quic::QuicCommand;
        let writable: Vec<_> = conn.writable().collect();
        let stopped: Vec<_> = writable
            .into_iter()
            .filter_map(|id| {
                if let Err(tokio_quiche::quiche::Error::StreamStopped(error_code)) =
                    conn.stream_capacity(id)
                {
                    Some((id, error_code))
                } else {
                    None
                }
            })
            .collect();
        if stopped.is_empty() {
            return Ok(());
        }
        // Pinned tokio-quiche 0.16 drops its body receiver before a final body
        // frame has finished writing. STOP_SENDING during a partial final write
        // then reaches a debug assertion. Its public ShutdownStream command
        // removes that stream before the regular write path can revisit it.
        for (stream_id, error_code) in stopped {
            self.cancellations.lock().unwrap().insert(stream_id);
            self.commands
                .send(H3Command::ShutdownStream {
                    stream_id,
                    shutdown: StreamShutdown::Write { error_code },
                })
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "H3 stream cancellation command was dropped",
                    )
                })?;
        }
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completion = completed.clone();
        self.commands
            .send(H3Command::QuicCmd(QuicCommand::Custom(Box::new(
                move |_| {
                    completion.store(true, std::sync::atomic::Ordering::Release);
                },
            ))))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "H3 cancellation barrier was dropped",
                )
            })?;
        // wait_for_data processes at least one queued controller command on
        // every ready poll. The FIFO marker bounds this synchronous drain to
        // the commands already ahead of it, even if other producers continue.
        while !completed.load(std::sync::atomic::Ordering::Acquire) {
            tokio_quiche::ApplicationOverQuic::wait_for_data(&mut self.inner, conn)
                .now_or_never()
                .ok_or_else(|| {
                    io::Error::other("H3 cancellation commands did not become ready")
                })??;
        }
        Ok(())
    }
}

#[cfg(feature = "h3-experimental")]
impl Drop for OwnedH3Driver {
    fn drop(&mut self) {
        if let Some(finished) = self.finished.take() {
            let _ = finished.send(self.close_result.take().unwrap_or_else(|| {
                Err(io::Error::other(
                    "H3 worker stopped before connection close",
                ))
            }));
        }
    }
}

#[cfg(feature = "h3-experimental")]
impl tokio_quiche::ApplicationOverQuic for OwnedH3Driver {
    fn on_conn_established(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
        info: &tokio_quiche::quic::HandshakeInfo,
    ) -> tokio_quiche::QuicResult<()> {
        self.inner.on_conn_established(conn, info)
    }
    fn should_act(&self) -> bool {
        self.inner.should_act()
    }
    fn buffer(&mut self) -> &mut [u8] {
        self.inner.buffer()
    }
    async fn wait_for_data(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        self.inner.wait_for_data(conn).await
    }
    fn process_reads(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        self.inner.process_reads(conn)
    }
    fn process_writes(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        self.cancel_stopped_streams(conn)?;
        self.inner.process_writes(conn)
    }
    fn on_conn_close<M: tokio_quiche::metrics::Metrics>(
        &mut self,
        conn: &mut tokio_quiche::quic::QuicheConnection,
        metrics: &M,
        result: &tokio_quiche::QuicResult<()>,
    ) {
        self.close_result = Some(
            result
                .as_ref()
                .copied()
                .map_err(|err| io::Error::other(err.to_string())),
        );
        self.inner.on_conn_close(conn, metrics, result);
    }
}

#[cfg(feature = "h3-experimental")]
async fn wait_h3_stop(mut stop: tokio::sync::watch::Receiver<bool>) {
    while !*stop.borrow_and_update() {
        if stop.changed().await.is_err() {
            break;
        }
    }
}

#[cfg(feature = "h3-experimental")]
struct H3StreamProgress {
    pending: Vec<u64>,
    acknowledged: Vec<u64>,
    cancelled: Vec<u64>,
}

#[cfg(feature = "h3-experimental")]
async fn pending_h3_streams(
    controller: &ServerH3Controller,
    streams: Vec<u64>,
    cancellations: &std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
) -> io::Result<H3StreamProgress> {
    use tokio_quiche::quic::QuicCommand;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let cancellations = cancellations.clone();
    controller
        .cmd_sender()
        .send(QuicCommand::Custom(Box::new(move |conn| {
            let result = if conn.is_closed() || conn.is_draining() {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "H3 transport disconnected before response completion",
                ))
            } else {
                // These IDs came from accepted requests, so each QUIC stream
                // existed. quiche collects a normal bidirectional stream only
                // after its request FIN is consumed and response data is ACKed.
                // STOP_SENDING is an explicit peer cancellation instead.
                let mut progress = H3StreamProgress {
                    pending: Vec::new(),
                    acknowledged: Vec::new(),
                    cancelled: Vec::new(),
                };
                let mut cancelled = cancellations.lock().unwrap();
                for id in streams {
                    if cancelled.remove(&id) {
                        progress.cancelled.push(id);
                        continue;
                    }
                    match conn.stream_capacity(id) {
                        Err(tokio_quiche::quiche::Error::InvalidStreamState(_)) => {
                            progress.acknowledged.push(id)
                        }
                        Err(tokio_quiche::quiche::Error::StreamStopped(_)) => {
                            progress.cancelled.push(id)
                        }
                        _ => progress.pending.push(id),
                    }
                }
                Ok(progress)
            };
            let _ = tx.send(result);
        })))
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "H3 driver stopped before response completion",
            )
        })?;
    tokio::time::timeout(std::time::Duration::from_secs(5), rx)
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "H3 completion probe did not finish",
            )
        })?
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "H3 completion probe was dropped"))?
}

#[cfg(feature = "h3-experimental")]
async fn finish_h3_transport(
    controller: &ServerH3Controller,
    mut streams: Vec<u64>,
    cancellations: &std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    finished: tokio::sync::oneshot::Receiver<io::Result<()>>,
) -> io::Result<()> {
    use tokio_quiche::quic::{ConnectionShutdownBehaviour, QuicCommand};
    let completion_result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !streams.is_empty() {
            let progress = pending_h3_streams(controller, streams, cancellations).await?;
            // Explicit peer cancellation completes cleanup without asserting
            // that the cancelled response was delivered or acknowledged.
            for id in progress.cancelled {
                tracing::debug!(stream_id = id, "H3 peer cancelled response");
            }
            streams = progress.pending;
            if !streams.is_empty() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
        Ok(())
    })
    .await
    .unwrap_or_else(|_| {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "H3 responses were not acknowledged before shutdown deadline",
        ))
    });
    let _ =
        controller
            .cmd_sender()
            .send(QuicCommand::ConnectionClose(ConnectionShutdownBehaviour {
                send_application_close: true,
                error_code: 0,
                reason: b"server shutdown".to_vec(),
            }));
    // In pinned tokio-quiche 0.16, resume owns the application context until
    // IoWorker::close's final socket flush finishes. Drop acknowledges that
    // lifetime; on_conn_close alone precedes the flush and is insufficient.
    let close_result = match tokio::time::timeout(std::time::Duration::from_secs(5), finished).await
    {
        Ok(Ok(result)) => result,
        Ok(_) => Err(io::Error::other(
            "H3 worker stopped without completing connection close",
        )),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "H3 connection close did not finish",
        )),
    };
    completion_result.and(close_result)
}

#[cfg(feature = "h3-experimental")]
struct ExperimentalTcpStream {
    inner: tokio::net::TcpStream,
    close: std::pin::Pin<Box<dyn Future<Output = ()> + Send>>,
    closed: bool,
}

#[cfg(feature = "h3-experimental")]
impl ExperimentalTcpStream {
    fn poll_close(&mut self, cx: &mut std::task::Context<'_>) -> io::Result<()> {
        if !self.closed && self.close.as_mut().poll(cx).is_ready() {
            self.closed = true;
        }
        if self.closed {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "experimental H1/H2 drain expired",
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "h3-experimental")]
impl tokio::io::AsyncRead for ExperimentalTcpStream {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        if let Err(err) = self.poll_close(cx) {
            return std::task::Poll::Ready(Err(err));
        }
        tokio::io::AsyncRead::poll_read(std::pin::Pin::new(&mut self.inner), cx, buf)
    }
}

#[cfg(feature = "h3-experimental")]
impl tokio::io::AsyncWrite for ExperimentalTcpStream {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        if let Err(err) = self.poll_close(cx) {
            return std::task::Poll::Ready(Err(err));
        }
        tokio::io::AsyncWrite::poll_write(std::pin::Pin::new(&mut self.inner), cx, buf)
    }
    fn is_write_vectored(&self) -> bool {
        tokio::io::AsyncWrite::is_write_vectored(&self.inner)
    }
    fn poll_write_vectored(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> std::task::Poll<io::Result<usize>> {
        if let Err(err) = self.poll_close(cx) {
            return std::task::Poll::Ready(Err(err));
        }
        tokio::io::AsyncWrite::poll_write_vectored(std::pin::Pin::new(&mut self.inner), cx, bufs)
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        if let Err(err) = self.poll_close(cx) {
            return std::task::Poll::Ready(Err(err));
        }
        tokio::io::AsyncWrite::poll_flush(std::pin::Pin::new(&mut self.inner), cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        tokio::io::AsyncWrite::poll_shutdown(std::pin::Pin::new(&mut self.inner), cx)
    }
}

#[cfg(feature = "h3-experimental")]
struct ExperimentalTcpListener {
    inner: tokio::net::TcpListener,
    close: tokio::sync::watch::Receiver<bool>,
}

#[cfg(feature = "h3-experimental")]
impl axum::serve::Listener for ExperimentalTcpListener {
    type Io = ExperimentalTcpStream;
    type Addr = std::net::SocketAddr;
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let (inner, addr) = axum::serve::Listener::accept(&mut self.inner).await;
        (
            ExperimentalTcpStream {
                inner,
                close: Box::pin(wait_h3_stop(self.close.clone())),
                closed: false,
            },
            addr,
        )
    }
    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

#[cfg(feature = "h3-experimental")]
async fn serve_h1h2_experimental(
    listener: tokio::net::TcpListener,
    app: Router,
    stop: tokio::sync::watch::Receiver<bool>,
) -> io::Result<()> {
    let addr = listener.local_addr()?;
    tracing::info!(%addr, "vidarax-api h1/h2 listening");
    let (close_tx, close_rx) = tokio::sync::watch::channel(false);
    let listener = ExperimentalTcpListener {
        inner: listener,
        close: close_rx,
    };
    let graceful_stop = stop.clone();
    let serving = async {
        axum::serve(listener, app)
            .with_graceful_shutdown(wait_h3_stop(graceful_stop))
            .await
    };
    tokio::pin!(serving);
    tokio::select! {
        result = &mut serving => result,
        _ = wait_h3_stop(stop) => {
            match tokio::time::timeout(std::time::Duration::from_secs(5), &mut serving).await {
                Ok(result) => result,
                Err(_) => {
                    // Axum privately spawns connection tasks. Closing their IO
                    // wakes them and releases response bodies before returning.
                    let _ = close_tx.send(true);
                    if tokio::time::timeout(std::time::Duration::from_secs(5), &mut serving).await.is_err() {
                        return Err(io::Error::new(io::ErrorKind::TimedOut,
                            "H1/H2 connection cleanup did not finish"));
                    }
                    Err(io::Error::new(io::ErrorKind::TimedOut,
                        "H1/H2 requests did not finish before experimental shutdown deadline"))
                }
            }
        }
    }
}

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
    let (stop_tx, h1_stop) = tokio::sync::watch::channel(false);
    let mut h3_stop = h1_stop.clone();
    let h3_stop_tx = stop_tx.clone();
    let h1 = serve_h1h2_experimental(h1_listener, app.clone(), h1_stop);
    let h3 = async move {
        let mut connections = tokio::task::JoinSet::new();
        let mut first_error = None;
        loop {
            let conn_res = tokio::select! {
                biased;
                _ = h3_stop.changed() => break,
                result = connections.join_next(), if !connections.is_empty() => {
                    if let Some(result) = result {
                        let result = result.unwrap_or_else(|err| Err(io::Error::other(err)));
                        if let Err(err) = result {
                            first_error.get_or_insert(err);
                        }
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
            let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
            let cancellations =
                std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
            let transport = conn.start(OwnedH3Driver {
                inner: driver,
                commands: controller.h3_cmd_sender(),
                cancellations: cancellations.clone(),
                finished: Some(finished_tx),
                close_result: None,
            });
            let app = app.clone();
            let stop = h3_stop.clone();
            connections.spawn(async move {
                let result =
                    serve_h3_connection(app, controller, stop, cancellations, finished_rx).await;
                drop(transport);
                result
            });
        }
        let _ = h3_stop_tx.send(true);
        // Keep the UDP listener alive while existing streams finish sending.
        while let Some(result) = connections.join_next().await {
            if let Err(err) = result.unwrap_or_else(|err| Err(io::Error::other(err))) {
                first_error.get_or_insert(err);
            }
        }
        drop(listeners);
        first_error.map_or(Ok(()), Err)
    };
    serve_owned_listeners(h1, h3, shutdown_signal(), stop_tx).await
}

#[cfg(feature = "h3-experimental")]
async fn serve_h3_connection(
    app: Router,
    mut controller: ServerH3Controller,
    stop: tokio::sync::watch::Receiver<bool>,
    cancellations: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    finished: tokio::sync::oneshot::Receiver<io::Result<()>>,
) -> io::Result<()> {
    let mut events: ServerEventStream = controller.take_event_receiver();
    let mut requests = tokio::task::JoinSet::new();
    let mut request_ids = std::collections::HashMap::new();
    let mut streams = std::collections::HashMap::<u64, tokio::sync::watch::Sender<bool>>::new();
    let mut pending = std::collections::HashSet::new();
    let mut retire = tokio::time::interval(std::time::Duration::from_millis(25));
    let shutdown = wait_h3_stop(stop.clone());
    tokio::pin!(shutdown);
    let mut stopping = false;
    let mut graceful_shutdown = false;
    let mut disconnected = false;
    let mut drain_deadline = None;
    let mut connection_result = Ok(());
    loop {
        if stopping && requests.is_empty() {
            break;
        }
        let event = tokio::select! {
            biased;
            _ = &mut shutdown, if !stopping => {
                stopping = true;
                graceful_shutdown = true;
                drain_deadline = Some(tokio::time::Instant::now() + std::time::Duration::from_secs(5));
                continue;
            }
            _ = async { tokio::time::sleep_until(drain_deadline.expect("guarded deadline")).await }, if drain_deadline.is_some() => {
                connection_result = Err(io::Error::new(io::ErrorKind::TimedOut,
                    "H3 accepted requests did not finish before shutdown deadline"));
                requests.abort_all();
                while requests.join_next().await.is_some() {}
                streams.clear();
                request_ids.clear();
                break;
            }
            result = requests.join_next_with_id(), if !requests.is_empty() => {
                match result {
                    Some(Ok((task_id, result))) => {
                        let id = request_ids.remove(&task_id).expect("owned H3 request ID");
                        streams.remove(&id);
                        match result {
                            Ok(true) => { pending.insert(id); }
                            Ok(false) => {
                                cancellations.lock().unwrap().remove(&id);
                                controller.shutdown_stream(id,
                                    tokio_quiche::http3::driver::StreamShutdown::Both {
                                        read_error_code: tokio_quiche::quiche::h3::WireErrorCode::RequestCancelled as u64,
                                        write_error_code: tokio_quiche::quiche::h3::WireErrorCode::RequestCancelled as u64,
                                    });
                            },
                            Err(err) => {
                                cancellations.lock().unwrap().remove(&id);
                                tracing::debug!(%err, stream_id = id, "H3 request send failed");
                                if graceful_shutdown { connection_result = Err(err); }
                            }
                        }
                    }
                    Some(Err(err)) => {
                        if let Some(id) = request_ids.remove(&err.id()) {
                            streams.remove(&id);
                            cancellations.lock().unwrap().remove(&id);
                        }
                        tracing::error!(%err, "H3 request task failed");
                        if graceful_shutdown { connection_result = Err(io::Error::other(err)); }
                    }
                    None => {}
                }
                continue;
            }
            _ = retire.tick(), if (!pending.is_empty() || !streams.is_empty()) && !disconnected => {
                let ids = pending.iter().copied().chain(streams.keys().copied()).collect();
                match pending_h3_streams(&controller, ids, &cancellations).await {
                    Ok(progress) => {
                        for id in &progress.acknowledged { pending.remove(id); }
                        for id in &progress.cancelled {
                            pending.remove(id);
                            tracing::debug!(stream_id = id, "H3 peer cancelled response");
                        }
                        let live: std::collections::HashSet<_> = progress.pending.into_iter().collect();
                        // An idle SSE source might never attempt another write,
                        // so the driver's next write cannot discover STOP_SENDING.
                        // Probe active streams too and cancel their body promptly.
                        for (id, cancel) in &streams {
                            if !live.contains(id) { let _ = cancel.send(true); }
                        }
                    },
                    Err(err) => {
                        // A probe can suspend across the shutdown signal.
                        // Classify its failure using the current watch value.
                        graceful_shutdown |= *stop.borrow();
                        connection_result = Err(err);
                        disconnected = true;
                        stopping = true;
                        for cancel in streams.values() { let _ = cancel.send(true); }
                    }
                }
                continue;
            }
            event = events.recv(), if !disconnected => event,
        };
        match event {
            Some(ServerH3Event::Core(
                H3Event::ConnectionShutdown(_) | H3Event::ConnectionError(_),
            ))
            | None => {
                graceful_shutdown |= *stop.borrow();
                disconnected = true;
                stopping = true;
                for cancel in streams.values() {
                    let _ = cancel.send(true);
                }
                // A peer may close an idle connection normally. Losing an
                // accepted response is surfaced to the listener's caller.
                if !requests.is_empty() || !pending.is_empty() {
                    connection_result = Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "H3 client disconnected with unfinished requests",
                    ));
                }
            }
            Some(ServerH3Event::Core(H3Event::ResetStream { stream_id })) => {
                cancellations.lock().unwrap().remove(&stream_id);
                pending.remove(&stream_id);
                if let Some(cancel) = streams.get(&stream_id) {
                    let _ = cancel.send(true);
                }
            }
            Some(ServerH3Event::Core(H3Event::StreamClosed { stream_id })) => {
                if let Some(cancel) = streams.get(&stream_id) {
                    let _ = cancel.send(true);
                } else if !pending.contains(&stream_id) {
                    cancellations.lock().unwrap().remove(&stream_id);
                }
            }
            Some(ServerH3Event::Headers {
                incoming_headers, ..
            }) if !stopping => {
                let id = incoming_headers.stream_id;
                let (disconnect_tx, disconnect_rx) = tokio::sync::watch::channel(false);
                streams.insert(id, disconnect_tx);
                let app = app.clone();
                let stop = stop.clone();
                let request = requests.spawn(async move {
                    tokio::select! {
                        biased;
                        _ = wait_h3_stop(disconnect_rx.clone()) => Ok(false),
                        result = handle_h3_headers(app, incoming_headers, stop, disconnect_rx) => result,
                    }
                });
                request_ids.insert(request.id(), id);
            }
            _ => {}
        }
    }
    let transport_result = finish_h3_transport(
        &controller,
        pending.into_iter().collect(),
        &cancellations,
        finished,
    )
    .await;
    let result = connection_result.and(transport_result);
    if graceful_shutdown || (!disconnected && *stop.borrow()) {
        result
    } else {
        // Client failures are confined to this connection while serving. A
        // failure during server-requested drain is returned to the caller.
        if let Err(err) = result {
            tracing::debug!(%err, "H3 peer connection ended");
        }
        Ok(())
    }
}

#[cfg(feature = "h3-experimental")]
async fn handle_h3_headers(
    app: Router,
    headers: IncomingH3Headers,
    stop: tokio::sync::watch::Receiver<bool>,
    disconnect: tokio::sync::watch::Receiver<bool>,
) -> io::Result<bool> {
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
            .await?;
            return Ok(true);
        }
    };

    let response = app
        .oneshot(request)
        .await
        .expect("axum router dispatch is infallible");

    send_h3_response(frame_sender, response, stop, disconnect).await
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
async fn send_h3_response(
    mut frame_sender: OutboundFrameSender,
    response: Response<Body>,
    stop: tokio::sync::watch::Receiver<bool>,
    disconnect: tokio::sync::watch::Receiver<bool>,
) -> io::Result<bool> {
    let streaming = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .is_some_and(|value| value.as_bytes().starts_with(b"text/event-stream"));
    tokio::select! {
        biased;
        _ = wait_h3_stop(disconnect) => Ok(false),
        _ = wait_h3_stop(stop), if streaming => Ok(false),
        result = async {
            if streaming {
                queue_h3_streaming_response(&mut frame_sender, response).await
            } else {
                queue_h3_response(&mut frame_sender, response).await
            }
        } => result.map(|()| true),
    }
}

#[cfg(feature = "h3-experimental")]
async fn queue_h3_streaming_response(
    frame_sender: &mut OutboundFrameSender,
    response: Response<Body>,
) -> io::Result<()> {
    let (parts, mut body) = response.into_parts();
    let mut headers = vec![Header::new(b":status", parts.status.as_str().as_bytes())];
    for (name, value) in &parts.headers {
        headers.push(Header::new(name.as_str().as_bytes(), value.as_bytes()));
    }
    frame_sender
        .send(OutboundFrame::Headers(headers, None))
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))?;
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(io::Error::other)?;
        if let Ok(bytes) = frame.into_data() {
            frame_sender
                .send(OutboundFrame::body(
                    BufFactory::buf_from_slice(&bytes),
                    false,
                ))
                .await
                .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))?;
        }
    }
    frame_sender
        .send(OutboundFrame::body(BufFactory::get_empty_buf(), true))
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))
}

#[cfg(feature = "h3-experimental")]
async fn queue_h3_response(
    frame_sender: &mut OutboundFrameSender,
    response: Response<Body>,
) -> io::Result<()> {
    let (parts, body) = response.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return send_h3_error_json(
                frame_sender,
                500,
                json!({
                    "error": {
                        "code": "internal_error",
                        "message": format!("failed to read response body: {err}")
                    }
                }),
            )
            .await;
        }
    };

    let mut h3_headers = vec![Header::new(b":status", parts.status.as_str().as_bytes())];
    for (name, value) in &parts.headers {
        h3_headers.push(Header::new(name.as_str().as_bytes(), value.as_bytes()));
    }
    frame_sender
        .send(OutboundFrame::Headers(h3_headers, None))
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))?;

    let body_frame = if body_bytes.is_empty() {
        OutboundFrame::body(BufFactory::get_empty_buf(), true)
    } else {
        OutboundFrame::body(BufFactory::buf_from_slice(body_bytes.as_ref()), true)
    };
    frame_sender
        .send(body_frame)
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))
}

#[cfg(feature = "h3-experimental")]
async fn send_h3_error_json(
    frame_sender: &mut OutboundFrameSender,
    status: u16,
    payload: serde_json::Value,
) -> io::Result<()> {
    let response = Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("error response must be constructible");

    let (parts, body) = response.into_parts();
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => return Err(io::Error::other(err)),
    };

    let mut h3_headers = vec![Header::new(b":status", parts.status.as_str().as_bytes())];
    for (name, value) in &parts.headers {
        h3_headers.push(Header::new(name.as_str().as_bytes(), value.as_bytes()));
    }
    frame_sender
        .send(OutboundFrame::Headers(h3_headers, None))
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))?;

    let body_frame = if body_bytes.is_empty() {
        OutboundFrame::body(BufFactory::get_empty_buf(), true)
    } else {
        OutboundFrame::body(BufFactory::buf_from_slice(body_bytes.as_ref()), true)
    };
    frame_sender
        .send(body_frame)
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::BrokenPipe, err))
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

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn experimental_h1_sse_shutdown_has_a_bounded_drain() {
        use axum::response::{sse::Event, Sse};
        let (events_tx, events_rx) =
            tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1);
        events_tx
            .send(Ok(Event::default().data("ready")))
            .await
            .unwrap();
        let source = Arc::new(Mutex::new(Some(events_rx)));
        let app = Router::new().route(
            "/events/stream",
            get(move || {
                let source = source.lock().unwrap().take().unwrap();
                async move { Sse::new(tokio_stream::wrappers::ReceiverStream::new(source)) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let h1 = super::serve_h1h2_experimental(listener, app, stop_rx.clone());
        let h3 = async move {
            super::wait_h3_stop(stop_rx).await;
            Ok(())
        };
        let task = tokio::spawn(super::serve_owned_listeners(
            h1,
            h3,
            async {
                let _ = shutdown_rx.await;
            },
            stop_tx,
        ));
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /events/stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut bytes = [0; 4096];
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !response.windows(5).any(|window| window == b"ready") {
                let count = stream.read(&mut bytes).await.unwrap();
                assert_ne!(count, 0);
                response.extend_from_slice(&bytes[..count]);
            }
        })
        .await
        .expect("H1 SSE did not send its first event");
        shutdown_tx.send(()).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(7), task)
            .await
            .expect("experimental H1 SSE outlived its drain deadline")
            .unwrap();
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::TimedOut);
        tokio::time::timeout(Duration::from_secs(1), events_tx.closed())
            .await
            .expect("timed-out H1 SSE retained its body");
    }

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn experimental_h1_finite_request_finishes_during_graceful_drain() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let controls = Arc::new(Mutex::new(Some((started_tx, release_rx))));
        let app = Router::new().route(
            "/finite",
            get(move || {
                let (started, release) = controls.lock().unwrap().take().unwrap();
                async move {
                    let _ = started.send(());
                    let _ = release.await;
                    "finite-complete"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let mut server = tokio::spawn(super::serve_h1h2_experimental(listener, app, stop_rx));
        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        client
            .write_all(b"GET /finite HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        stop_tx.send(true).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(50), &mut server)
            .await
            .is_err());
        release_tx.send(()).unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        assert!(response
            .windows(15)
            .any(|window| window == b"finite-complete"));
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn experimental_h1_stalled_handler_and_upload_release_on_deadline() {
        use axum::{body::Body, http::Request, routing::post};
        use http_body_util::BodyExt;
        struct OnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
            }
        }
        for upload in [false, true] {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel();
            let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
            let controls = Arc::new(Mutex::new(Some((started_tx, dropped_tx))));
            let app = Router::new().route(
                "/stall",
                post(move |request: Request<Body>| {
                    let (started, dropped) = controls.lock().unwrap().take().unwrap();
                    async move {
                        let _guard = OnDrop(Some(dropped));
                        let _ = started.send(());
                        if upload {
                            let _ = request.into_body().collect().await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                        "finished"
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
            let server = tokio::spawn(super::serve_h1h2_experimental(listener, app, stop_rx));
            let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
            client
                .write_all(
                    b"POST /stall HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\nx",
                )
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(2), started_rx)
                .await
                .unwrap()
                .unwrap();
            stop_tx.send(true).unwrap();
            let result = tokio::time::timeout(Duration::from_secs(7), server)
                .await
                .expect("experimental H1 stalled request exceeded bounded cleanup")
                .unwrap();
            assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::TimedOut);
            tokio::time::timeout(Duration::from_secs(1), dropped_rx)
                .await
                .expect("H1 IO cancellation retained its request future")
                .unwrap();
        }
    }

    #[cfg(feature = "h3-experimental")]
    mod h3_loopback {
        use super::*;
        use crate::server::{self, OwnedH3Driver};
        use futures_util::StreamExt;
        use tokio_quiche::quiche::{self, h3};

        struct TestCertificate(std::path::PathBuf);

        impl TestCertificate {
            fn new() -> Self {
                static NEXT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let path = std::env::temp_dir().join(format!(
                    "vidarax-h3-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&path).unwrap();
                // A disposable self-signed loopback certificate; no production
                // TLS assets or credentials are used by this fixture.
                let output = std::process::Command::new("openssl")
                    .args([
                        "req",
                        "-x509",
                        "-newkey",
                        "rsa:2048",
                        "-nodes",
                        "-days",
                        "1",
                        "-subj",
                        "/CN=localhost",
                        "-keyout",
                    ])
                    .arg(path.join("key.pem"))
                    .arg("-out")
                    .arg(path.join("cert.pem"))
                    .output()
                    .expect("openssl is required for H3 loopback tests");
                assert!(
                    output.status.success(),
                    "failed to generate loopback TLS certificate"
                );
                Self(path)
            }
        }

        impl Drop for TestCertificate {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        struct Client {
            socket: tokio::net::UdpSocket,
            conn: quiche::Connection,
            h3: h3::Connection,
            body: Vec<u8>,
            fin: bool,
        }

        impl Client {
            async fn flush(&mut self) {
                let mut packet = [0; 1350];
                while let Ok((len, info)) = self.conn.send(&mut packet) {
                    self.socket.send_to(&packet[..len], info.to).await.unwrap();
                }
            }

            async fn step(&mut self) {
                self.flush().await;
                let mut packet = [0; 65535];
                let timer = self.conn.timeout().unwrap_or(Duration::from_millis(20));
                let wait = timer.min(Duration::from_millis(20));
                match tokio::time::timeout(wait, self.socket.recv_from(&mut packet)).await {
                    Ok(Ok((len, from))) => {
                        self.conn
                            .recv(
                                &mut packet[..len],
                                quiche::RecvInfo {
                                    from,
                                    to: self.socket.local_addr().unwrap(),
                                },
                            )
                            .unwrap();
                    }
                    Err(_) if timer <= wait => self.conn.on_timeout(),
                    Err(_) => {}
                    Ok(Err(err)) => panic!("loopback receive failed: {err}"),
                }
                while let Ok((id, event)) = self.h3.poll(&mut self.conn) {
                    match event {
                        h3::Event::Data => {
                            let mut bytes = [0; 4096];
                            while let Ok(len) = self.h3.recv_body(&mut self.conn, id, &mut bytes) {
                                self.body.extend_from_slice(&bytes[..len]);
                            }
                        }
                        h3::Event::Finished => self.fin = true,
                        _ => {}
                    }
                }
            }

            async fn request(&mut self, path: &str) -> u64 {
                let id = self
                    .h3
                    .send_request(
                        &mut self.conn,
                        &[
                            h3::Header::new(b":method", b"GET"),
                            h3::Header::new(b":scheme", b"https"),
                            h3::Header::new(b":authority", b"localhost"),
                            h3::Header::new(b":path", path.as_bytes()),
                        ],
                        true,
                    )
                    .unwrap();
                self.flush().await;
                id
            }
        }

        async fn fixture(
            app: Router,
        ) -> (
            Client,
            tokio::sync::watch::Sender<bool>,
            tokio::task::JoinHandle<std::io::Result<()>>,
        ) {
            let certificate = TestCertificate::new();
            let cert = certificate.0.join("cert.pem");
            let key = certificate.0.join("key.pem");
            let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let peer = socket.local_addr().unwrap();
            let mut settings = server::QuicSettings::default();
            settings.disable_client_ip_validation = true;
            let mut listeners = server::listen(
                [socket],
                server::ConnectionParams::new_server(
                    settings,
                    server::TlsCertificatePaths {
                        cert: cert.to_str().unwrap(),
                        private_key: key.to_str().unwrap(),
                        kind: server::CertificateKind::X509,
                    },
                    server::Hooks::default(),
                ),
                server::DefaultMetrics,
            )
            .unwrap();
            let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
            let task = tokio::spawn(async move {
                let conn = listeners[0]
                    .next()
                    .await
                    .expect("listener stopped")
                    .unwrap();
                let (driver, controller) =
                    server::ServerH3Driver::new(server::Http3Settings::default());
                let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
                let cancellations = Arc::new(Mutex::new(std::collections::HashSet::new()));
                let transport = conn.start(OwnedH3Driver {
                    inner: driver,
                    commands: controller.h3_cmd_sender(),
                    cancellations: cancellations.clone(),
                    finished: Some(finished_tx),
                    close_result: None,
                });
                let result = server::serve_h3_connection(
                    app,
                    controller,
                    stop_rx,
                    cancellations,
                    finished_rx,
                )
                .await;
                drop(transport);
                drop(listeners);
                drop(certificate);
                result
            });
            let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let mut config = quiche::Config::new(quiche::PROTOCOL_VERSION).unwrap();
            config.verify_peer(false);
            config
                .set_application_protos(h3::APPLICATION_PROTOCOL)
                .unwrap();
            config.set_max_idle_timeout(15_000);
            config.set_max_recv_udp_payload_size(1350);
            config.set_max_send_udp_payload_size(1350);
            config.set_initial_max_data(16 * 1024 * 1024);
            config.set_initial_max_stream_data_bidi_local(4 * 1024 * 1024);
            config.set_initial_max_stream_data_bidi_remote(4 * 1024 * 1024);
            config.set_initial_max_stream_data_uni(1024 * 1024);
            config.set_initial_max_streams_bidi(100);
            config.set_initial_max_streams_uni(100);
            let cid = quiche::ConnectionId::from_ref(b"vidarax-loopback");
            let mut conn = quiche::connect(
                Some("localhost"),
                &cid,
                socket.local_addr().unwrap(),
                peer,
                &mut config,
            )
            .unwrap();
            tokio::time::timeout(Duration::from_secs(3), async {
                let mut packet = [0; 65535];
                while !conn.is_established() {
                    while let Ok((len, info)) = conn.send(&mut packet) {
                        socket.send_to(&packet[..len], info.to).await.unwrap();
                    }
                    let (len, from) = socket.recv_from(&mut packet).await.unwrap();
                    conn.recv(
                        &mut packet[..len],
                        quiche::RecvInfo {
                            from,
                            to: socket.local_addr().unwrap(),
                        },
                    )
                    .unwrap();
                }
            })
            .await
            .expect("loopback QUIC handshake timed out");
            let h3 =
                h3::Connection::with_transport(&mut conn, &h3::Config::new().unwrap()).unwrap();
            (
                Client {
                    socket,
                    conn,
                    h3,
                    body: Vec::new(),
                    fin: false,
                },
                stop_tx,
                task,
            )
        }

        #[tokio::test]
        async fn finite_response_waits_for_transport_progress_before_shutdown() {
            const BODY_LEN: usize = 512 * 1024;
            let admitted = Arc::new(AtomicBool::new(false));
            let handler_admitted = admitted.clone();
            let app = Router::new().route(
                "/finite",
                get(move || {
                    handler_admitted.store(true, Ordering::SeqCst);
                    async { vec![b'x'; BODY_LEN] }
                }),
            );
            let (mut client, stop, mut server) = fixture(app).await;
            client.request("/finite").await;
            tokio::time::timeout(Duration::from_secs(2), async {
                while !admitted.load(Ordering::SeqCst) {
                    client.step().await;
                }
            })
            .await
            .unwrap();
            stop.send(true).unwrap();
            // Stop processing UDP packets/ACKs after admission. A channel send
            // can complete here, but the finite response cannot have arrived.
            assert!(
                tokio::time::timeout(Duration::from_millis(100), &mut server)
                    .await
                    .is_err(),
                "server returned while the response still lacked transport ACKs"
            );
            tokio::time::timeout(Duration::from_secs(5), async {
                while !server.is_finished() {
                    client.step().await;
                }
            })
            .await
            .expect("H3 shutdown did not finish after transport resumed");
            server.await.unwrap().unwrap();
            assert!(client.fin, "client did not receive the response FIN");
            assert_eq!(client.body, vec![b'x'; BODY_LEN]);
        }

        #[tokio::test]
        async fn infinite_sse_releases_body_on_stream_disconnect_and_shutdown() {
            use axum::response::{sse::Event, Sse};
            for cancellation in ["stream", "connection", "shutdown"] {
                let (events_tx, events_rx) =
                    tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1);
                events_tx
                    .send(Ok(Event::default().data("ready")))
                    .await
                    .unwrap();
                let source = Arc::new(Mutex::new(Some(events_rx)));
                let app = Router::new().route(
                    "/events/stream",
                    get(move || {
                        let source = source.lock().unwrap().take().unwrap();
                        async move { Sse::new(tokio_stream::wrappers::ReceiverStream::new(source)) }
                    }),
                );
                let (mut client, stop, server) = fixture(app).await;
                let id = client.request("/events/stream").await;
                tokio::time::timeout(Duration::from_secs(2), async {
                    while client.body.is_empty() {
                        client.step().await;
                    }
                })
                .await
                .expect("H3 SSE did not deliver its first event");
                match cancellation {
                    "stream" => {
                        client
                            .conn
                            .stream_shutdown(
                                id,
                                quiche::Shutdown::Read,
                                h3::WireErrorCode::RequestCancelled as u64,
                            )
                            .unwrap();
                        client.flush().await;
                    }
                    "connection" => {
                        client.conn.close(true, 0, b"client disconnect").unwrap();
                        client.flush().await;
                    }
                    _ => {
                        let _ = stop.send(true);
                    }
                }
                tokio::time::timeout(Duration::from_secs(2), events_tx.closed())
                    .await
                    .expect("infinite SSE body stayed live after stream disconnect/shutdown");
                let _ = stop.send(true);
                tokio::time::timeout(Duration::from_secs(5), async {
                    while !server.is_finished() {
                        client.step().await;
                    }
                })
                .await
                .expect("H3 SSE shutdown left its connection task alive");
                server.await.unwrap().unwrap();
            }
        }

        #[tokio::test]
        async fn finite_peer_cancellation_finishes_cleanup_without_delivery() {
            const BODY_LEN: usize = 512 * 1024;
            let app = Router::new().route("/finite", get(|| async { vec![b'x'; BODY_LEN] }));
            let (mut client, stop, server) = fixture(app).await;
            let id = client.request("/finite").await;
            tokio::time::timeout(Duration::from_secs(2), async {
                while client.body.is_empty() {
                    client.step().await;
                }
            })
            .await
            .unwrap();
            assert!(client.body.len() < BODY_LEN);
            client
                .conn
                .stream_shutdown(
                    id,
                    quiche::Shutdown::Read,
                    h3::WireErrorCode::RequestCancelled as u64,
                )
                .unwrap();
            client.flush().await;
            stop.send(true).unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                while !server.is_finished() {
                    client.step().await;
                }
            })
            .await
            .expect("peer-cancelled H3 response left shutdown blocked");
            server.await.unwrap().unwrap();
            assert!(
                client.body.len() < BODY_LEN,
                "cancelled response was unexpectedly delivered in full"
            );
        }

        #[tokio::test]
        async fn stalled_request_body_has_a_bounded_shutdown_drain() {
            let app = Router::new().route("/upload", get(|| async { "complete" }));
            let (mut client, stop, mut server) = fixture(app).await;
            client
                .h3
                .send_request(
                    &mut client.conn,
                    &[
                        h3::Header::new(b":method", b"GET"),
                        h3::Header::new(b":scheme", b"https"),
                        h3::Header::new(b":authority", b"localhost"),
                        h3::Header::new(b":path", b"/upload"),
                    ],
                    false,
                )
                .unwrap();
            client.flush().await;
            // Leave the request's write half open, allowing headers to reach
            // the server without ever sending a body FIN.
            tokio::time::timeout(Duration::from_millis(100), async {
                loop {
                    client.step().await;
                }
            })
            .await
            .unwrap_err();
            stop.send(true).unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(100), &mut server)
                    .await
                    .is_err(),
                "fixture did not admit the unfinished request before shutdown"
            );
            tokio::time::timeout(Duration::from_secs(7), async {
                while !server.is_finished() {
                    client.step().await;
                }
            })
            .await
            .expect("unfinished H3 request outlived its drain deadline");
            assert_eq!(
                server.await.unwrap().unwrap_err().kind(),
                std::io::ErrorKind::TimedOut
            );
        }

        #[tokio::test]
        async fn stalled_transport_reports_shutdown_failure() {
            let admitted = Arc::new(AtomicBool::new(false));
            let handler_admitted = admitted.clone();
            let app = Router::new().route(
                "/finite",
                get(move || {
                    handler_admitted.store(true, Ordering::SeqCst);
                    async { vec![b'x'; 512 * 1024] }
                }),
            );
            let (mut client, stop, server) = fixture(app).await;
            client.request("/finite").await;
            tokio::time::timeout(Duration::from_secs(2), async {
                while !admitted.load(Ordering::SeqCst) {
                    client.step().await;
                }
            })
            .await
            .unwrap();
            stop.send(true).unwrap();
            let result = tokio::time::timeout(Duration::from_secs(11), server)
                .await
                .expect("stalled H3 transport exceeded its shutdown deadlines")
                .unwrap();
            assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::TimedOut);
        }
    }

    #[cfg(feature = "h3-experimental")]
    #[tokio::test]
    async fn h3_sse_body_stops_after_shutdown_and_disconnect() {
        use axum::{
            response::sse::Event,
            response::{IntoResponse, Sse},
        };
        for disconnect in [false, true] {
            let (events_tx, events_rx) =
                tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1);
            events_tx
                .send(Ok(Event::default().data("ready")))
                .await
                .unwrap();
            let response =
                Sse::new(tokio_stream::wrappers::ReceiverStream::new(events_rx)).into_response();
            let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel(8);
            let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
            let (disconnect_tx, disconnect_rx) = tokio::sync::watch::channel(false);
            let mut task = tokio::spawn(super::send_h3_response(
                super::OutboundFrameSender::new(frame_tx),
                response,
                stop_rx,
                disconnect_rx,
            ));
            // Allow body collection to start while its event source stays open.
            tokio::task::yield_now().await;
            if disconnect {
                disconnect_tx.send(true).unwrap_or(());
            } else {
                stop_tx.send(true).unwrap_or(());
            }
            assert!(
                tokio::time::timeout(Duration::from_secs(1), &mut task)
                    .await
                    .is_ok(),
                "SSE response task remained live after shutdown/disconnect"
            );
            drop(events_tx);
            while frame_rx.try_recv().is_ok() {}
        }
    }

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
