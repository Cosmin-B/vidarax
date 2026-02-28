#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelationIds {
    pub trace_id: String,
    pub span_id: String,
    pub run_id: String,
    pub stream_id: String,
    pub request_id: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct LogRecord {
    pub level: &'static str,
    pub message: String,
    pub labels: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub name: String,
    pub value: f64,
    pub labels: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct TraceRecord {
    pub span_name: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub attributes: Vec<(String, String)>,
}

pub trait LogSink {
    fn emit_log(&mut self, record: LogRecord) -> Result<(), String>;
}

pub trait MetricSink {
    fn emit_metric(&mut self, record: MetricRecord) -> Result<(), String>;
}

pub trait TraceSink {
    fn emit_trace(&mut self, record: TraceRecord) -> Result<(), String>;
}

pub struct TelemetryMux<L: LogSink, M: MetricSink, T: TraceSink> {
    pub logs: L,
    pub metrics: M,
    pub traces: T,
}

impl<L: LogSink, M: MetricSink, T: TraceSink> TelemetryMux<L, M, T> {
    pub fn new(logs: L, metrics: M, traces: T) -> Self {
        Self {
            logs,
            metrics,
            traces,
        }
    }

    pub fn emit_event(
        &mut self,
        correlation: &CorrelationIds,
        event_name: &str,
        message: &str,
        latency_ms: f64,
    ) -> Result<(), String> {
        let common = common_labels(correlation);

        self.logs.emit_log(LogRecord {
            level: "INFO",
            message: message.to_string(),
            labels: extend(
                common.clone(),
                [("event".to_string(), event_name.to_string())],
            ),
        })?;

        self.metrics.emit_metric(MetricRecord {
            name: "vidarax_event_latency_ms".to_string(),
            value: latency_ms,
            labels: extend(
                common.clone(),
                [("event".to_string(), event_name.to_string())],
            ),
        })?;

        self.traces.emit_trace(TraceRecord {
            span_name: event_name.to_string(),
            start_ns: 0,
            end_ns: (latency_ms.max(0.0) * 1_000_000.0) as u64,
            attributes: extend(common, [("message".to_string(), message.to_string())]),
        })?;

        Ok(())
    }
}

pub fn victoria_log_line(record: &LogRecord) -> String {
    let mut label_parts: Vec<String> = record
        .labels
        .iter()
        .map(|(k, v)| format!("{k}={}", sanitize(v)))
        .collect();
    label_parts.sort();
    format!(
        "{{{}}} level={} msg={}",
        label_parts.join(","),
        record.level,
        sanitize(&record.message)
    )
}

fn common_labels(c: &CorrelationIds) -> Vec<(String, String)> {
    vec![
        ("trace_id".to_string(), c.trace_id.clone()),
        ("span_id".to_string(), c.span_id.clone()),
        ("run_id".to_string(), c.run_id.clone()),
        ("stream_id".to_string(), c.stream_id.clone()),
        ("request_id".to_string(), c.request_id.clone()),
        ("model".to_string(), c.model.clone()),
    ]
}

fn extend<const N: usize>(
    mut labels: Vec<(String, String)>,
    extras: [(String, String); N],
) -> Vec<(String, String)> {
    labels.extend(extras);
    labels
}

fn sanitize(value: &str) -> String {
    value.replace('\\', "\\\\").replace(' ', "\\ ")
}

#[cfg(test)]
mod tests {
    use super::{
        victoria_log_line, CorrelationIds, LogRecord, LogSink, MetricRecord, MetricSink,
        TelemetryMux, TraceRecord, TraceSink,
    };

    struct BufferSink<T> {
        records: Vec<T>,
    }

    impl<T> Default for BufferSink<T> {
        fn default() -> Self {
            Self {
                records: Vec::new(),
            }
        }
    }

    impl LogSink for BufferSink<LogRecord> {
        fn emit_log(&mut self, record: LogRecord) -> Result<(), String> {
            self.records.push(record);
            Ok(())
        }
    }

    impl MetricSink for BufferSink<MetricRecord> {
        fn emit_metric(&mut self, record: MetricRecord) -> Result<(), String> {
            self.records.push(record);
            Ok(())
        }
    }

    impl TraceSink for BufferSink<TraceRecord> {
        fn emit_trace(&mut self, record: TraceRecord) -> Result<(), String> {
            self.records.push(record);
            Ok(())
        }
    }

    fn ids() -> CorrelationIds {
        CorrelationIds {
            trace_id: "t".to_string(),
            span_id: "s".to_string(),
            run_id: "r".to_string(),
            stream_id: "st".to_string(),
            request_id: "req".to_string(),
            model: "m".to_string(),
        }
    }

    #[test]
    fn telemetry_mux_fans_out() {
        let mut mux = TelemetryMux::new(
            BufferSink::<LogRecord>::default(),
            BufferSink::<MetricRecord>::default(),
            BufferSink::<TraceRecord>::default(),
        );
        mux.emit_event(&ids(), "gate.keepframe", "accepted", 3.2)
            .unwrap();
        assert_eq!(mux.logs.records.len(), 1);
        assert_eq!(mux.metrics.records.len(), 1);
        assert_eq!(mux.traces.records.len(), 1);
    }

    #[test]
    fn builds_victoria_log_line() {
        let line = victoria_log_line(&LogRecord {
            level: "INFO",
            message: "hello world".to_string(),
            labels: vec![("run_id".to_string(), "run-1".to_string())],
        });
        assert!(line.contains("run_id=run-1"));
        assert!(line.contains("msg=hello\\ world"));
    }
}
