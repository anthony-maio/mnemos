//! Opt-in diagnostic build only. Never records request or configuration data.

#[cfg(feature = "stage-timing")]
mod enabled {
    use std::cell::RefCell;
    use std::io::Write;
    use std::time::Instant;

    thread_local! {
        static TRACE: RefCell<(Instant, Vec<(&'static str, f64)>)> =
            RefCell::new((Instant::now(), Vec::with_capacity(16)));
    }

    pub fn mark(name: &'static str) {
        TRACE.with(|trace| {
            let mut trace = trace.borrow_mut();
            let elapsed = trace.0.elapsed().as_secs_f64() * 1000.0;
            trace.1.push((name, elapsed));
        });
    }

    pub fn emit() {
        TRACE.with(|trace| {
            let trace = trace.borrow();
            let record = serde_json::json!({
                "diagnostic": "native_stage_timing_v1",
                "unit": "milliseconds_since_main",
                "marks": trace.1,
            });
            // A closed diagnostic sink must not alter hook output or exit status.
            let _ = writeln!(std::io::stderr().lock(), "{record}");
        });
    }
}

#[cfg(feature = "stage-timing")]
pub use enabled::{emit, mark};

#[cfg(not(feature = "stage-timing"))]
#[inline(always)]
pub fn mark(_: &'static str) {}

#[cfg(not(feature = "stage-timing"))]
#[inline(always)]
pub fn emit() {}
