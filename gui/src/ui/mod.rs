//! One module per tab. Each has its own `State`, `Msg`, `update` and `view`;
//! `update` gets read-only access to what the app shares (`Ctx`) and returns
//! what it wants the app to do (`Out`) -- start jobs, refresh the op list.

pub mod common;
pub mod compare;
pub mod evidence;
pub mod jobs;
pub mod ops;
pub mod resolved;
pub mod results;
pub mod settings;
pub mod sweep;

use crate::backend::{self, Invocation};
use crate::experiment::{ExperimentRequest, ExperimentSchema, ResolvedExperiment};
use crate::jobs::JobKind;
use crate::model::{Knobs, OpInfo};
use crate::settings::Settings;
use iced::Task;
use serde::de::DeserializeOwned;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Ops,
    Compare,
    Sweep,
    Results,
    Jobs,
    Settings,
}

impl Tab {
    pub const ALL: [Tab; 6] = [Tab::Ops, Tab::Compare, Tab::Sweep, Tab::Results, Tab::Jobs, Tab::Settings];

    pub fn parse(s: &str) -> Option<Tab> {
        Some(match s.to_ascii_lowercase().as_str() {
            "ops" => Tab::Ops,
            "compare" => Tab::Compare,
            "sweep" => Tab::Sweep,
            "results" => Tab::Results,
            "jobs" => Tab::Jobs,
            "settings" => Tab::Settings,
            _ => return None,
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Tab::Ops => "Models & ops",
            Tab::Compare => "Compare cores",
            Tab::Sweep => "Sweep",
            Tab::Results => "Sweep results",
            Tab::Jobs => "Jobs",
            Tab::Settings => "Settings",
        }
    }
}

pub struct Ctx<'a> {
    pub settings: &'a Settings,
    pub ops: &'a [OpInfo],
    pub knobs: Option<&'a Knobs>,
    /// experiment-schema's answer, once it has come and if it parsed.
    pub schema: Option<&'a ExperimentSchema>,
}

#[derive(Debug, Clone)]
pub struct NewJob {
    pub title: String,
    pub kind: JobKind,
    pub inv: Invocation,
}

pub struct Out<M> {
    pub task: Task<M>,
    pub jobs: Vec<NewJob>,
    pub refresh_ops: bool,
    pub goto: Option<Tab>,
    pub toast: Option<String>,
}

impl<M> Out<M> {
    pub fn none() -> Self {
        Out { task: Task::none(), jobs: Vec::new(), refresh_ops: false, goto: None, toast: None }
    }

    pub fn task(task: Task<M>) -> Self {
        Out { task, ..Out::none() }
    }

    pub fn jobs(jobs: Vec<NewJob>) -> Self {
        Out { jobs, goto: Some(Tab::Jobs), ..Out::none() }
    }

    pub fn toast(msg: impl Into<String>) -> Self {
        Out { toast: Some(msg.into()), ..Out::none() }
    }
}

/// Ask pipeline/gui_query.py something and parse the answer.
pub async fn query<T: DeserializeOwned>(settings: Settings, args: Vec<String>) -> Result<T, String> {
    let out = backend::capture(settings, Invocation::new("pipeline/gui_query.py").args(args)).await?;
    serde_json::from_str(out.trim()).map_err(|e| format!("unexpected answer from gui_query: {e}"))
}

/// Write `request` to `rel` (workspace-relative, so the container sees it
/// too) and have the backend resolve it. Builds and runs nothing.
pub async fn resolve(
    settings: Settings,
    rel: String,
    request: ExperimentRequest,
) -> Result<ResolvedExperiment, String> {
    let path = settings.workspace.join(&rel);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(&request).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
    let out =
        backend::capture(settings, Invocation::new("pipeline/gui_query.py").arg("resolve-experiment").arg(rel)).await?;
    ResolvedExperiment::parse(out.trim())
}

/// Resolve several requests at once, answers in the same order.
pub async fn resolve_all(
    settings: Settings,
    requests: Vec<(String, ExperimentRequest)>,
) -> Vec<Result<ResolvedExperiment, String>> {
    iced::futures::future::join_all(requests.into_iter().map(|(rel, req)| resolve(settings.clone(), rel, req))).await
}

/// Seconds since the epoch, as a sortable id for output directories.
pub fn stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{secs}")
}

/// A name that is safe as a directory name on every OS.
pub fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}
