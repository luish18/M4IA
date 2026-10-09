//! Sweep results: a live or finished sweep's rows, report.py's sensitivity
//! and Pareto analysis, and CSV export.

use super::common::*;
use super::{Ctx, Out, evidence, resolved};
use crate::experiment::short;
use crate::model::{Report, RunManifest, SweepRow, fmt_cycles, fmt_knob_value, parse_jsonl};
use crate::widgets::charts::{Bar, BarChart, Point2, Scatter};
use iced::widget::{column, container, pick_list, progress_bar, row, scrollable, table, text};
use iced::{Element, Length, Task};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct Live {
    pub done: usize,
    pub total: usize,
    pub cell: String,
    pub phase: String,
    pub eta_s: Option<f64>,
    pub running: bool,
}

#[derive(Debug, Default)]
pub struct State {
    /// Workspace-relative sweep directories found under work/gui/sweeps.
    sweeps: Vec<String>,
    current: Option<String>,
    rows: Vec<SweepRow>,
    report: Option<Result<Report, String>>,
    analysing: bool,
    model: Option<String>,
    sort: Sort,
    pub live: Option<(String, Live)>,
    error: Option<String>,
    /// The cell whose manifest is open: its path in the sweep, and what was read.
    inspect: Option<(String, Result<RunManifest, String>)>,
    /// The node of that manifest whose mapping explanation is open.
    open_node: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sort {
    #[default]
    Run,
    Cycles,
    Area,
}

impl Sort {
    const ALL: [Sort; 3] = [Sort::Run, Sort::Cycles, Sort::Area];
}

impl std::fmt::Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Sort::Run => "run order",
            Sort::Cycles => "cycles",
            Sort::Area => "area",
        })
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    Refresh,
    Select(String),
    Open,
    Opened(Option<PathBuf>),
    Analyse,
    Analysed(String, Result<Report, String>),
    Model(String),
    Sort(Sort),
    Export,
    Exported(Result<Option<PathBuf>, String>),
    /// Open a cell's manifest (its path relative to the sweep), or close it.
    Inspect(Option<String>),
    OpenNode(Option<i64>),
}

const ALL_MODELS: &str = "all models";

impl State {
    pub fn scan(&mut self, ws: &Path) {
        let root = ws.join("work").join("gui").join("sweeps");
        let mut found: Vec<(std::time::SystemTime, String)> = std::fs::read_dir(&root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("sweep.jsonl").is_file() || e.path().join("designs.json").is_file())
            .map(|e| {
                let t = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                (t, format!("work/gui/sweeps/{}", e.file_name().to_string_lossy()))
            })
            .collect();
        found.sort_by_key(|f| std::cmp::Reverse(f.0));
        let mut sweeps: Vec<String> = found.into_iter().map(|(_, s)| s).collect();
        if let Some(c) = &self.current
            && !sweeps.contains(c)
        {
            sweeps.insert(0, c.clone());
        }
        self.sweeps = sweeps;
    }

    /// Re-read the current sweep's JSONL from disk.
    pub fn reload(&mut self, ws: &Path) {
        if let Some(c) = &self.current {
            self.rows =
                std::fs::read_to_string(ws.join(c).join("sweep.jsonl")).map(|t| parse_jsonl(&t)).unwrap_or_default();
        }
    }

    pub fn select(&mut self, ws: &Path, dir: String) {
        if self.current.as_deref() != Some(dir.as_str()) {
            self.report = None;
            self.model = None;
            self.inspect = None;
        }
        self.current = Some(dir);
        self.reload(ws);
        self.scan(ws);
    }

    #[cfg(test)]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    #[cfg(test)]
    pub fn report(&self) -> Option<&Report> {
        self.report.as_ref().and_then(|r| r.as_ref().ok())
    }

    #[cfg(test)]
    pub fn inspected(&self) -> Option<&Result<RunManifest, String>> {
        self.inspect.as_ref().map(|(_, m)| m)
    }

    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// A row the running sweep just finished.
    pub fn push_live_row(&mut self, dir: &str, row: SweepRow) {
        if self.current.as_deref() == Some(dir) {
            self.rows.push(row);
            self.report = None;
        }
    }

    /// Median wall time per cell across the sweeps on disk, for estimates.
    pub fn per_cell_estimate(&self, ws: &Path) -> Option<f64> {
        let mut walls: Vec<f64> = self
            .sweeps
            .iter()
            .filter_map(|d| std::fs::read_to_string(ws.join(d).join("sweep.jsonl")).ok())
            .flat_map(|t| parse_jsonl(&t))
            .filter(|r| r.status == "ok")
            .filter_map(|r| r.wall_s)
            .collect();
        if walls.is_empty() {
            return None;
        }
        walls.sort_by(f64::total_cmp);
        Some(walls[walls.len() / 2])
    }

    pub fn analyse(&mut self, ctx: &Ctx) -> Task<Msg> {
        let Some(dir) = self.current.clone() else { return Task::none() };
        self.analysing = true;
        let jsonl = format!("{dir}/sweep.jsonl");
        let settings = ctx.settings.clone();
        Task::perform(
            async move {
                let out = crate::backend::capture(
                    settings,
                    crate::backend::Invocation::new("pipeline/sweep/report.py").arg(jsonl).arg("--json"),
                )
                .await?;
                serde_json::from_str::<Report>(out.trim()).map_err(|e| format!("unexpected report: {e}"))
            },
            move |r| Msg::Analysed(dir.clone(), r),
        )
    }

    pub fn update(&mut self, msg: Msg, ctx: &Ctx) -> Out<Msg> {
        let ws = ctx.settings.workspace.clone();
        match msg {
            Msg::Refresh => {
                self.scan(&ws);
                self.reload(&ws);
            }
            Msg::Select(d) => {
                self.select(&ws, d);
                if !self.rows.is_empty() && !self.is_live_current() {
                    return Out::task(self.analyse(ctx));
                }
            }
            Msg::Open => {
                let dir = ws.join("work");
                return Out::task(Task::perform(
                    async move {
                        rfd::AsyncFileDialog::new()
                            .set_directory(dir)
                            .add_filter("sweep rows", &["jsonl"])
                            .pick_file()
                            .await
                            .map(|h| h.path().to_path_buf())
                    },
                    Msg::Opened,
                ));
            }
            Msg::Opened(Some(p)) => match self.import(&ws, &p) {
                Ok(dir) => {
                    self.select(&ws, dir);
                    return Out::task(self.analyse(ctx));
                }
                Err(e) => self.error = Some(e),
            },
            Msg::Opened(None) => {}
            Msg::Analyse => return Out::task(self.analyse(ctx)),
            Msg::Analysed(dir, r) => {
                if self.current.as_deref() == Some(dir.as_str()) {
                    self.analysing = false;
                    self.report = Some(r);
                }
            }
            Msg::Model(m) => self.model = (m != ALL_MODELS).then_some(m),
            Msg::Sort(s) => self.sort = s,
            Msg::Export => {
                let csv = self.csv();
                let name = self.current.as_deref().and_then(|c| c.rsplit('/').next()).unwrap_or("sweep").to_string();
                return Out::task(Task::perform(
                    async move {
                        let Some(h) = rfd::AsyncFileDialog::new()
                            .set_file_name(format!("{name}.csv"))
                            .add_filter("CSV", &["csv"])
                            .save_file()
                            .await
                        else {
                            return Ok(None);
                        };
                        std::fs::write(h.path(), csv).map_err(|e| e.to_string())?;
                        Ok(Some(h.path().to_path_buf()))
                    },
                    Msg::Exported,
                ));
            }
            Msg::Exported(Ok(Some(p))) => return Out::toast(format!("exported {}", p.display())),
            Msg::Exported(Ok(None)) => {}
            Msg::Exported(Err(e)) => self.error = Some(e),
            Msg::Inspect(None) => self.inspect = None,
            Msg::Inspect(Some(rel)) => {
                let Some(dir) = &self.current else { return Out::none() };
                let path = ws.join(dir).join(&rel);
                let m = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))
                    .and_then(|t| RunManifest::parse(&t).map_err(|e| format!("{rel}: {e}")));
                self.inspect = Some((rel, m));
                self.open_node = None;
            }
            Msg::OpenNode(n) => self.open_node = n,
        }
        Out::none()
    }

    fn is_live_current(&self) -> bool {
        matches!((&self.live, &self.current), (Some((d, l)), Some(c)) if d == c && l.running)
    }

    /// A sweep.jsonl from outside work/gui/sweeps: report.py has to reach it
    /// from inside the container too, so bring it into the workspace.
    fn import(&self, ws: &Path, p: &Path) -> Result<String, String> {
        if let Ok(rel) = p.parent().unwrap_or(p).strip_prefix(ws)
            && p.file_name().is_some_and(|f| f == "sweep.jsonl")
        {
            return Ok(rel.to_string_lossy().replace('\\', "/"));
        }
        let stem = p
            .parent()
            .and_then(|d| d.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "imported".into());
        let rel = format!("work/gui/sweeps/imported-{stem}");
        let dir = ws.join(&rel);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::copy(p, dir.join("sweep.jsonl")).map_err(|e| e.to_string())?;
        Ok(rel)
    }

    fn baseline(&self, model: &str) -> Option<&SweepRow> {
        self.rows.iter().find(|r| r.model == model && r.design_slug == "baseline")
    }

    /// "SPATZ_NB_LANES=8, L2_WAYS=4" against the model's baseline.
    fn describe(&self, r: &SweepRow) -> String {
        if r.design_slug == "baseline" {
            return "baseline".into();
        }
        match self.baseline(&r.model).or_else(|| self.rows.iter().find(|x| x.design_slug == "baseline")) {
            Some(b) => {
                let d = r.diff(&b.design);
                if d.is_empty() {
                    r.design_slug.clone()
                } else {
                    d.iter().map(|(k, v)| format!("{k}={}", fmt_knob_value(k, *v))).collect::<Vec<_>>().join(", ")
                }
            }
            None => r.design_slug.clone(),
        }
    }

    fn csv(&self) -> String {
        let knobs: Vec<String> = self.rows.first().map(|r| r.design.keys().cloned().collect()).unwrap_or_default();
        let mut out = format!(
            "design_slug,model,status,cycles,area_au,cache_dynamic_pj,wall_s,dram_kind,artifact_key,run_fingerprint,{}\n",
            knobs.join(",")
        );
        for r in &self.rows {
            let opt = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_default();
            let vals: Vec<String> =
                knobs.iter().map(|k| r.design.get(k).map(|v| v.to_string()).unwrap_or_default()).collect();
            let s = |v: &Option<String>| v.clone().unwrap_or_default();
            out += &format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                r.design_slug,
                r.model,
                r.status,
                r.cycles().map(|c| c.to_string()).unwrap_or_default(),
                opt(r.area_au),
                opt(r.cache_dynamic_pj),
                opt(r.wall_s),
                s(&r.dram_kind),
                s(&r.artifact_key),
                s(&r.run_fingerprint),
                vals.join(",")
            );
        }
        out
    }

    pub fn view<'a>(&'a self, _ctx: &Ctx<'a>) -> Element<'a, Msg> {
        let picker = row![
            text("Sweep").size(14),
            pick_list(self.sweeps.as_slice(), self.current.as_ref(), Msg::Select)
                .placeholder("choose a sweep")
                .text_size(13),
            small_button("Refresh", Some(Msg::Refresh)),
            small_button("Open sweep.jsonl…", Some(Msg::Open)),
            small_button(
                if self.analysing { "Analysing…" } else { "Analyse" },
                (self.current.is_some() && !self.analysing && !self.rows.is_empty()).then_some(Msg::Analyse)
            ),
            small_button("Export CSV…", (!self.rows.is_empty()).then_some(Msg::Export)),
        ]
        .spacing(8)
        .align_y(iced::Center);

        let mut body = column![picker].spacing(12);
        if let Some(e) = &self.error {
            body = body.push(error(e.as_str()));
        }
        if let Some((dir, l)) = &self.live
            && self.current.as_deref() == Some(dir.as_str())
        {
            let frac = if l.total > 0 { l.done as f32 / l.total as f32 } else { 0.0 };
            let state = if l.running {
                format!(
                    "{}/{} cells · {} · {}{}",
                    l.done,
                    l.total,
                    l.cell,
                    l.phase,
                    l.eta_s.map(|e| format!(" · about {} left", super::sweep::human_duration(e))).unwrap_or_default()
                )
            } else {
                format!("{}/{} cells, finished", l.done, l.total)
            };
            body = body.push(section(
                "Progress",
                column![progress_bar(0.0..=1.0, frac).girth(10), text(state).size(13)].spacing(6),
            ));
        }
        if self.current.is_none() {
            return body.push(muted("Pick a sweep, or run one from the Sweep tab.")).into();
        }
        if self.rows.is_empty() {
            return body.push(muted("No rows yet.")).into();
        }

        let mut models: Vec<String> = self.rows.iter().map(|r| r.model.clone()).collect();
        models.sort();
        models.dedup();
        let mut model_opts = vec![ALL_MODELS.to_string()];
        model_opts.extend(models.iter().cloned());
        let shown_model = self.model.clone().unwrap_or_else(|| ALL_MODELS.into());
        let filters = row![
            text("Model").size(13),
            pick_list(model_opts, Some(shown_model), Msg::Model).text_size(13),
            text("Sort by").size(13),
            pick_list(Sort::ALL, Some(self.sort), Msg::Sort).text_size(13),
        ]
        .spacing(8)
        .align_y(iced::Center);
        body = body.push(filters);

        let ok = self.rows.iter().filter(|r| r.status == "ok").count();
        body = body.push(text(format!("{} cells: {ok} ok, {} not", self.rows.len(), self.rows.len() - ok)).size(14));

        match &self.report {
            Some(Ok(rep)) => {
                body = body.push(self.sensitivity_view(rep));
                body = body.push(self.pareto_view(rep));
            }
            Some(Err(e)) => body = body.push(error(format!("report.py failed: {e}"))),
            None if self.analysing => body = body.push(muted("running report.py…")),
            None => body = body.push(muted("Press Analyse for the sensitivity table and the Pareto front.")),
        }
        body = body.push(section("Cells", self.rows_table()));
        if let Some((rel, m)) = &self.inspect {
            body = body.push(match m {
                Ok(m) => self.manifest_view(m),
                Err(e) => section("Run evidence", column![error(e.as_str()), muted(rel.as_str())].spacing(4)),
            });
        }
        scrollable(body.padding(iced::Padding { right: 12.0, ..Default::default() })).into()
    }

    fn sensitivity_view<'a>(&'a self, rep: &'a Report) -> Element<'a, Msg> {
        let mut c = column![].spacing(12);
        for s in rep.sensitivity.iter().filter(|s| self.model.as_ref().is_none_or(|m| m == &s.model)) {
            let bars = s
                .knobs
                .iter()
                .flat_map(|k| {
                    let base = k
                        .base_value
                        .as_i64()
                        .map(|b| fmt_knob_value(&k.knob, b))
                        .unwrap_or_else(|| k.base_value.to_string());
                    k.points.iter().map(move |p| Bar {
                        label: format!("{} {}→{}", k.knob, base, fmt_knob_value(&k.knob, p.value)),
                        value: p.pct,
                        shown: format!("{:+.1}%  ({})", p.pct, fmt_cycles(p.cycles)),
                        highlight: false,
                        missing: false,
                    })
                })
                .collect::<Vec<_>>();
            let mut block =
                column![text(format!("{} — baseline {} cycles", s.model, fmt_cycles(s.base_cycles))).size(14)]
                    .spacing(6);
            if bars.is_empty() {
                block = block.push(muted("No one-knob-at-a-time points to compare against the baseline."));
            } else {
                block = block.push(BarChart { bars, log: false }.view());
            }
            if !s.flat.is_empty() {
                block = block.push(muted(format!("No measurable effect: {}", s.flat.join(", "))));
            }
            c = c.push(block);
        }
        section("Sensitivity (vs baseline, negative is faster)", c)
    }

    fn pareto_view<'a>(&'a self, rep: &'a Report) -> Element<'a, Msg> {
        let Some(p) = &rep.pareto else {
            return section("Pareto front", muted("No area figures in these rows."));
        };
        let front: std::collections::HashSet<&str> = p.front.iter().map(|f| f.key.as_str()).collect();
        let points: Vec<Point2> = self
            .rows
            .iter()
            .filter(|r| r.status == "ok" && self.model.as_ref().is_none_or(|m| m == &r.model))
            .filter_map(|r| {
                Some(Point2 {
                    x: r.cycles()? as f64,
                    y: r.area_au?,
                    label: format!("{} · {}", self.describe(r), r.model),
                    emphasis: front.contains(format!("{}|{}", r.design_slug, r.model).as_str()),
                })
            })
            .collect();
        let mut c = column![
            text(format!(
                "{} of {} points are non-dominated over {}.",
                p.front.len(),
                p.points,
                p.objectives.join(", ")
            ))
            .size(13),
            Scatter { points, x_label: "cycles".into(), y_label: "area (a.u.)".into(), log_x: false }.view(360.0),
        ]
        .spacing(6);
        if let Some(r) = &rep.robustness {
            c = c.push(if r.sourced {
                muted("All area coefficients are sourced.")
            } else {
                muted(format!(
                    "Area coefficients are placeholders: {} of {} front members survive ±{:.0}%{}. Treat cycles as the only measured objective.",
                    r.stable.len(),
                    r.front,
                    r.perturb * 100.0,
                    if r.lost.is_empty() { String::new() } else { format!(" (lost: {})", r.lost.join(", ")) }
                ))
            });
        }
        section("Pareto front (cycles vs area; hover for the design)", c)
    }

    fn rows_table<'a>(&'a self) -> Element<'a, Msg> {
        let mut rows: Vec<&SweepRow> =
            self.rows.iter().filter(|r| self.model.as_ref().is_none_or(|m| m == &r.model)).collect();
        match self.sort {
            Sort::Run => {}
            Sort::Cycles => rows.sort_by_key(|r| r.cycles().unwrap_or(u64::MAX)),
            Sort::Area => rows.sort_by(|a, b| a.area_au.unwrap_or(f64::MAX).total_cmp(&b.area_au.unwrap_or(f64::MAX))),
        }
        #[derive(Clone)]
        struct R {
            design: String,
            model: String,
            status: String,
            cycles: String,
            vs: String,
            area: String,
            energy: String,
            wall: String,
            memory: String,
            calib: String,
            manifest: Option<String>,
            why: String,
        }
        let data: Vec<R> = rows
            .iter()
            .map(|r| {
                let base = self.baseline(&r.model).and_then(|b| b.cycles());
                R {
                    design: self.describe(r),
                    model: r.model.clone(),
                    status: r.status.clone(),
                    cycles: r.cycles().map(fmt_cycles).unwrap_or_else(|| "-".into()),
                    vs: match (base, r.cycles()) {
                        (Some(b), Some(c)) if r.design_slug != "baseline" => {
                            format!("{:+.1}%", 100.0 * (c as f64 - b as f64) / b as f64)
                        }
                        _ => String::new(),
                    },
                    area: r.area_au.map(|a| format!("{:.2}M", a / 1e6)).unwrap_or_else(|| "-".into()),
                    energy: r.cache_dynamic_pj.map(|p| format!("{:.2} µJ", p / 1e6)).unwrap_or_else(|| "-".into()),
                    wall: r.wall_s.map(|w| format!("{w:.0}s")).unwrap_or_default(),
                    memory: r.dram_kind.clone().unwrap_or_default(),
                    calib: r.calibration_cache.clone().unwrap_or_default().chars().take(24).collect(),
                    manifest: r.manifest.clone(),
                    why: r.reasons.join("; ").chars().take(140).collect(),
                }
            })
            .collect();
        container(
            table(
                [
                    table::column(text("design").size(12), |r: R| text(r.design).size(12)),
                    table::column(text("model").size(12), |r: R| text(r.model).size(12)),
                    table::column(text("status").size(12), |r: R| status(&r.status)),
                    table::column(text("cycles").size(12), |r: R| mono(r.cycles)).align_x(iced::Right),
                    table::column(text("vs base").size(12), |r: R| mono(r.vs)).align_x(iced::Right),
                    table::column(text("area").size(12), |r: R| mono(r.area)).align_x(iced::Right),
                    table::column(text("cache energy").size(12), |r: R| mono(r.energy)).align_x(iced::Right),
                    table::column(text("wall").size(12), |r: R| mono(r.wall)).align_x(iced::Right),
                    table::column(text("memory").size(12), |r: R| mono(r.memory)),
                    table::column(text("calibration").size(12), |r: R| mono(r.calib)),
                    table::column(text("evidence").size(12), |r: R| {
                        small_button("details", r.manifest.map(|m| Msg::Inspect(Some(m))))
                    }),
                    table::column(text("why not").size(12), |r: R| muted(r.why)).width(Length::Fill),
                ],
                data,
            )
            .padding_x(8)
            .padding_y(2),
        )
        .into()
    }

    /// One cell's manifest, layer by layer, in the order the evidence arises.
    fn manifest_view<'a>(&'a self, m: &'a RunManifest) -> Element<'a, Msg> {
        let layer = |title: &'a str, hint: &'a str| {
            row![text(title).size(14), muted(hint)].spacing(10).align_y(iced::Alignment::End)
        };
        let requested = m.requested.iter().map(|(k, v)| format!("{k} = {v}")).collect::<Vec<_>>().join("\n");
        let r = &m.measured;
        let mut measured = vec![format!("status {}", r.status)];
        if let Some((c, unit)) = r.headline() {
            measured.push(format!("{} {unit}", fmt_cycles(c)));
        }
        if !r.per_engine_cycles.is_empty() {
            let e = r.per_engine_cycles.iter().map(|(k, v)| format!("{k} {}", fmt_cycles(*v))).collect::<Vec<_>>();
            measured.push(format!("engines: {}", e.join(", ")));
        }
        if let Some(a) = r.accuracy {
            measured.push(format!("accuracy {:.1}%", a * 100.0));
        }
        if let Some(d) = r.maxdiff {
            measured.push(format!("max |err| {d:.2e}"));
        }
        if let Some(w) = r.wall_s {
            measured.push(format!("wall {w:.0}s"));
        }
        let identity = [
            ("artifact key", m.artifact_key.clone()),
            (
                "machine",
                format!("{}  {}", m.resolved.machine.machine_key, short(&m.resolved.machine.machine_fingerprint)),
            ),
            ("run input", m.run_input_fingerprint.clone()),
            ("completed run", m.run_fingerprint.clone()),
            ("calibration", format!("{}  input {}", m.calibration.path, short(&m.calibration.input_fingerprint))),
            ("calib. metadata", m.calibration.metadata_digest.clone()),
            ("result", format!("{}  {}", m.artifacts.result.path, short(&m.artifacts.result.digest))),
            ("run sources", m.run_provenance.source_set.digest.clone()),
        ]
        .into_iter()
        .map(|(k, v)| row![text(k).size(12).width(Length::Fixed(110.0)), mono(v)].spacing(8).into());
        let lines = evidence::join(&m.generated, &m.actual.nodes, &m.measured.nodes);

        let body = column![
            row![
                muted(format!("cells/{}", m.artifact_key)).width(Length::Fill),
                small_button("Close", Some(Msg::Inspect(None)))
            ]
            .align_y(iced::Center),
            layer("Requested", "what the sweep asked for"),
            mono(requested),
            layer("Resolved", "the canonical experiment and machine it became"),
            resolved::summary(&m.resolved.experiment, None),
            mono(format!(
                "effective: {}",
                m.resolved.execution.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
            )),
            layer("Generated · Actual · Measured", "per node: placement, what completed nodes prove ran, cycles"),
            evidence::table(&lines, self.open_node, Msg::OpenNode),
            layer("Measured", "observations of this run"),
            mono(measured.join(" · ")),
            layer("Identity", "fingerprints and linked artifacts"),
            lines_of(identity),
        ]
        .spacing(6);
        section("Run evidence", body)
    }
}

fn lines_of<'a>(items: impl Iterator<Item = Element<'a, Msg>>) -> Element<'a, Msg> {
    column(items).spacing(2).into()
}
