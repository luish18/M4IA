//! Compare cores: run ops on a chosen set of cores, standalone (run.py) or on
//! the heterogeneous SoC (run_hetero.py), and put the numbers side by side.

use super::common::*;
use super::{Ctx, NewJob, Out, stamp};
use crate::backend::Invocation;
use crate::jobs::JobKind;
use crate::model::{HeteroResult, IsolatedResult, RunResult, fmt_cycles};
use crate::widgets::charts::{Bar, BarChart};
use iced::widget::{button, checkbox, column, container, pick_list, radio, row, scrollable, table, text, text_input};
use iced::{Element, Length, Task};
use std::collections::BTreeSet;
use std::path::PathBuf;

pub const CORES: [&str; 4] = ["cva6", "snitch", "spatz", "ara"];
const PLACEMENTS: [&str; 4] = ["mapped", "cva6", "snitch", "spatz"];

fn with_explicit_dram(inv: Invocation, dram: DramChoice) -> Invocation {
    inv.arg("--dram").arg(dram.arg())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Isolated,
    Hetero,
}

#[derive(Debug, Clone)]
pub struct Loaded {
    pub source: String,
    pub result: RunResult,
}

#[derive(Debug, Clone)]
pub struct State {
    mode: Mode,
    filter: String,
    selected: BTreeSet<String>,
    cores: BTreeSet<&'static str>,
    memory: &'static str,
    dram: DramChoice,
    spatz_kernels: &'static str,
    timeout: String,
    host: &'static str,
    placements: BTreeSet<&'static str>,
    power: bool,
    images: String,
    frontend: &'static str,
    serial: bool,
    results: Vec<Loaded>,
    log_scale: bool,
    show_caches: bool,
    show_mapping: bool,
    error: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        State {
            mode: Mode::Isolated,
            filter: String::new(),
            selected: BTreeSet::new(),
            cores: ["cva6", "snitch", "spatz"].into(),
            memory: "real",
            dram: DramChoice::Fixed,
            spatz_kernels: "tuned",
            timeout: "600".into(),
            host: "cva6",
            placements: ["mapped"].into(),
            power: false,
            images: "16".into(),
            frontend: "snitch",
            serial: false,
            results: Vec::new(),
            log_scale: false,
            show_caches: false,
            show_mapping: false,
            error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    Mode(Mode),
    Filter(String),
    ToggleOp(String, bool),
    ToggleCore(&'static str, bool),
    Memory(&'static str),
    Dram(DramChoice),
    SpatzKernels(&'static str),
    Timeout(String),
    Host(&'static str),
    TogglePlacement(&'static str, bool),
    Power(bool),
    Images(String),
    Frontend(&'static str),
    Serial(bool),
    Run,
    Open,
    Opened(Option<PathBuf>),
    Remove(usize),
    Clear,
    LogScale(bool),
    ShowCaches(bool),
    ShowMapping(bool),
}

impl State {
    /// A finished run's result file, `rel` being workspace-relative.
    pub fn load(&mut self, ws: &std::path::Path, rel: &str) -> Result<(), String> {
        self.load_file(&ws.join(rel), rel.to_string())
    }

    #[cfg(test)]
    pub fn loaded(&self) -> &[Loaded] {
        &self.results
    }

    /// Any result file, shown under `source`.
    pub fn load_file(&mut self, path: &std::path::Path, source: String) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{source}: {e}"))?;
        let result = RunResult::parse(&text).map_err(|e| format!("{source}: {e}"))?;
        self.results.retain(|l| l.source != source);
        self.results.push(Loaded { source, result });
        Ok(())
    }

    pub fn update(&mut self, msg: Msg, ctx: &Ctx) -> Out<Msg> {
        match msg {
            Msg::Mode(m) => self.mode = m,
            Msg::Filter(s) => self.filter = s,
            Msg::ToggleOp(a, on) => {
                if on {
                    self.selected.insert(a);
                } else {
                    self.selected.remove(&a);
                }
            }
            Msg::ToggleCore(c, on) => {
                if on {
                    self.cores.insert(c);
                } else {
                    self.cores.remove(c);
                }
            }
            Msg::Memory(m) => self.memory = m,
            Msg::Dram(d) => self.dram = d,
            Msg::SpatzKernels(k) => self.spatz_kernels = k,
            Msg::Timeout(s) => self.timeout = s,
            Msg::Host(h) => self.host = h,
            Msg::TogglePlacement(p, on) => {
                if on {
                    self.placements.insert(p);
                } else {
                    self.placements.remove(p);
                }
            }
            Msg::Power(b) => self.power = b,
            Msg::Images(s) => self.images = s,
            Msg::Frontend(f) => self.frontend = f,
            Msg::Serial(b) => self.serial = b,
            Msg::Run => match self.jobs(ctx) {
                Ok(jobs) => {
                    self.error = None;
                    return Out::jobs(jobs);
                }
                Err(e) => self.error = Some(e),
            },
            Msg::Open => {
                let dir = ctx.settings.workspace.join("results");
                return Out::task(Task::perform(
                    async move {
                        rfd::AsyncFileDialog::new()
                            .set_directory(dir)
                            .add_filter("result JSON", &["json"])
                            .pick_file()
                            .await
                            .map(|h| h.path().to_path_buf())
                    },
                    Msg::Opened,
                ));
            }
            Msg::Opened(Some(p)) => {
                let source = p.strip_prefix(&ctx.settings.workspace).unwrap_or(&p).display().to_string();
                self.error = self.load_file(&p, source).err();
            }
            Msg::Opened(None) => {}
            Msg::Remove(i) => {
                if i < self.results.len() {
                    self.results.remove(i);
                }
            }
            Msg::Clear => self.results.clear(),
            Msg::LogScale(b) => self.log_scale = b,
            Msg::ShowCaches(b) => self.show_caches = b,
            Msg::ShowMapping(b) => self.show_mapping = b,
        }
        Out::none()
    }

    fn jobs(&self, ctx: &Ctx) -> Result<Vec<NewJob>, String> {
        if self.selected.is_empty() {
            return Err("select at least one op".into());
        }
        let run = format!("work/gui/runs/{}", stamp());
        let mut jobs = Vec::new();
        match self.mode {
            Mode::Isolated => {
                if self.cores.is_empty() {
                    return Err("select at least one core".into());
                }
                let timeout: u32 = self.timeout.trim().parse().map_err(|_| "timeout must be whole seconds")?;
                let cores: Vec<&str> = CORES.iter().copied().filter(|c| self.cores.contains(c)).collect();
                // The device only exists in the modelled memory system.
                let dram = if self.memory == "real" { self.dram } else { DramChoice::Fixed };
                for op in &self.selected {
                    let out = format!("{run}/{}.json", op.replace('/', "_"));
                    let mut inv = Invocation::new("pipeline/run.py")
                        .arg(op.clone())
                        .arg("--cores")
                        .arg(cores.join(","))
                        .arg("--memory")
                        .arg(self.memory)
                        .arg("--spatz-kernels")
                        .arg(self.spatz_kernels)
                        .arg("--timeout")
                        .arg(timeout.to_string())
                        .arg("--out")
                        .arg(out.clone());
                    inv = with_explicit_dram(inv, dram);
                    let mem = if dram == DramChoice::Fixed { self.memory.to_string() } else { dram.arg().to_string() };
                    jobs.push(NewJob {
                        title: format!("{op} on {} ({mem} memory)", cores.join(", ")),
                        kind: JobKind::Run { out },
                        inv,
                    });
                }
            }
            Mode::Hetero => {
                if self.placements.is_empty() {
                    return Err("select at least one placement".into());
                }
                let images: u32 = self.images.trim().parse().map_err(|_| "samples must be a whole number")?;
                for op in &self.selected {
                    let kws = ctx.ops.iter().any(|o| &o.arg == op && o.app.as_deref() == Some("kws"));
                    for p in PLACEMENTS.iter().copied().filter(|p| self.placements.contains(p)) {
                        let mut tag = format!("{}-{}-{p}", op.replace('/', "_"), self.host);
                        let mut inv =
                            Invocation::new("pipeline/run_hetero.py").arg(op.clone()).arg("--host").arg(self.host);
                        if p != "mapped" {
                            inv = inv.arg("--pin").arg(p);
                        }
                        inv = inv.arg("--images").arg(images.to_string()).flag(self.power, "--power").arg("-q");
                        inv = with_explicit_dram(inv, self.dram);
                        if self.dram != DramChoice::Fixed {
                            tag += &format!("-{}", self.dram.arg());
                        }
                        if kws {
                            inv = inv.arg("--frontend").arg(self.frontend).flag(self.serial, "--serial");
                            tag += &format!("-fe{}{}", self.frontend, if self.serial { "-serial" } else { "" });
                        }
                        let out = format!("{run}/{tag}.json");
                        inv = inv.arg("--out").arg(out.clone());
                        let place = if p == "mapped" { "mapped".to_string() } else { format!("pinned to {p}") };
                        let mem =
                            if self.dram == DramChoice::Fixed { String::new() } else { format!(", {}", self.dram) };
                        jobs.push(NewJob {
                            title: format!("{op} on the SoC, {} host, {place}{mem}", self.host),
                            kind: JobKind::Run { out },
                            inv,
                        });
                    }
                }
            }
        }
        Ok(jobs)
    }

    pub fn view<'a>(&'a self, ctx: &Ctx<'a>) -> Element<'a, Msg> {
        let f = self.filter.to_ascii_lowercase();
        let ops = ctx
            .ops
            .iter()
            .filter(|o| {
                o.source == "workspace"
                    || self.selected.contains(&o.arg)
                    || (!f.is_empty() && o.name.to_ascii_lowercase().contains(&f))
            })
            .filter(|o| f.is_empty() || o.name.to_ascii_lowercase().contains(&f))
            .map(|o| {
                let a = o.arg.clone();
                checkbox(self.selected.contains(&o.arg))
                    .label(match &o.app {
                        Some(app) => format!("{}  ({app})", o.name),
                        None => o.name.clone(),
                    })
                    .on_toggle(move |b| Msg::ToggleOp(a.clone(), b))
                    .size(16)
                    .text_size(13)
                    .into()
            });
        let op_picker = column![
            text_input("filter (type to search the Deeploy tests too)", &self.filter).on_input(Msg::Filter),
            scrollable(lines(ops)).height(Length::Fixed(170.0)),
        ]
        .spacing(6);

        let mode = row![
            radio("Isolated cores (run.py)", Mode::Isolated, Some(self.mode), Msg::Mode).size(16).text_size(14),
            radio("Heterogeneous SoC (run_hetero.py)", Mode::Hetero, Some(self.mode), Msg::Mode).size(16).text_size(14),
        ]
        .spacing(24);

        let options: Element<'a, Msg> = match self.mode {
            Mode::Isolated => column![
                labelled(
                    "Cores",
                    row(CORES.iter().map(|&c| checkbox(self.cores.contains(c)).label(c).on_toggle(move |b| Msg::ToggleCore(c, b)).into())).spacing(16)
                ),
                labelled("Memory", radios(&[("modelled (real)", "real"), ("ideal (1-cycle)", "ideal")], self.memory, Msg::Memory)),
                labelled("Main memory", pick_list(DramChoice::ALL, Some(self.dram), Msg::Dram).text_size(13)),
                labelled("Spatz kernels", radios(&[("hand-written RVV", "tuned"), ("autovectorized", "autovec")], self.spatz_kernels, Msg::SpatzKernels)),
                labelled("Timeout [s]", text_input("600", &self.timeout).on_input(Msg::Timeout).width(Length::Fixed(90.0))),
                muted("ara is the CVA6 host with its Ara vector unit; each core runs the best code the pipeline has for it."),
            ]
            .spacing(8)
            .into(),
            Mode::Hetero => {
                let any_kws = self.selected.iter().any(|a| ctx.ops.iter().any(|o| &o.arg == a && o.app.as_deref() == Some("kws")));
                let mut c = column![
                    labelled("Host", radios(&[("CVA6", "cva6"), ("CVA6 + Ara", "ara")], self.host, Msg::Host)),
                    labelled("Main memory", pick_list(DramChoice::ALL, Some(self.dram), Msg::Dram).text_size(13)),
                    labelled(
                        "Placement",
                        row(PLACEMENTS.iter().map(|&p| {
                            let label = if p == "mapped" { "mapper decides".to_string() } else { format!("pin to {p}") };
                            checkbox(self.placements.contains(p)).label(label).on_toggle(move |b| Msg::TogglePlacement(p, b)).into()
                        }))
                        .spacing(16)
                    ),
                    labelled("Samples", text_input("16", &self.images).on_input(Msg::Images).width(Length::Fixed(90.0))),
                    labelled("", checkbox(self.power).label("Measure energy (--power, slower)").on_toggle(Msg::Power)),
                ]
                .spacing(8);
                if any_kws {
                    c = c.push(labelled(
                        "KWS front-end",
                        row![
                            radios(&[("on Snitch", "snitch"), ("on Spatz", "spatz")], self.frontend, Msg::Frontend),
                            checkbox(self.serial).label("serial (no overlap)").on_toggle(Msg::Serial),
                        ]
                        .spacing(16),
                    ));
                }
                c.push(muted("One job per op and placement. Samples caps how much of an application's evaluation set runs.")).into()
            }
        };

        let n_jobs = match self.mode {
            Mode::Isolated => self.selected.len(),
            Mode::Hetero => self.selected.len() * self.placements.len(),
        };
        let run_row = row![
            button(text(format!("Run {n_jobs} job{}", if n_jobs == 1 { "" } else { "s" })).size(14))
                .style(button::primary)
                .on_press_maybe((n_jobs > 0).then_some(Msg::Run)),
        ]
        .push(self.error.as_deref().map(error))
        .spacing(12)
        .align_y(iced::Center);

        let setup = section(
            "Run",
            column![
                mode,
                row![
                    container(op_picker).width(Length::FillPortion(2)),
                    container(options).width(Length::FillPortion(3))
                ]
                .spacing(16),
                run_row
            ]
            .spacing(12),
        );

        let toolbar = row![
            small_button("Open result file…", Some(Msg::Open)),
            small_button("Clear", (!self.results.is_empty()).then_some(Msg::Clear)),
            checkbox(self.log_scale).label("log scale").on_toggle(Msg::LogScale),
            checkbox(self.show_caches).label("cache counters").on_toggle(Msg::ShowCaches),
            checkbox(self.show_mapping).label("node mapping").on_toggle(Msg::ShowMapping),
        ]
        .spacing(12)
        .align_y(iced::Center);

        let mut cards = column![].spacing(12);
        if self.results.is_empty() {
            cards = cards.push(muted(
                "Results of finished runs appear here. You can also open earlier ones, e.g. results/*.json.",
            ));
        }
        // Isolated results one card each; hetero results grouped by op, since
        // the comparison there is across placements of the same op.
        let mut hetero_ops: Vec<&str> = Vec::new();
        for (i, l) in self.results.iter().enumerate() {
            match &l.result {
                RunResult::Isolated(r) => cards = cards.push(self.isolated_card(i, l, r)),
                RunResult::Hetero(r) => {
                    if !hetero_ops.contains(&r.op.as_str()) {
                        hetero_ops.push(&r.op);
                    }
                }
            }
        }
        for op in hetero_ops {
            let group: Vec<(usize, &Loaded, &HeteroResult)> = self
                .results
                .iter()
                .enumerate()
                .filter_map(|(i, l)| match &l.result {
                    RunResult::Hetero(r) if r.op == op => Some((i, l, &**r)),
                    _ => None,
                })
                .collect();
            cards = cards.push(self.hetero_card(op, &group));
        }

        scrollable(
            column![setup, section("Results", column![toolbar, cards].spacing(12))]
                .spacing(12)
                .padding(iced::Padding { right: 12.0, ..Default::default() }),
        )
        .into()
    }

    fn isolated_card<'a>(&'a self, i: usize, l: &'a Loaded, r: &'a IsolatedResult) -> Element<'a, Msg> {
        let base = r.baseline();
        let best = r.results.iter().filter_map(|c| c.cycles).min();
        let bars = r
            .results
            .iter()
            .map(|c| match c.cycles {
                Some(cyc) => Bar {
                    label: c.core.clone(),
                    value: cyc as f64,
                    shown: match base {
                        Some(b) => format!("{} ({:.1}×)", fmt_cycles(cyc), b as f64 / cyc as f64),
                        None => fmt_cycles(cyc),
                    },
                    highlight: Some(cyc) == best,
                    missing: false,
                },
                None => {
                    Bar { label: c.core.clone(), value: 0.0, shown: c.status.clone(), highlight: false, missing: true }
                }
            })
            .collect();

        let t = table(
            [
                table::column(text("core").size(13), |c: &crate::model::CoreResult| text(c.core.clone()).size(13)),
                table::column(text("status").size(13), |c: &crate::model::CoreResult| status(&c.status)),
                table::column(text("cycles").size(13), |c: &crate::model::CoreResult| {
                    mono(c.cycles.map_or("-".into(), |v| v.to_string()))
                })
                .align_x(iced::Right),
                table::column(text("vs cva6").size(13), move |c: &crate::model::CoreResult| {
                    mono(match (base, c.cycles) {
                        (Some(b), Some(v)) => format!("{:.2}×", b as f64 / v as f64),
                        _ => "-".into(),
                    })
                })
                .align_x(iced::Right),
                table::column(text("max |err|").size(13), |c: &crate::model::CoreResult| {
                    mono(c.maxdiff.map_or("-".into(), |v| format!("{v:.2e}")))
                })
                .align_x(iced::Right),
                table::column(text("sim wall").size(13), |c: &crate::model::CoreResult| {
                    mono(c.sim_wall_s.map_or("-".into(), |v| format!("{v:.1}s")))
                })
                .align_x(iced::Right),
            ],
            r.results.iter(),
        )
        .padding_x(10)
        .padding_y(3);

        let mut body = column![
            row![
                text(format!(
                    "{} — {} memory, spatz {}",
                    r.op,
                    if r.memory == "ideal" {
                        "ideal".to_string()
                    } else if r.dram.is_empty() || r.dram == "fixed" {
                        "modelled".to_string()
                    } else {
                        DramChoice::from_arg(&r.dram).to_string()
                    },
                    r.spatz_kernels
                ))
                .size(15)
                .width(Length::Fill),
                small_button("Remove", Some(Msg::Remove(i))),
            ]
            .align_y(iced::Center),
            muted(l.source.clone()),
            BarChart { bars, log: self.log_scale }.view(),
            t,
        ]
        .spacing(8);
        if self.show_caches {
            let rows: Vec<(String, crate::model::Cache)> =
                r.results.iter().flat_map(|c| c.caches.iter().map(move |k| (c.core.clone(), k.clone()))).collect();
            if !rows.is_empty() {
                body = body.push(caches_table(rows));
            }
        }
        container(body).padding(10).style(container::rounded_box).width(Length::Fill).into()
    }

    fn hetero_card<'a>(&'a self, op: &'a str, group: &[(usize, &'a Loaded, &'a HeteroResult)]) -> Element<'a, Msg> {
        let best = group.iter().filter_map(|(_, _, r)| r.result.headline().map(|h| h.0)).min();
        let unit = group.iter().find_map(|(_, _, r)| r.result.headline().map(|h| h.1)).unwrap_or("cycles");
        let bars = group
            .iter()
            .map(|(_, _, r)| match r.result.headline() {
                Some((c, _)) => Bar {
                    label: r.variant(),
                    value: c as f64,
                    shown: fmt_cycles(c),
                    highlight: Some(c) == best,
                    missing: false,
                },
                None => Bar {
                    label: r.variant(),
                    value: 0.0,
                    shown: r.result.status.clone(),
                    highlight: false,
                    missing: true,
                },
            })
            .collect();

        let mut rows = column![].spacing(4);
        for (i, l, r) in group {
            let engines = r
                .result
                .per_engine_cycles
                .iter()
                .map(|(k, v)| format!("{k} {}", fmt_cycles(*v)))
                .collect::<Vec<_>>()
                .join(", ");
            let mut extra = Vec::new();
            if let Some(a) = r.result.accuracy {
                extra.push(format!("accuracy {:.1}%", a * 100.0));
            }
            if let Some(m) = r.result.maxdiff {
                extra.push(format!("max |err| {m:.2e}"));
            }
            if let Some(o) = r.result.offload_failures.filter(|o| *o > 0) {
                extra.push(format!("{o} failed offloads"));
            }
            if let Some(w) = r.result.wall_s {
                extra.push(format!("wall {w:.0}s"));
            }
            rows = rows.push(
                row![
                    status(&r.result.status).width(Length::Fixed(70.0)),
                    text(r.variant()).size(13).width(Length::Fixed(230.0)),
                    column![
                        text(format!("engines: {engines}")).size(12),
                        muted(format!("{}   {}", extra.join(" · "), l.source))
                    ]
                    .width(Length::Fill),
                    small_button("Remove", Some(Msg::Remove(*i))),
                ]
                .spacing(8)
                .align_y(iced::Center),
            );
            if r.result.status != "ok" && !r.result.log_tail.is_empty() {
                rows = rows
                    .push(mono(r.result.log_tail.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n")));
            }
        }
        let mut body = column![
            text(format!("{op} on the SoC — {unit}")).size(15),
            BarChart { bars, log: self.log_scale }.view(),
            rows
        ]
        .spacing(8);
        if self.show_mapping {
            for (_, _, r) in group {
                let nodes = r
                    .mapping
                    .nodes
                    .iter()
                    .map(|n| format!("{}:{}→{}", n.node, n.op, n.engine))
                    .collect::<Vec<_>>()
                    .join("  ");
                body = body.push(column![text(r.variant()).size(12), mono(nodes)].spacing(2));
            }
        }
        if self.show_caches {
            let rows: Vec<(String, crate::model::Cache)> = group
                .iter()
                .flat_map(|(_, _, r)| r.result.caches.iter().map(move |k| (r.variant(), k.clone())))
                .collect();
            if !rows.is_empty() {
                body = body.push(caches_table(rows));
            }
        }
        container(body).padding(10).style(container::rounded_box).width(Length::Fill).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_jobs_make_fixed_memory_explicit() {
        let invocation =
            with_explicit_dram(Invocation::new("pipeline/run_hetero.py"), DramChoice::Fixed);
        assert_eq!(invocation.args, vec!["--dram".to_string(), "fixed".to_string()]);
    }

    #[test]
    fn compare_jobs_keep_non_fixed_memory_explicit() {
        let invocation =
            with_explicit_dram(Invocation::new("pipeline/run_hetero.py"), DramChoice::Lpddr5);
        assert_eq!(invocation.args, vec!["--dram".to_string(), "lpddr5".to_string()]);
    }
}

fn radios<'a>(
    opts: &[(&'static str, &'static str)],
    sel: &'static str,
    on: fn(&'static str) -> Msg,
) -> Element<'a, Msg> {
    row(opts.iter().map(|&(label, v)| radio(label, v, Some(sel), on).size(16).text_size(13).into())).spacing(16).into()
}

fn caches_table<'a>(rows: Vec<(String, crate::model::Cache)>) -> Element<'a, Msg> {
    type R = (String, crate::model::Cache);
    table(
        [
            table::column(text("run").size(12), |r: R| text(r.0).size(12)),
            table::column(text("cache").size(12), |r: R| text(r.1.cache).size(12)),
            table::column(text("accesses").size(12), |r: R| mono(r.1.accesses.to_string())).align_x(iced::Right),
            table::column(text("misses").size(12), |r: R| mono(r.1.misses.to_string())).align_x(iced::Right),
            table::column(text("hit rate").size(12), |r: R| {
                mono(r.1.hit_rate().map_or("-".into(), |h| format!("{:.2}%", h * 100.0)))
            })
            .align_x(iced::Right),
            table::column(text("latency cyc").size(12), |r: R| mono(r.1.latency_cycles.to_string()))
                .align_x(iced::Right),
            table::column(text("dyn. energy").size(12), |r: R| {
                mono(r.1.dynamic_pj.map_or("-".into(), |p| format!("{:.1} nJ", p / 1e3)))
            })
            .align_x(iced::Right),
        ],
        rows,
    )
    .padding_x(10)
    .padding_y(2)
    .into()
}
