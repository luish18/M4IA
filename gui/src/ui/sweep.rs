//! Sweep editor: models, options and the design space; preview; launch.

use super::common::*;
use super::{Ctx, NewJob, Out, Tab, query, resolved, valid_name};
use crate::experiment::{ExperimentRequest, ResolvedExperiment};
use crate::jobs::JobKind;
use crate::model::{Validation, fmt_knob_value};
use crate::space::{Design, SpaceMode, SweepSpec, parse_values};
use iced::widget::{
    button, checkbox, column, container, pick_list, radio, row, scrollable, text, text_editor, text_input, tooltip,
};
use iced::{Element, Length, Task};
use std::path::PathBuf;

/// Above this many cells the preview warns before launching.
const BIG: usize = 60;

pub struct State {
    pub spec: SweepSpec,
    list: text_editor::Content,
    validation: Option<Result<Validation, String>>,
    validating: bool,
    /// Median seconds per cell over earlier sweeps, for the estimate.
    per_cell_s: Option<f64>,
    error: Option<String>,
    /// The baseline cell of each model, as the backend resolves it.
    resolved: Vec<(String, Result<ResolvedExperiment, String>)>,
    resolving: bool,
    resolved_open: Option<usize>,
}

impl Default for State {
    fn default() -> Self {
        State {
            spec: SweepSpec::default(),
            list: text_editor::Content::new(),
            validation: None,
            validating: false,
            per_cell_s: None,
            error: None,
            resolved: Vec::new(),
            resolving: false,
            resolved_open: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    ToggleModel(String, bool),
    Preset(&'static [&'static str]),
    Name(String),
    Host(Choice),
    Images(String),
    Power(bool),
    Frontend(FrontendChoice),
    Serial(bool),
    Dram(Choice),
    Mode(SpaceMode),
    Values(String, String),
    LoadOfat,
    ClearValues,
    ListEdit(text_editor::Action),
    Validate,
    Validated(Result<Validation, String>),
    Resolve,
    /// The spec the answers are for, and one answer per model.
    Resolved(Box<SweepSpec>, Vec<(String, Result<ResolvedExperiment, String>)>),
    OpenResolved(Option<usize>),
    Copy(String),
    Save,
    Saved(Result<Option<PathBuf>, String>),
    Load,
    Loaded(Option<Result<SweepSpec, String>>),
    Run,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontendChoice {
    Default,
    Snitch,
    Spatz,
}

impl FrontendChoice {
    const ALL: [FrontendChoice; 3] = [FrontendChoice::Default, FrontendChoice::Snitch, FrontendChoice::Spatz];
}

impl std::fmt::Display for FrontendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FrontendChoice::Default => "default (snitch)",
            FrontendChoice::Snitch => "snitch",
            FrontendChoice::Spatz => "spatz",
        })
    }
}

impl State {
    fn known(ctx: &Ctx) -> Vec<String> {
        ctx.knobs.map(|k| k.defaults.keys().cloned().collect()).unwrap_or_default()
    }

    fn designs(&self, ctx: &Ctx) -> Result<Vec<Design>, String> {
        self.spec.expand(&Self::known(ctx))
    }

    pub fn out_dir(&self) -> String {
        format!("work/gui/sweeps/{}", self.spec.name)
    }

    /// Refuse what sweep/run.py would refuse after a docker start: a
    /// build-time knob (VLEN) away from the build that exists.
    fn build_time_problem(&self, ctx: &Ctx, designs: &[Design]) -> Option<String> {
        let k = ctx.knobs?;
        for d in designs {
            for bt in &k.build_time {
                if let Some(v) = d.get(bt)
                    && Some(*v) != k.default_of(bt)
                {
                    return Some(format!(
                        "{bt}={v} needs a GVSoC build that does not exist (VLEN is compiled into the model); only {} is built",
                        k.default_of(bt).unwrap_or_default()
                    ));
                }
            }
        }
        None
    }

    pub fn set_per_cell(&mut self, s: Option<f64>) {
        self.per_cell_s = s;
    }

    pub fn update(&mut self, msg: Msg, ctx: &Ctx) -> Out<Msg> {
        let before = self.spec.clone();
        let out = match msg {
            Msg::ToggleModel(m, on) => {
                self.spec.models.retain(|x| x != &m);
                if on {
                    self.spec.models.push(m);
                }
                Out::none()
            }
            Msg::Preset(ms) => {
                self.spec.models = ms.iter().map(|s| s.to_string()).collect();
                Out::none()
            }
            Msg::Name(s) => {
                self.spec.name = s;
                Out::none()
            }
            Msg::Host(h) => {
                self.spec.host = h.value;
                Out::none()
            }
            Msg::Images(s) => {
                if let Ok(n) = s.trim().parse() {
                    self.spec.images = n;
                } else if s.trim().is_empty() {
                    self.spec.images = 0;
                }
                Out::none()
            }
            Msg::Power(b) => {
                self.spec.power = b;
                Out::none()
            }
            Msg::Dram(d) => {
                self.spec.dram = d.value;
                Out::none()
            }
            Msg::Frontend(f) => {
                self.spec.frontend = match f {
                    FrontendChoice::Default => None,
                    FrontendChoice::Snitch => Some("snitch".into()),
                    FrontendChoice::Spatz => Some("spatz".into()),
                };
                Out::none()
            }
            Msg::Serial(b) => {
                self.spec.serial = b;
                Out::none()
            }
            Msg::Mode(m) => {
                self.spec.mode = m;
                Out::none()
            }
            Msg::Values(k, v) => {
                self.spec.values.insert(k, v);
                Out::none()
            }
            Msg::LoadOfat => {
                if let Some(k) = ctx.knobs {
                    self.spec.values.clear();
                    for (knob, vs) in &k.ofat {
                        let vals: Vec<String> = vs
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|v| v.as_i64())
                            .map(|v| v.to_string())
                            .collect();
                        self.spec.values.insert(knob.clone(), vals.join(", "));
                    }
                    self.spec.mode = SpaceMode::Ofat;
                }
                Out::none()
            }
            Msg::ClearValues => {
                self.spec.values.clear();
                Out::none()
            }
            Msg::ListEdit(a) => {
                self.list.perform(a);
                self.spec.list = self.list.text();
                Out::none()
            }
            Msg::Validate => match self.designs(ctx) {
                Ok(designs) => {
                    let rel = "work/gui/tmp/validate.json";
                    let path = ctx.settings.workspace.join(rel);
                    let write = std::fs::create_dir_all(path.parent().unwrap())
                        .and_then(|_| std::fs::write(&path, serde_json::to_string(&designs).unwrap()));
                    if let Err(e) = write {
                        self.error = Some(e.to_string());
                        return Out::none();
                    }
                    self.validating = true;
                    Out::task(Task::perform(
                        query(ctx.settings.clone(), vec!["validate".into(), rel.into()]),
                        Msg::Validated,
                    ))
                }
                Err(e) => {
                    self.error = Some(e);
                    Out::none()
                }
            },
            Msg::Validated(r) => {
                self.validating = false;
                self.validation = Some(r);
                return Out::none();
            }
            Msg::Resolve => {
                if self.spec.models.is_empty() {
                    self.error = Some("choose at least one model".into());
                    return Out::none();
                }
                let requests = self.baseline_requests(ctx);
                let models: Vec<String> = self.spec.models.clone();
                let spec = Box::new(self.spec.clone());
                self.resolving = true;
                self.resolved.clear();
                self.resolved_open = None;
                return Out::task(Task::perform(super::resolve_all(ctx.settings.clone(), requests), move |answers| {
                    Msg::Resolved(spec.clone(), models.iter().cloned().zip(answers).collect())
                }));
            }
            Msg::Resolved(spec, answers) => {
                self.resolving = false;
                if *spec == self.spec {
                    self.resolved = answers;
                }
                return Out::none();
            }
            Msg::OpenResolved(i) => {
                self.resolved_open = i;
                return Out::none();
            }
            Msg::Copy(s) => {
                return Out { task: iced::clipboard::write(s), toast: Some("copied".into()), ..Out::none() };
            }
            Msg::Save => {
                let spec = self.spec.clone();
                Out::task(Task::perform(
                    async move {
                        let Some(h) = rfd::AsyncFileDialog::new()
                            .set_file_name(format!("{}.toml", spec.name))
                            .add_filter("sweep spec", &["toml"])
                            .save_file()
                            .await
                        else {
                            return Ok(None);
                        };
                        let text = toml::to_string_pretty(&spec).map_err(|e| e.to_string())?;
                        std::fs::write(h.path(), text).map_err(|e| e.to_string())?;
                        Ok(Some(h.path().to_path_buf()))
                    },
                    Msg::Saved,
                ))
            }
            Msg::Saved(Ok(Some(p))) => Out::toast(format!("saved {}", p.display())),
            Msg::Saved(Ok(None)) => Out::none(),
            Msg::Saved(Err(e)) => {
                self.error = Some(e);
                Out::none()
            }
            Msg::Load => {
                let dir = ctx.settings.workspace.join("work").join("gui").join("sweeps");
                Out::task(Task::perform(
                    async move {
                        let h = rfd::AsyncFileDialog::new()
                            .set_directory(dir)
                            .add_filter("sweep spec", &["toml"])
                            .pick_file()
                            .await?;
                        Some(
                            std::fs::read_to_string(h.path())
                                .map_err(|e| e.to_string())
                                .and_then(|t| toml::from_str::<SweepSpec>(&t).map_err(|e| e.to_string())),
                        )
                    },
                    Msg::Loaded,
                ))
            }
            Msg::Loaded(Some(Ok(spec))) => {
                self.list = text_editor::Content::with_text(&spec.list);
                self.spec = spec;
                Out::none()
            }
            Msg::Loaded(Some(Err(e))) => {
                self.error = Some(e);
                Out::none()
            }
            Msg::Loaded(None) => Out::none(),
            Msg::Run => match self.launch(ctx) {
                Ok(job) => {
                    self.error = None;
                    let mut out = Out::jobs(vec![job]);
                    out.goto = Some(Tab::Results);
                    out
                }
                Err(e) => {
                    self.error = Some(e);
                    Out::none()
                }
            },
        };
        if self.spec != before {
            // The preview is of the old space now.
            self.validation = None;
            self.error = None;
            self.resolved.clear();
            self.resolved_open = None;
        }
        out
    }

    /// One request per model for the baseline design, built as
    /// sweep/run.py::resolve_cell_experiment builds the cell's, so the
    /// fingerprint shown is the one the driver will record.
    pub fn baseline_requests(&self, ctx: &Ctx) -> Vec<(String, ExperimentRequest)> {
        let s = &self.spec;
        s.models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let app = ctx.ops.iter().find(|o| &o.arg == m).and_then(|o| o.app.as_deref());
                let req = ExperimentRequest::new(m, &s.host, &s.dram).execution(
                    app,
                    s.frontend.as_deref(),
                    s.serial,
                    s.power,
                );
                (format!("work/gui/tmp/sweep-preview/{i}.request.json"), req)
            })
            .collect()
    }

    #[cfg(test)]
    pub fn resolved(&self) -> &[(String, Result<ResolvedExperiment, String>)] {
        &self.resolved
    }

    fn launch(&self, ctx: &Ctx) -> Result<NewJob, String> {
        if !valid_name(&self.spec.name) {
            return Err("sweep name: letters, digits, '-' and '_' only".into());
        }
        if self.spec.models.is_empty() {
            return Err("choose at least one model".into());
        }
        if self.spec.images == 0 {
            return Err("samples per run must be at least 1".into());
        }
        let designs = self.designs(ctx)?;
        if let Some(p) = self.build_time_problem(ctx, &designs) {
            return Err(p);
        }
        let out = self.out_dir();
        let dir = ctx.settings.workspace.join(&out);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        std::fs::write(dir.join("designs.json"), serde_json::to_string_pretty(&designs).unwrap())
            .map_err(|e| e.to_string())?;
        std::fs::write(dir.join("spec.toml"), toml::to_string_pretty(&self.spec).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let inv = self.spec.invocation(&out, &format!("{out}/designs.json"));
        let cells = self.spec.cells(designs.len());
        Ok(NewJob { title: format!("Sweep {} ({cells} cells)", self.spec.name), kind: JobKind::Sweep { out }, inv })
    }

    pub fn view<'a>(&'a self, ctx: &Ctx<'a>, busy: bool) -> Element<'a, Msg> {
        let s = &self.spec;
        let models = ctx.ops.iter().filter(|o| o.source == "workspace" || s.models.contains(&o.arg)).map(|o| {
            let a = o.arg.clone();
            checkbox(s.models.contains(&o.arg))
                .label(match &o.app {
                    Some(app) => format!("{}  ({app})", o.arg),
                    None => o.arg.clone(),
                })
                .on_toggle(move |b| Msg::ToggleModel(a.clone(), b))
                .size(16)
                .text_size(13)
                .into()
        });
        let presets = row![
            text("Presets").size(13),
            small_button("MNIST", Some(Msg::Preset(&["ops/mnist"]))),
            small_button("KWS", Some(Msg::Preset(&["ops/kws"]))),
            small_button("MNIST + KWS", Some(Msg::Preset(&["ops/mnist", "ops/kws"]))),
        ]
        .spacing(8)
        .align_y(iced::Center);
        let any_kws = s.models.iter().any(|m| ctx.ops.iter().any(|o| &o.arg == m && o.app.as_deref() == Some("kws")));
        let fe = match s.frontend.as_deref() {
            Some("snitch") => FrontendChoice::Snitch,
            Some("spatz") => FrontendChoice::Spatz,
            _ => FrontendChoice::Default,
        };
        let hosts = host_choices(ctx.schema);
        let memories = memory_choices(ctx.schema);
        let mut opts = column![
            labelled("Name", text_input("sweep", &s.name).on_input(Msg::Name).width(Length::Fixed(220.0))),
            labelled("", muted(format!("writes {}/", self.out_dir()))),
            labelled("Host", pick_list(hosts.clone(), Some(chosen(&hosts, &s.host)), Msg::Host).text_size(13)),
            labelled(
                "Samples/run",
                text_input("16", &s.images.to_string()).on_input(Msg::Images).width(Length::Fixed(90.0))
            ),
            labelled("", checkbox(s.power).label("Measure energy (--power, slower)").on_toggle(Msg::Power)),
            labelled(
                "Main memory",
                pick_list(memories.clone(), Some(chosen(&memories, &s.dram)), Msg::Dram).text_size(13)
            ),
        ]
        .spacing(8);
        if any_kws {
            opts = opts.push(labelled(
                "KWS front-end",
                row![
                    pick_list(FrontendChoice::ALL, Some(fe), Msg::Frontend).text_size(13),
                    checkbox(s.serial).label("serial").on_toggle(Msg::Serial),
                ]
                .spacing(12)
                .align_y(iced::Center),
            ));
        }

        let setup = section(
            "Models and options",
            row![
                column![presets, scrollable(lines(models)).height(Length::Fixed(150.0))]
                    .spacing(8)
                    .width(Length::FillPortion(1)),
                opts.width(Length::FillPortion(1)),
            ]
            .spacing(16),
        );

        let space = section("Parameter space", self.space_view(ctx));

        // --- preview ---
        let mut preview = column![].spacing(6);
        let mut can_run = !busy;
        match self.designs(ctx) {
            Err(e) => {
                preview = preview.push(error(e));
                can_run = false;
            }
            Ok(designs) => {
                let cells = s.cells(designs.len());
                let eta = self.per_cell_s.map(|p| p * cells as f64);
                preview = preview.push(text(format!(
                    "{} design point{} + baseline × {} model{} = {cells} cells{}",
                    designs.len(),
                    if designs.len() == 1 { "" } else { "s" },
                    s.models.len(),
                    if s.models.len() == 1 { "" } else { "s" },
                    eta.map(|e| format!(", roughly {}", human_duration(e))).unwrap_or_default()
                )));
                if cells > BIG {
                    preview = preview.push(warn(format!(
                        "That is a lot of cells; a screening pass is usually one factor at a time (≤ {BIG})."
                    )));
                }
                if let Some(p) = self.build_time_problem(ctx, &designs) {
                    preview = preview.push(error(p));
                    can_run = false;
                }
                if designs.is_empty() {
                    preview = preview.push(muted("Only the baseline would run: give some knob values."));
                }
            }
        }
        match &self.validation {
            Some(Ok(v)) => {
                let bad: Vec<_> = v.designs.iter().filter(|d| !d.reasons.is_empty()).collect();
                if v.designs.iter().any(|d| d.needs_build) {
                    preview = preview.push(error(
                        "Some points need a GVSoC build that does not exist; the driver will refuse the sweep.",
                    ));
                }
                if bad.is_empty() {
                    preview = preview.push(ok("Every design point is buildable."));
                } else {
                    preview = preview.push(warn(format!(
                        "{} of {} points are not buildable; they will be recorded as infeasible, not run:",
                        bad.len(),
                        v.designs.len()
                    )));
                    for d in bad.iter().take(12) {
                        preview = preview.push(mono(format!(
                            "  {}: {}",
                            d.slug.clone().unwrap_or_default(),
                            d.reasons.join("; ")
                        )));
                    }
                }
            }
            Some(Err(e)) => preview = preview.push(error(format!("check failed: {e}"))),
            None => {}
        }
        for (i, (model, r)) in self.resolved.iter().enumerate() {
            let open = self.resolved_open == Some(i);
            preview = preview.push(match r {
                Ok(r) => row![
                    text(format!("baseline · {model}")).size(13).width(Length::Fixed(220.0)),
                    mono(format!(
                        "{} · {} · {}",
                        r.memory.label,
                        r.simulator.target,
                        crate::experiment::short(&r.resolved_fingerprint)
                    ))
                    .width(Length::Fill),
                    small_button(if open { "Hide" } else { "Details" }, Some(Msg::OpenResolved((!open).then_some(i)))),
                ]
                .spacing(8)
                .align_y(iced::Center),
                Err(e) => {
                    row![text(format!("baseline · {model}")).size(13).width(Length::Fixed(220.0)), error(e.as_str())]
                        .spacing(8)
                }
            });
            if open && let Ok(r) = r {
                preview = preview.push(
                    container(resolved::summary(r, Some(Msg::Copy(r.raw.clone()))))
                        .padding(8)
                        .style(container::rounded_box),
                );
            }
        }
        preview = preview.push(
            row![
                small_button(
                    if self.validating { "Checking…" } else { "Check designs" },
                    (!self.validating).then_some(Msg::Validate)
                ),
                small_button(
                    if self.resolving { "Resolving…" } else { "Resolve baseline" },
                    (!self.resolving && !s.models.is_empty()).then_some(Msg::Resolve)
                ),
                small_button("Save spec…", Some(Msg::Save)),
                small_button("Load spec…", Some(Msg::Load)),
                button(text("Run sweep").size(14)).style(button::primary).on_press_maybe(can_run.then_some(Msg::Run)),
            ]
            .push(busy.then(|| muted("a sweep is already queued or running")))
            .spacing(8)
            .align_y(iced::Center),
        );
        if let Some(e) = &self.error {
            preview = preview.push(error(e.as_str()));
        }

        scrollable(
            column![setup, space, section("Preview", preview)]
                .spacing(12)
                .padding(iced::Padding { right: 12.0, ..Default::default() }),
        )
        .into()
    }

    fn space_view<'a>(&'a self, ctx: &Ctx<'a>) -> Element<'a, Msg> {
        let s = &self.spec;
        let modes = row(SpaceMode::ALL
            .iter()
            .map(|&m| radio(m.to_string(), m, Some(s.mode), Msg::Mode).size(16).text_size(13).into()))
        .spacing(20);
        let hint = muted(match s.mode {
            SpaceMode::Ofat => {
                "The baseline, then each value below with every other knob at its baseline. This is what the sensitivity report reads."
            }
            SpaceMode::Factorial => {
                "Every combination of the values below. Include a knob's baseline value in its list to also get the one-knob-at-a-time points the sensitivity report uses."
            }
            SpaceMode::List => {
                "One design per line, e.g. SPATZ_NB_LANES=8 SPATZ_NB_CORE=17 (hex and 16k/1M accepted; # starts a comment). Unset knobs stay at baseline."
            }
        });
        let Some(k) = ctx.knobs else {
            return column![modes, muted("Loading the knob list from the pipeline…")].spacing(8).into();
        };
        if s.mode == SpaceMode::List {
            let names = k.defaults.keys().cloned().collect::<Vec<_>>().join("  ");
            return column![
                modes,
                hint,
                text_editor(&self.list)
                    .on_action(Msg::ListEdit)
                    .height(Length::Fixed(180.0))
                    .font(iced::Font::MONOSPACE),
                muted(format!("Knobs: {names}")),
            ]
            .spacing(8)
            .into();
        }
        let rows = k.defaults.iter().map(|(knob, base)| {
            let base = base.as_i64().unwrap_or_default();
            let typed = s.values.get(knob).cloned().unwrap_or_default();
            let parsed = parse_values(&typed);
            let build_time = k.build_time.contains(knob);
            let note: Element<'a, Msg> = match (&parsed, build_time) {
                (Err(e), _) => error(e.clone()).into(),
                (Ok(v), true) if v.iter().any(|x| *x != base) => warn("needs its own GVSoC build").into(),
                (Ok(_), true) => muted("build-time").into(),
                (Ok(v), false) if !v.is_empty() => {
                    muted(v.iter().map(|x| fmt_knob_value(knob, *x)).collect::<Vec<_>>().join(", ")).into()
                }
                _ => text("").into(),
            };
            let kn = knob.clone();
            let meta = ctx.schema.and_then(|sc| sc.param(knob));
            let name: Element<'a, Msg> = match meta {
                Some(m) => tooltip(
                    mono(knob.as_str()),
                    container(text(m.description.clone()).size(12)).padding(6).style(container::rounded_box),
                    tooltip::Position::Top,
                )
                .into(),
                None => mono(knob.as_str()).into(),
            };
            let label = meta
                .map(|m| match &m.unit {
                    Some(u) => format!("{} [{u}]", m.label),
                    None => m.label.clone(),
                })
                .unwrap_or_default();
            row![
                container(name).width(Length::Fixed(170.0)),
                muted(label).width(Length::Fixed(170.0)),
                mono(fmt_knob_value(knob, base)).width(Length::Fixed(70.0)),
                text_input("values, e.g. 2, 8", &typed)
                    .on_input(move |v| Msg::Values(kn.clone(), v))
                    .width(Length::Fixed(260.0))
                    .size(13),
                note,
            ]
            .spacing(10)
            .align_y(iced::Center)
            .into()
        });
        column![
            modes,
            hint,
            row![
                small_button("Load built-in OFAT grid", Some(Msg::LoadOfat)),
                small_button("Clear values", (!s.values.is_empty()).then_some(Msg::ClearValues)),
            ]
            .spacing(8),
            row![
                muted("knob").width(Length::Fixed(170.0)),
                muted("").width(Length::Fixed(170.0)),
                muted("baseline").width(Length::Fixed(70.0)),
                muted("values to try")
            ]
            .spacing(10),
            container(lines(rows)),
        ]
        .spacing(8)
        .into()
    }
}

pub fn human_duration(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}
