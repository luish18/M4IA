//! The application: shared state, the job queue, and routing between tabs.

use crate::backend;
use crate::experiment::ExperimentSchema;
use crate::jobs::{self, Channel, Job, JobEvent, JobKind, JobStatus};
use crate::model::{Knobs, OpInfo, OpsList, SweepEvent};
use crate::settings::Settings;
use crate::ui::{self, Ctx, NewJob, Out, Tab, common::muted};
use iced::widget::{button, column, container, row, rule, text};
use iced::{Element, Length, Subscription, Task, Theme};
use std::collections::HashSet;
use std::time::{Duration, Instant};

pub struct App {
    settings: Settings,
    draft: Settings,
    tab: Tab,
    ops: Vec<OpInfo>,
    ops_loading: bool,
    ops_error: Option<String>,
    knobs: Option<Knobs>,
    schema: Option<Result<ExperimentSchema, String>>,
    check: Vec<(String, Result<String, String>)>,
    checking: bool,
    jobs: Vec<Job>,
    next_id: u64,
    selected_job: Option<u64>,
    cancel_requested: HashSet<u64>,
    ops_tab: ui::ops::State,
    compare: ui::compare::State,
    sweep: ui::sweep::State,
    results: ui::results::State,
    toast: Option<(String, Instant)>,
    tour: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone)]
pub enum Message {
    Tab(Tab),
    Ops(ui::ops::Msg),
    Compare(ui::compare::Msg),
    Sweep(ui::sweep::Msg),
    Results(ui::results::Msg),
    Jobs(ui::jobs::Msg),
    Settings(ui::settings::Msg),
    OpsLoaded(Result<OpsList, String>),
    KnobsLoaded(Result<Knobs, String>),
    SchemaLoaded(Result<ExperimentSchema, String>),
    Job(u64, JobEvent),
    Tick,
    Noop,
    /// `--tour`: show tab `n`, screenshot it, move on.
    Tour(usize),
    TourShot(usize, iced::window::Screenshot),
}

impl App {
    pub fn new(cli: crate::Cli) -> (App, Task<Message>) {
        let settings = Settings::load();
        let mut app = App {
            draft: settings.clone(),
            settings,
            tab: Tab::Compare,
            ops: Vec::new(),
            ops_loading: false,
            ops_error: None,
            knobs: None,
            schema: None,
            check: Vec::new(),
            checking: false,
            jobs: Vec::new(),
            next_id: 1,
            selected_job: None,
            cancel_requested: HashSet::new(),
            ops_tab: Default::default(),
            compare: Default::default(),
            sweep: Default::default(),
            results: Default::default(),
            toast: None,
            tour: cli.tour.clone(),
        };
        let mut tasks = vec![app.reload_all()];
        let ws = std::fs::canonicalize(&app.settings.workspace).unwrap_or_else(|_| app.settings.workspace.clone());
        // Workspace-relative when inside it (the form every other path takes),
        // so the backend can reach it from inside the container too.
        let rel_to_ws = |p: &std::path::Path| {
            let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            match abs.strip_prefix(&ws) {
                Ok(r) => (abs.clone(), r.to_string_lossy().replace('\\', "/")),
                Err(_) => (abs.clone(), abs.to_string_lossy().into_owned()),
            }
        };
        for p in &cli.open_results {
            let (abs, shown) = rel_to_ws(p);
            if let Err(e) = app.compare.load_file(&abs, shown) {
                app.toast(e);
            }
        }
        if let Some(dir) = &cli.open_sweep {
            let (_, rel) = rel_to_ws(dir);
            app.results.select(&app.settings.workspace.clone(), rel);
            let ctx = Ctx {
                settings: &app.settings,
                ops: &app.ops,
                knobs: app.knobs.as_ref(),
                schema: app.schema.as_ref().and_then(|r| r.as_ref().ok()),
            };
            tasks.push(app.results.analyse(&ctx).map(Message::Results));
            app.tab = Tab::Results;
        }
        if let Some(t) = cli.tab {
            app.tab = t;
        }
        if app.tour.is_some() {
            tasks.push(Task::perform(tokio::time::sleep(Duration::from_secs(10)), |_| Message::Tour(0)));
        }
        (app, Task::batch(tasks))
    }

    pub fn title(&self) -> String {
        let running = self.jobs.iter().filter(|j| j.status == JobStatus::Running).count();
        if running > 0 { format!("hetero-sim ({running} running)") } else { "hetero-sim".into() }
    }

    pub fn theme(&self) -> Option<Theme> {
        // None: follow the system's light/dark preference.
        None
    }

    pub fn subscription(&self) -> Subscription<Message> {
        if self.jobs.iter().any(|j| j.status == JobStatus::Running) || self.toast.is_some() {
            iced::time::every(Duration::from_secs(1)).map(|_| Message::Tick)
        } else {
            Subscription::none()
        }
    }

    /// Everything that depends on the settings: the op list, the knobs, the
    /// environment check, and what is on disk.
    fn reload_all(&mut self) -> Task<Message> {
        self.results.scan(&self.settings.workspace);
        self.sweep.set_per_cell(self.results.per_cell_estimate(&self.settings.workspace));
        self.checking = true;
        let s = self.settings.clone();
        Task::batch([
            self.refresh_ops(),
            Task::perform(ui::query::<Knobs>(s.clone(), vec!["knobs".into()]), Message::KnobsLoaded),
            Task::perform(
                ui::query::<ExperimentSchema>(s.clone(), vec!["experiment-schema".into()]),
                Message::SchemaLoaded,
            ),
            Task::perform(backend::check(s), |r| Message::Settings(ui::settings::Msg::Checked(r))),
        ])
    }

    fn refresh_ops(&mut self) -> Task<Message> {
        self.ops_loading = true;
        Task::perform(ui::query::<OpsList>(self.settings.clone(), vec!["ops".into()]), Message::OpsLoaded)
    }

    fn ctx(&self) -> Ctx<'_> {
        Ctx {
            settings: &self.settings,
            ops: &self.ops,
            knobs: self.knobs.as_ref(),
            schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
        }
    }

    fn toast(&mut self, s: impl Into<String>) {
        self.toast = Some((s.into(), Instant::now()));
    }

    /// Apply what a tab asked for.
    fn absorb<M: 'static + Send>(&mut self, out: Out<M>, wrap: fn(M) -> Message) -> Task<Message> {
        let mut tasks = vec![out.task.map(wrap)];
        if out.refresh_ops {
            tasks.push(self.refresh_ops());
        }
        if let Some(t) = out.toast {
            self.toast(t);
        }
        let started_sweep = out.jobs.iter().find_map(|j| match &j.kind {
            JobKind::Sweep { out } => Some(out.clone()),
            _ => None,
        });
        for j in out.jobs {
            self.enqueue(j);
        }
        if let Some(dir) = started_sweep {
            self.results.select(&self.settings.workspace, dir.clone());
            self.results.live =
                Some((dir, ui::results::Live { running: true, phase: "queued".into(), ..Default::default() }));
        }
        if let Some(tab) = out.goto {
            self.tab = tab;
        }
        tasks.push(self.pump());
        Task::batch(tasks)
    }

    fn enqueue(&mut self, j: NewJob) {
        let id = self.next_id;
        self.next_id += 1;
        self.jobs.push(Job::new(id, j.title, j.kind, j.inv));
        self.selected_job = Some(id);
    }

    /// Start queued jobs while there is room.
    fn pump(&mut self) -> Task<Message> {
        let mut tasks = Vec::new();
        let limit = self.settings.concurrency.max(1);
        loop {
            let running = self.jobs.iter().filter(|j| j.status == JobStatus::Running).count();
            if running >= limit {
                break;
            }
            let Some(job) = self.jobs.iter_mut().find(|j| j.status == JobStatus::Queued) else { break };
            let launch = backend::launch(&self.settings, &job.invocation, job.id);
            job.push_line(format!("$ {}", launch.display()));
            job.launch = Some(launch.clone());
            job.status = JobStatus::Running;
            job.started = Some(Instant::now());
            let id = job.id;
            tasks.push(Task::run(jobs::run(self.settings.clone(), launch), move |e| Message::Job(id, e)));
        }
        Task::batch(tasks)
    }

    fn on_job_event(&mut self, id: u64, e: JobEvent) -> Task<Message> {
        let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) else { return Task::none() };
        match e {
            JobEvent::Started { pid } => job.pid = pid,
            JobEvent::Line(ch, line) => {
                if let (JobKind::Sweep { out }, Channel::Stdout) = (&job.kind, ch)
                    && let Ok(ev) = serde_json::from_str::<SweepEvent>(&line)
                {
                    let out = out.clone();
                    let summary = sweep_event(job, &ev);
                    if let Some(s) = summary {
                        job.push_line(s);
                    }
                    self.on_sweep_event(&out, ev);
                    return Task::none();
                }
                if let JobKind::Run { .. } = job.kind {
                    if let Some(p) = jobs::parse_progress(&line) {
                        job.progress = Some(p);
                    }
                    if ch == Channel::Stdout && !line.trim().is_empty() && line.starts_with('[') {
                        job.phase = line.chars().take(70).collect();
                    }
                }
                let hint = backend::explain_missing_script(&self.settings, &line);
                job.push_line(line);
                if let Some(h) = hint {
                    job.push_line(h.clone());
                    job.phase = h;
                }
            }
            JobEvent::SpawnFailed(msg) => {
                job.push_line(msg.clone());
                job.status = JobStatus::Failed(msg);
                job.ended = Some(Instant::now());
                let t = self.finished(id);
                return Task::batch([t, self.pump()]);
            }
            JobEvent::Exited(code) => {
                job.status =
                    if self.cancel_requested.remove(&id) { JobStatus::Cancelled } else { JobStatus::Finished(code) };
                job.push_line(format!("[{}]", job.status.label()));
                job.ended = Some(Instant::now());
                let t = self.finished(id);
                return Task::batch([t, self.pump()]);
            }
        }
        Task::none()
    }

    fn on_sweep_event(&mut self, out: &str, ev: SweepEvent) {
        let Some((dir, live)) = self.results.live.as_mut() else { return };
        if dir != out {
            return;
        }
        match ev {
            SweepEvent::Start { total } => live.total = total,
            SweepEvent::Cell { cell, .. } => {
                live.cell = cell;
                live.phase = "starting".into();
            }
            SweepEvent::Phase { phase, .. } => live.phase = phase,
            SweepEvent::Row { done, total, eta_s, row } => {
                live.done = done;
                live.total = total;
                live.eta_s = eta_s;
                self.results.push_live_row(out, *row);
            }
            SweepEvent::Done { done, total } => {
                live.done = done;
                live.total = total;
                live.running = false;
            }
        }
    }

    /// What a finished job feeds back into.
    fn finished(&mut self, id: u64) -> Task<Message> {
        let Some(job) = self.jobs.iter().find(|j| j.id == id) else { return Task::none() };
        let ok = job.status == JobStatus::Finished(Some(0));
        let title = job.title.clone();
        match job.kind.clone() {
            JobKind::MakeOp => {
                self.toast(if ok { format!("{title}: done") } else { format!("{title}: failed, see Jobs") });
                self.refresh_ops()
            }
            JobKind::Run { out } => {
                // run_hetero.py exits non-zero for a failed simulation but still
                // writes the result, which is worth showing.
                match self.compare.load(&self.settings.workspace, &out) {
                    Ok(()) => self.toast(format!("{title}: result in Compare cores")),
                    Err(_) if job.status == JobStatus::Cancelled => {}
                    Err(_) => self.toast(format!("{title}: no result written, see Jobs")),
                }
                Task::none()
            }
            JobKind::Sweep { out } => {
                if let Some((dir, live)) = self.results.live.as_mut()
                    && *dir == out
                {
                    live.running = false;
                }
                self.toast(format!("{title}: {}", job.status.label()));
                self.results.scan(&self.settings.workspace);
                self.sweep.set_per_cell(self.results.per_cell_estimate(&self.settings.workspace));
                if self.results.current() == Some(out.as_str()) {
                    self.results.reload(&self.settings.workspace);
                    let ctx = Ctx {
                        settings: &self.settings,
                        ops: &self.ops,
                        knobs: self.knobs.as_ref(),
                        schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
                    };
                    return self.results.analyse(&ctx).map(Message::Results);
                }
                Task::none()
            }
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tab(t) => {
                self.tab = t;
                if t == Tab::Results {
                    self.results.scan(&self.settings.workspace);
                }
                Task::none()
            }
            Message::Ops(m) => {
                let out = {
                    self.ops_tab.update(
                        m,
                        &Ctx {
                            settings: &self.settings,
                            ops: &self.ops,
                            knobs: self.knobs.as_ref(),
                            schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
                        },
                    )
                };
                self.absorb(out, Message::Ops)
            }
            Message::Compare(m) => {
                let out = self.compare.update(
                    m,
                    &Ctx {
                        settings: &self.settings,
                        ops: &self.ops,
                        knobs: self.knobs.as_ref(),
                        schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
                    },
                );
                self.absorb(out, Message::Compare)
            }
            Message::Sweep(m) => {
                let out = self.sweep.update(
                    m,
                    &Ctx {
                        settings: &self.settings,
                        ops: &self.ops,
                        knobs: self.knobs.as_ref(),
                        schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
                    },
                );
                self.absorb(out, Message::Sweep)
            }
            Message::Results(m) => {
                let out = self.results.update(
                    m,
                    &Ctx {
                        settings: &self.settings,
                        ops: &self.ops,
                        knobs: self.knobs.as_ref(),
                        schema: self.schema.as_ref().and_then(|r| r.as_ref().ok()),
                    },
                );
                self.absorb(out, Message::Results)
            }
            Message::Jobs(m) => self.jobs_update(m),
            Message::Settings(m) => self.settings_update(m),
            Message::OpsLoaded(r) => {
                self.ops_loading = false;
                match r {
                    Ok(l) => {
                        self.ops = l.ops;
                        self.ops_error = None;
                    }
                    Err(e) => self.ops_error = Some(e),
                }
                Task::none()
            }
            Message::KnobsLoaded(r) => {
                match r {
                    Ok(k) => self.knobs = Some(k),
                    Err(e) => self.toast(format!("could not read the knob list: {e}")),
                }
                Task::none()
            }
            Message::SchemaLoaded(r) => {
                self.schema = Some(r);
                Task::none()
            }
            Message::Job(id, e) => self.on_job_event(id, e),
            Message::Tick => {
                if self.toast.as_ref().is_some_and(|(_, t)| t.elapsed() > Duration::from_secs(8)) {
                    self.toast = None;
                }
                Task::none()
            }
            Message::Noop => Task::none(),
            Message::Tour(i) => {
                let Some(&tab) = Tab::ALL.get(i) else { return iced::exit() };
                self.tab = tab;
                Task::perform(tokio::time::sleep(Duration::from_millis(1500)), |_| ())
                    .then(|_| iced::window::latest())
                    .then(|id| id.map(iced::window::screenshot).unwrap_or_else(Task::none))
                    .map(move |shot| Message::TourShot(i, shot))
            }
            Message::TourShot(i, shot) => {
                if let Some(dir) = &self.tour {
                    let _ = std::fs::create_dir_all(dir);
                    // Binary PPM: no image dependency, and every converter reads it.
                    let (w, h) = (shot.size.width, shot.size.height);
                    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                    for px in shot.rgba.as_chunks::<4>().0 {
                        ppm.extend_from_slice(&px[..3]);
                    }
                    let name = Tab::ALL[i].label().to_ascii_lowercase().replace(' ', "-").replace('&', "and");
                    let _ = std::fs::write(dir.join(format!("{i}-{name}.ppm")), ppm);
                }
                Task::done(Message::Tour(i + 1))
            }
        }
    }

    fn jobs_update(&mut self, m: ui::jobs::Msg) -> Task<Message> {
        use ui::jobs::Msg;
        match m {
            Msg::Select(id) => self.selected_job = Some(id),
            Msg::Copy(id) => {
                if let Some(cmd) =
                    self.jobs.iter().find(|j| j.id == id).and_then(|j| j.launch.as_ref()).map(|l| l.display())
                {
                    self.toast("command copied");
                    return iced::clipboard::write(cmd);
                }
            }
            Msg::Cancel(id) => {
                let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) else { return Task::none() };
                match job.status {
                    JobStatus::Queued => {
                        job.status = JobStatus::Cancelled;
                        if let JobKind::Sweep { out } = &job.kind
                            && let Some((dir, live)) = self.results.live.as_mut()
                            && dir == out
                        {
                            live.running = false;
                        }
                    }
                    JobStatus::Running => {
                        self.cancel_requested.insert(id);
                        job.push_line("[cancelling…]".into());
                        if let Some(l) = job.launch.clone() {
                            return Task::perform(backend::cancel(l, job.pid), |_| Message::Noop);
                        }
                    }
                    _ => {}
                }
            }
            Msg::ClearFinished => {
                self.jobs.retain(|j| j.status.is_active());
                if self.selected_job.is_some_and(|id| !self.jobs.iter().any(|j| j.id == id)) {
                    self.selected_job = None;
                }
            }
        }
        Task::none()
    }

    fn settings_update(&mut self, m: ui::settings::Msg) -> Task<Message> {
        use ui::settings::Msg;
        match m {
            Msg::Backend(b) => self.draft.backend = b,
            Msg::Workspace(s) => self.draft.workspace = s.into(),
            Msg::Browse => {
                return Task::perform(
                    async { rfd::AsyncFileDialog::new().pick_folder().await.map(|h| h.path().to_path_buf()) },
                    |p| Message::Settings(Msg::Browsed(p)),
                );
            }
            Msg::Browsed(Some(p)) => self.draft.workspace = p,
            Msg::Browsed(None) => {}
            Msg::Image(s) => self.draft.docker_image = s,
            Msg::MountSources(b) => self.draft.mount_sources = b,
            Msg::Concurrency(n) => self.draft.concurrency = n,
            Msg::Revert => self.draft = self.settings.clone(),
            Msg::Apply => {
                self.settings = self.draft.clone();
                if let Err(e) = self.settings.save() {
                    self.toast(format!("settings not saved: {e}"));
                }
                self.knobs = None;
                self.schema = None;
                return Task::batch([self.reload_all(), self.pump()]);
            }
            Msg::Check => {
                self.checking = true;
                return Task::perform(backend::check(self.settings.clone()), |r| Message::Settings(Msg::Checked(r)));
            }
            Msg::Checked(r) => {
                self.checking = false;
                self.check = r;
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        let active = self.jobs.iter().filter(|j| j.status.is_active()).count();
        let tabs = row(Tab::ALL.iter().map(|&t| {
            let label =
                if t == Tab::Jobs && active > 0 { format!("{} ({active})", t.label()) } else { t.label().to_string() };
            button(text(label).size(14))
                .padding([6, 14])
                .style(if t == self.tab { button::primary } else { button::text })
                .on_press(Message::Tab(t))
                .into()
        }))
        .spacing(4);

        let ctx = self.ctx();
        let body: Element<'_, Message> = match self.tab {
            Tab::Ops => self.ops_tab.view(&ctx, self.ops_loading, self.ops_error.as_deref()).map(Message::Ops),
            Tab::Compare => self.compare.view(&ctx).map(Message::Compare),
            Tab::Sweep => {
                let busy = self.jobs.iter().any(|j| {
                    j.status.is_active() && matches!(&j.kind, JobKind::Sweep { out } if *out == self.sweep.out_dir())
                });
                self.sweep.view(&ctx, busy).map(Message::Sweep)
            }
            Tab::Results => self.results.view(&ctx).map(Message::Results),
            Tab::Jobs => ui::jobs::view(&self.jobs, self.selected_job).map(Message::Jobs),
            Tab::Settings => {
                ui::settings::view(&self.draft, &self.settings, &self.check, self.checking).map(Message::Settings)
            }
        };

        let backend = match self.settings.backend {
            crate::settings::BackendKind::Native => "native".to_string(),
            crate::settings::BackendKind::Docker => format!("docker: {}", self.settings.docker_image),
        };
        let problem = self
            .check
            .iter()
            .find_map(|(what, r)| r.as_ref().err().map(|e| format!("{what}: {}", e.lines().next().unwrap_or(""))))
            .or_else(|| match &self.schema {
                Some(Err(e)) => Some(schema_problem(&self.settings, e)),
                _ => None,
            });
        let status_line =
            row![text(self.toast.as_ref().map(|(t, _)| t.clone()).unwrap_or_default()).size(13).width(Length::Fill),]
                .push(problem.map(ui::common::warn))
                .push(muted(format!("{backend} · {}", self.settings.workspace.display())))
                .spacing(12);

        container(
            column![tabs, rule::horizontal(1), container(body).height(Length::Fill), rule::horizontal(1), status_line]
                .spacing(8),
        )
        .padding(12)
        .into()
    }
}

/// Why the memory, host and pin choices are cut down to the defaults.
fn schema_problem(settings: &crate::settings::Settings, e: &str) -> String {
    if crate::experiment::bridge_too_old(e) && settings.backend == crate::settings::BackendKind::Docker {
        format!(
            "\"{}\" predates experiment-schema: only default memory/host offered. Rebuild it or mount sources in Settings.",
            settings.docker_image
        )
    } else {
        format!("experiment-schema: {}", e.lines().next().unwrap_or(""))
    }
}

/// A readable log line for a sweep event, and the job's progress from it.
fn sweep_event(job: &mut Job, ev: &SweepEvent) -> Option<String> {
    match ev {
        SweepEvent::Start { total } => {
            job.progress = Some((0, *total));
            Some(format!("sweep: {total} cells"))
        }
        SweepEvent::Cell { index, cell } => {
            job.phase = cell.clone();
            Some(format!("[{}] {cell}", index + 1))
        }
        SweepEvent::Phase { phase, .. } => {
            job.phase = format!("{} · {phase}", job.phase.split(" · ").take(2).collect::<Vec<_>>().join(" · "));
            None
        }
        SweepEvent::Row { done, total, row, .. } => {
            job.progress = Some((*done, *total));
            let cyc = row.cycles().map(crate::model::fmt_cycles).unwrap_or_else(|| "-".into());
            let why = row
                .reasons
                .first()
                .map(|r| format!("  ({})", r.lines().next().unwrap_or("").chars().take(120).collect::<String>()))
                .unwrap_or_default();
            Some(format!(
                "  {} · {}: {} cycles={cyc} wall={}s{why}",
                row.design_slug,
                row.model,
                row.status,
                row.wall_s.unwrap_or(0.0)
            ))
        }
        SweepEvent::Done { done, total } => Some(format!("sweep finished: {done}/{total} cells")),
    }
}

#[cfg(test)]
mod tests {
    //! The app's own plumbing -- tab -> job -> result -> view state -- driven
    //! through `update` with a real backend and no window. Ignored by default
    //! for the same reason as `e2e`: they need Docker and a built image.

    use super::*;
    use crate::model::RunResult;
    use crate::ui::{compare, results, sweep};
    use iced::futures::StreamExt;

    fn app() -> App {
        let ws = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).canonicalize().unwrap();
        let cfg = ws.join("work/gui/e2e/settings.toml");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        let image = std::env::var("HETERO_GUI_TEST_IMAGE").unwrap_or_else(|_| "hetero-sim:gui".into());
        std::fs::write(
            &cfg,
            // Mount the checkout's pipeline/, runtime/ and targets/: these test
            // this checkout's bridge, not whatever the image was built with.
            format!(
                "backend = \"Docker\"\nworkspace = {:?}\ndocker_image = {image:?}\nmount_sources = true\n",
                ws.to_string_lossy()
            ),
        )
        .unwrap();
        let _ = crate::settings::CONFIG_OVERRIDE.set(cfg);
        App::new(crate::Cli::default()).0
    }

    async fn load_shared(app: &mut App) {
        let ops = ui::query::<OpsList>(app.settings.clone(), vec!["ops".into()]).await;
        let _ = app.update(Message::OpsLoaded(ops));
        let knobs = ui::query::<Knobs>(app.settings.clone(), vec!["knobs".into()]).await;
        let _ = app.update(Message::KnobsLoaded(knobs));
        let schema = ui::query::<ExperimentSchema>(app.settings.clone(), vec!["experiment-schema".into()]).await;
        let _ = app.update(Message::SchemaLoaded(schema));
        assert!(app.ops.iter().any(|o| o.arg == "ops/mnist"));
        assert!(app.knobs.is_some());
        assert!(matches!(&app.schema, Some(Ok(s)) if !s.main_memories.is_empty()), "{:?}", app.schema);
    }

    /// What the iced runtime would do with Compare's Run task: resolve the
    /// pending requests and hand the answers back.
    async fn resolve_compare(app: &mut App) {
        let outcomes = ui::resolve_all(app.settings.clone(), app.compare.pending_requests()).await;
        let generation = app.compare.generation();
        let _ = app.update(Message::Compare(compare::Msg::Resolved { launch: true, generation, outcomes }));
    }

    /// Run every job the app has started, feeding its events back in, until
    /// the queue is empty -- what the iced runtime does with `pump`'s tasks.
    async fn drain(app: &mut App) {
        while let Some((id, launch)) =
            app.jobs.iter().find(|j| j.status == JobStatus::Running).map(|j| (j.id, j.launch.clone().unwrap()))
        {
            let mut s = Box::pin(jobs::run(app.settings.clone(), launch));
            while let Some(e) = s.next().await {
                let _ = app.update(Message::Job(id, e));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn compare_tab_runs_isolated_and_hetero_jobs_and_shows_results() {
        let mut app = app();
        load_shared(&mut app).await;

        let _ = app.update(Message::Compare(compare::Msg::ToggleOp("ops/mymatmul".into(), true)));
        let _ = app.update(Message::Compare(compare::Msg::ToggleCore("ara", true)));
        let _ = app.update(Message::Compare(compare::Msg::Run));
        assert_eq!(app.tab, Tab::Jobs);
        drain(&mut app).await;

        let _ = app.update(Message::Compare(compare::Msg::Mode(compare::Mode::Hetero)));
        let _ = app.update(Message::Compare(compare::Msg::TogglePlacement("spatz".into(), true)));
        let _ = app.update(Message::Compare(compare::Msg::Run));
        assert_eq!(app.jobs.len(), 1, "SoC jobs wait for their requests to resolve");
        resolve_compare(&mut app).await;
        assert_eq!(app.jobs.len(), 3, "one isolated job, then mapped + pinned");
        let previews = app.compare.previews();
        assert!(previews.iter().all(|(_, r)| r.is_ok()), "{previews:?}");
        assert_eq!(previews[1].1.as_ref().unwrap().mapping.pin.as_deref(), Some("spatz"));
        drain(&mut app).await;

        for j in &app.jobs {
            assert_eq!(j.status, JobStatus::Finished(Some(0)), "{}: {:?}", j.title, j.log);
        }
        let loaded = app.compare.loaded();
        assert_eq!(loaded.len(), 3);
        let RunResult::Isolated(iso) = &loaded[0].result else { panic!("first should be run.py") };
        let cores: Vec<&str> = iso.results.iter().map(|c| c.core.as_str()).collect();
        assert_eq!(cores, ["cva6", "snitch", "spatz", "ara"]);
        assert!(iso.results.iter().all(|c| c.status == "ok"));
        let variants: Vec<String> = loaded[1..]
            .iter()
            .map(|l| match &l.result {
                RunResult::Hetero(h) => h.variant(),
                _ => panic!("expected hetero"),
            })
            .collect();
        assert_eq!(variants, ["cva6 host, mapped", "cva6 host, pinned spatz"]);
        for l in &loaded[1..] {
            let RunResult::Hetero(h) = &l.result else { unreachable!() };
            assert!(l.resolved.is_some(), "{}: no resolved sidecar", l.source);
            assert!(crate::ui::evidence::has_evidence(&h.mapping, &h.result.nodes), "{}", l.source);
            assert!(h.mapping.nodes.iter().all(|n| n.mapping_explanation.is_some()));
            assert!(h.result.nodes.iter().all(|n| n.implementation.is_some()));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn compare_tab_refuses_what_the_backend_cannot_resolve() {
        let mut app = app();
        load_shared(&mut app).await;
        for m in [
            compare::Msg::Mode(compare::Mode::Hetero),
            compare::Msg::ToggleOp("ops/mymatmul".into(), true),
            compare::Msg::Dram(crate::ui::common::Choice { value: "ddr42".into(), label: "ddr42".into() }),
            compare::Msg::Run,
        ] {
            let _ = app.update(Message::Compare(m));
        }
        resolve_compare(&mut app).await;
        assert!(app.jobs.is_empty(), "nothing queued");
        let previews = app.compare.previews();
        let err = previews[0].1.as_ref().unwrap_err();
        assert!(err.contains("unknown main memory"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn sweep_tab_runs_a_factorial_sweep_and_results_tab_analyses_it() {
        let mut app = app();
        load_shared(&mut app).await;
        let name = "e2e-factorial";
        let _ = std::fs::remove_dir_all(app.settings.workspace.join("work/gui/sweeps").join(name));

        for m in [
            sweep::Msg::Preset(&["ops/mnist"]),
            sweep::Msg::Name(name.into()),
            sweep::Msg::Images("2".into()),
            sweep::Msg::Mode(crate::space::SpaceMode::Factorial),
            sweep::Msg::Values("SPATZ_NB_LANES".into(), "2, 4".into()),
            sweep::Msg::Values("SPATZ_NB_CORE".into(), "5,9".into()),
            sweep::Msg::Run,
        ] {
            let _ = app.update(Message::Sweep(m));
        }
        assert_eq!(app.tab, Tab::Results);
        let out = format!("work/gui/sweeps/{name}");
        assert_eq!(app.results.current(), Some(out.as_str()));
        drain(&mut app).await;

        let job = app.jobs.last().unwrap();
        assert_eq!(job.status, JobStatus::Finished(Some(0)), "{:?}", job.log);
        // 2 x 2 factorial, one point of which is the baseline itself: the
        // driver runs it once.
        assert_eq!(job.progress, Some((4, 4)));
        let live = &app.results.live.as_ref().unwrap().1;
        assert!(!live.running);
        assert_eq!((live.done, live.total), (4, 4));
        assert_eq!(app.results.row_count(), 4);

        let report = crate::backend::capture(
            app.settings.clone(),
            crate::backend::Invocation::new("pipeline/sweep/report.py").arg(format!("{out}/sweep.jsonl")).arg("--json"),
        )
        .await
        .map(|s| serde_json::from_str(s.trim()).unwrap());
        let _ = app.update(Message::Results(results::Msg::Analysed(out.clone(), report)));
        let rep = app.results.report().expect("report loaded");
        assert!(rep.pareto.as_ref().is_some_and(|p| !p.front.is_empty()));
        // (2, 9) and (4, 5) each move one knob from the baseline, so they are
        // the sensitivity points; (2, 5) moves both and is not.
        let knobs: Vec<&str> = rep.sensitivity[0].knobs.iter().map(|k| k.knob.as_str()).collect();
        assert_eq!(knobs.len(), 2, "{knobs:?}");

        // Every cell has a manifest; the baseline's opens in Results.
        let rows = crate::model::parse_jsonl(
            &std::fs::read_to_string(app.settings.workspace.join(&out).join("sweep.jsonl")).unwrap(),
        );
        assert!(rows.iter().all(|r| r.manifest.is_some()), "{rows:?}");
        let base = rows.iter().find(|r| r.design_slug == "baseline").unwrap();
        let _ = app.update(Message::Results(results::Msg::Inspect(base.manifest.clone())));
        let manifest = app.results.inspected().unwrap().as_ref().expect("manifest parses").clone();
        assert_eq!(Some(&manifest.run_fingerprint), base.run_fingerprint.as_ref());

        // The Sweep tab's baseline preview is the experiment the driver resolved.
        let requests = app.sweep.baseline_requests(&app.ctx());
        let answers = ui::resolve_all(app.settings.clone(), requests).await;
        let spec = Box::new(app.sweep.spec.clone());
        let models = app.sweep.spec.models.clone();
        let _ = app.update(Message::Sweep(sweep::Msg::Resolved(spec, models.into_iter().zip(answers).collect())));
        let (_, preview) = &app.sweep.resolved()[0];
        assert_eq!(
            preview.as_ref().unwrap().resolved_fingerprint,
            manifest.resolved.experiment.resolved_fingerprint,
            "the GUI's request must be the driver's"
        );
    }
}
