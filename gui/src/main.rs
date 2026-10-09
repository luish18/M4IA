//! hetero-gui: a desktop front-end for the hetero-sim pipeline.
//!
//! It drives the same Python CLIs a person would -- pipeline/make_op.py,
//! run.py, run_hetero.py, sweep/run.py, sweep/report.py and gui_query.py --
//! natively on a Linux checkout or inside the hetero-sim Docker image, and
//! renders what they write.

// No console window behind the GUI on Windows release builds.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod backend;
#[cfg(test)]
mod e2e;
mod experiment;
mod jobs;
mod model;
mod settings;
mod space;
mod ui;
mod widgets;

use std::path::PathBuf;

const USAGE: &str = "\
usage: hetero-gui [options]

  --config FILE        read and save settings here instead of the per-user config
  --open-result FILE   show a run.py / run_hetero.py result JSON in Compare (repeatable)
  --open-sweep DIR     show a sweep directory (holding sweep.jsonl) in Sweep results
  --tab NAME           start on: ops, compare, sweep, results, jobs, settings
  --tour DIR           screenshot every tab into DIR and exit (for docs and checks)
";

#[derive(Debug, Clone, Default)]
pub struct Cli {
    pub open_results: Vec<PathBuf>,
    pub open_sweep: Option<PathBuf>,
    pub tab: Option<ui::Tab>,
    pub tour: Option<PathBuf>,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--config" => {
                let _ = settings::CONFIG_OVERRIDE.set(PathBuf::from(val("--config")?));
            }
            "--open-result" => cli.open_results.push(val("--open-result")?.into()),
            "--open-sweep" => cli.open_sweep = Some(val("--open-sweep")?.into()),
            "--tab" => {
                let v = val("--tab")?;
                cli.tab = Some(ui::Tab::parse(&v).ok_or(format!("unknown tab {v:?}"))?);
            }
            "--tour" => cli.tour = Some(val("--tour")?.into()),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(cli)
}

fn main() -> iced::Result {
    let cli = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("hetero-gui: {e}");
            }
            eprint!("{USAGE}");
            std::process::exit(if e.is_empty() { 0 } else { 2 });
        }
    };
    iced::application(move || app::App::new(cli.clone()), app::App::update, app::App::view)
        .title(app::App::title)
        .theme(app::App::theme)
        .subscription(app::App::subscription)
        .window_size((1280.0, 860.0))
        .run()
}
