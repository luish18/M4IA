//! What the backend resolved a request to, shown before anything runs.

use super::common::*;
use crate::experiment::{ResolvedExperiment, short};
use iced::widget::{column, row, text};
use iced::{Element, Length};

/// The resolved experiment, one fact per line. `copy` is the message that
/// puts the backend's JSON on the clipboard.
pub fn summary<'a, M: Clone + 'a>(r: &'a ResolvedExperiment, copy: Option<M>) -> Element<'a, M> {
    let w = &r.workload;
    let ops = w
        .op_counts
        .iter()
        .take(4)
        .map(|(k, v)| format!("{k}×{}", v.as_u64().unwrap_or_default()))
        .collect::<Vec<_>>()
        .join(" ");
    let clusters = r
        .resources
        .clusters
        .iter()
        .map(|(name, c)| {
            let lanes = c
                .spatz_vector
                .as_ref()
                .map(|v| format!(", {} useful vector lanes", v.useful_vector_lanes))
                .unwrap_or_default();
            format!("{name} {} cores ({} compute){lanes}", c.modeled_cores, c.compute_cores)
        })
        .collect::<Vec<_>>()
        .join("; ");
    let mapping = match &r.mapping.pin {
        Some(p) => format!("{}, pin {p} ({})", r.mapping.strategy, r.mapping.pin_semantics.replace('_', " ")),
        None => r.mapping.strategy.clone(),
    };
    let e = &r.execution;
    let mut execution = vec![];
    if let Some(fe) = &e.frontend {
        execution.push(format!("front-end on {fe}"));
    }
    if e.serial {
        execution.push("serial".into());
    }
    if e.power {
        execution.push("energy".into());
    }
    let memory = if r.memory.overrides.is_empty() {
        format!("{} ({})  {}", r.memory.label, r.memory.kind, short(&r.memory.fingerprint))
    } else {
        format!("{} ({}, overridden)  {}", r.memory.label, r.memory.kind, short(&r.memory.fingerprint))
    };
    column![
        fact(
            "Workload",
            format!(
                "{}{} · {} nodes: {ops}  {}",
                w.path,
                w.application.as_ref().map(|a| format!(" ({a})")).unwrap_or_default(),
                w.node_count,
                short(&w.workload_fingerprint)
            )
        ),
        fact("Host", format!("{} → {} {}", r.host_profile, r.simulator.kind, r.simulator.target)),
        fact(
            "Design",
            format!("{} · build {}  {}", r.hardware.design_slug, r.hardware.build_key, short(&r.hardware.fingerprint))
        ),
        fact("Memory", memory),
        fact("Resources", format!("{} modelled cores: {clusters}", r.resources.platform.total_modeled_cores)),
        fact("Mapping", mapping),
        fact("Execution", if execution.is_empty() { "defaults".into() } else { execution.join(", ") }),
        row![text("Resolved").size(12).width(Length::Fixed(90.0)), mono(r.resolved_fingerprint.as_str())]
            .push(copy.map(|m| small_button("Copy JSON", Some(m))))
            .spacing(8)
            .align_y(iced::Center),
    ]
    .spacing(2)
    .into()
}

fn fact<'a, M: 'a>(label: &'a str, value: String) -> Element<'a, M> {
    row![text(label).size(12).width(Length::Fixed(90.0)), mono(value)].spacing(8).into()
}
