//! Per-node evidence: what code generation placed (GENERATED), what the
//! completed node proves ran (ACTUAL) and how long it took (MEASURED), side by
//! side but never merged into one claim.

use super::common::*;
use crate::model::{Mapping, NodeMap, RuntimeNode, fmt_cycles};
use iced::widget::{button, column, row, text};
use iced::{Element, Length};

/// One table line: a generated node, the completed node with its index, or
/// both. Joined on the generated `index` and the runtime `node`, the same key
/// `annotate_completed_nodes` uses.
pub struct NodeLine<'a> {
    pub index: Option<i64>,
    pub generated: Option<&'a NodeMap>,
    pub actual: Option<&'a RuntimeNode>,
    pub measured: Option<&'a RuntimeNode>,
}

/// `actual` carries the implementations and `measured` the cycles. For a
/// run_hetero result both are `result.nodes`; a manifest keeps them apart.
pub fn join<'a>(generated: &'a Mapping, actual: &'a [RuntimeNode], measured: &'a [RuntimeNode]) -> Vec<NodeLine<'a>> {
    let mut lines: Vec<NodeLine> = generated
        .nodes
        .iter()
        .map(|g| NodeLine {
            index: g.index,
            generated: Some(g),
            actual: g.index.and_then(|i| actual.iter().find(|a| a.node == i)),
            measured: g.index.and_then(|i| measured.iter().find(|m| m.node == i)),
        })
        .collect();
    // Completed nodes code generation does not account for still show.
    for a in actual {
        if !lines.iter().any(|l| l.index == Some(a.node)) {
            lines.push(NodeLine {
                index: Some(a.node),
                generated: None,
                actual: Some(a),
                measured: measured.iter().find(|m| m.node == a.node),
            });
        }
    }
    lines
}

/// Whether there is any Foundation evidence to show, as opposed to a result
/// written before it existed.
pub fn has_evidence(generated: &Mapping, actual: &[RuntimeNode]) -> bool {
    generated.nodes.iter().any(|n| n.index.is_some() || n.mapping_explanation.is_some())
        || actual.iter().any(|a| a.implementation.is_some())
}

fn rule(g: &NodeMap) -> String {
    match &g.mapping_explanation {
        None => "-".into(),
        Some(x) if x.status.as_deref() == Some("unavailable") => {
            format!("unavailable: {}", x.reason.as_deref().unwrap_or("?").replace('_', " "))
        }
        Some(x) => x.selection_rule.as_deref().unwrap_or("-").replace('_', " "),
    }
}

fn implementation(a: Option<&RuntimeNode>) -> (String, String) {
    let Some(a) = a else { return ("not completed".into(), String::new()) };
    let Some(imp) = &a.implementation else { return ("-".into(), String::new()) };
    let id = match imp.unknown_reason() {
        Some(why) => format!("unknown: {why}"),
        None => imp.implementation_id.clone().unwrap_or_default(),
    };
    let fallback = match imp.fallback_used {
        Some(true) => format!("yes: {}", imp.fallback_reason.as_deref().unwrap_or("?").replace('_', " ")),
        Some(false) => "no".into(),
        None => "-".into(),
    };
    (id, fallback)
}

/// The node table. `expanded` is the index whose per-engine explanation is
/// open; `on_expand` toggles one.
pub fn table<'a, M: Clone + 'a>(
    lines: &[NodeLine<'a>],
    expanded: Option<i64>,
    on_expand: impl Fn(Option<i64>) -> M,
) -> Element<'a, M> {
    let head = row![
        muted("#").width(Length::Fixed(28.0)),
        muted("node").width(Length::Fixed(150.0)),
        muted("op").width(Length::Fixed(80.0)),
        muted("generated engine").width(Length::Fixed(110.0)),
        muted("selection").width(Length::Fixed(215.0)),
        muted("est. cost").width(Length::Fixed(70.0)),
        muted("actual implementation").width(Length::Fill),
        muted("fallback").width(Length::Fixed(90.0)),
        muted("measured cyc").width(Length::Fixed(80.0)),
        text("").width(Length::Fixed(50.0)),
    ]
    .spacing(8);
    let mut c = column![head].spacing(3);
    for l in lines {
        let (imp, fallback) = implementation(l.actual);
        let explained = l.generated.and_then(|g| g.mapping_explanation.as_ref()).is_some_and(|x| !x.engines.is_empty());
        let open = explained && l.index.is_some() && l.index == expanded;
        let toggle = explained.then(|| on_expand(if open { None } else { l.index }));
        c = c.push(
            row![
                mono(l.index.map(|i| i.to_string()).unwrap_or_else(|| "-".into())).width(Length::Fixed(28.0)),
                mono(l.generated.map(|g| g.node.clone()).unwrap_or_else(|| "-".into())).width(Length::Fixed(150.0)),
                mono(l.generated.map(|g| g.op.clone()).or_else(|| l.actual.map(|a| a.op.clone())).unwrap_or_default())
                    .width(Length::Fixed(80.0)),
                mono(l.generated.map(|g| g.engine.clone()).unwrap_or_else(|| "-".into())).width(Length::Fixed(110.0)),
                mono(l.generated.map(rule).unwrap_or_default()).width(Length::Fixed(215.0)),
                mono(
                    l.generated
                        .and_then(|g| g.mapping_explanation.as_ref())
                        .and_then(|x| x.selected_estimated_cost_cycles)
                        .map(|c| fmt_cycles(c.round() as u64))
                        .unwrap_or_else(|| "-".into())
                )
                .width(Length::Fixed(70.0)),
                mono(imp).width(Length::Fill),
                mono(fallback).width(Length::Fixed(90.0)),
                mono(l.measured.and_then(|m| m.cycles).map(fmt_cycles).unwrap_or_else(|| "-".into()))
                    .width(Length::Fixed(80.0)),
                button(text(if open { "hide" } else { "why" }).size(12))
                    .padding([1, 6])
                    .style(button::text)
                    .on_press_maybe(toggle)
                    .width(Length::Fixed(50.0)),
            ]
            .spacing(8)
            .align_y(iced::Center),
        );
        if open && let Some(g) = l.generated {
            c = c.push(explanation(g));
        }
    }
    c.into()
}

/// The mapper's per-engine view of one node.
fn explanation<'a, M: 'a>(g: &NodeMap) -> Element<'a, M> {
    let mut c = column![].spacing(1).padding(iced::Padding { left: 36.0, ..Default::default() });
    if let Some(args) = &g.kernel_arguments {
        let a = args.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
        c = c.push(muted(format!("emitted kernel arguments: {a}")));
    }
    let Some(x) = &g.mapping_explanation else { return c.into() };
    if let Some(p) = &x.requested_pin {
        c = c.push(muted(format!("requested pin {p}: a compatible-node preference, not a whole-graph placement")));
    }
    for e in &x.engines {
        let what = if !e.compatible {
            format!("incompatible ({})", e.exclusion_reason.as_deref().unwrap_or("?").replace('_', " "))
        } else {
            let cost = match (e.estimated_cycles(), e.cost_unavailable_reason()) {
                (Some(c), _) => format!("est. {} cycles", fmt_cycles(c.round() as u64)),
                (None, Some(why)) => format!("no safe cost ({})", why.replace('_', " ")),
                (None, None) => "cost not evaluated".into(),
            };
            let pin = match e.pin_influence.as_deref() {
                Some("not_requested") | None => String::new(),
                Some(p) => format!(", {}", p.replace('_', " ")),
            };
            let used = if e.cost_used_for_selection { ", chosen on cost" } else { "" };
            format!("compatible, {cost}{pin}{used}")
        };
        c = c.push(mono(format!("{:<7} {what}", e.engine)));
    }
    c.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{RunManifest, RunResult};

    #[test]
    fn joins_generated_and_completed_nodes_by_index() {
        let RunResult::Hetero(r) = RunResult::parse(include_str!("../../tests/fixtures/hetero-evidence.json")).unwrap()
        else {
            panic!()
        };
        assert!(has_evidence(&r.mapping, &r.result.nodes));
        let lines = join(&r.mapping, &r.result.nodes, &r.result.nodes);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].measured.unwrap().cycles, Some(6694));
        let (imp, fallback) = implementation(lines[0].actual);
        assert_eq!((imp.as_str(), fallback.as_str()), ("spatz.fp32.matmul_gemm.rvv_tuned", "no"));

        // A manifest keeps actual and measured apart; the join still lines them up.
        let m = RunManifest::parse(include_str!("../../tests/fixtures/manifest.json")).unwrap();
        let lines = join(&m.generated, &m.actual.nodes, &m.measured.nodes);
        assert!(lines[0].actual.unwrap().implementation.is_some());
        assert_eq!(lines[0].measured.unwrap().cycles, Some(6694));

        // A pre-Foundation result: no index, nothing joined, nothing claimed.
        let RunResult::Hetero(old) = RunResult::parse(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../results/mnist-hetero.json")).unwrap(),
        )
        .unwrap() else {
            panic!()
        };
        assert!(!has_evidence(&old.mapping, &old.result.nodes));
        let lines = join(&old.mapping, &old.result.nodes, &old.result.nodes);
        assert!(lines.iter().all(|l| l.actual.is_none() || l.generated.is_none()));
    }
}
