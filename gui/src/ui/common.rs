//! Small view helpers shared by the tabs.

use crate::experiment::ExperimentSchema;
use iced::widget::{Column, button, column, container, row, text};
use iced::{Color, Element, Font, Length, Theme};

/// Status colours legible on both themes: the palette's own success and
/// danger are tuned for button backgrounds, too dark for text on dark.
#[derive(Clone, Copy)]
pub enum Tone {
    Good,
    Bad,
    Warn,
    Info,
    Muted,
}

pub fn tone(t: &Theme, k: Tone) -> Color {
    let dark = t.extended_palette().is_dark;
    let pick = |d: [f32; 3], l: [f32; 3]| {
        let c = if dark { d } else { l };
        Color::from_rgb(c[0], c[1], c[2])
    };
    match k {
        Tone::Good => pick([0.40, 0.82, 0.55], [0.08, 0.50, 0.25]),
        Tone::Bad => pick([0.96, 0.45, 0.45], [0.75, 0.12, 0.12]),
        Tone::Warn => pick([0.95, 0.75, 0.35], [0.62, 0.42, 0.02]),
        Tone::Info => pick([0.55, 0.65, 1.0], [0.20, 0.30, 0.80]),
        Tone::Muted => {
            let mut c = t.extended_palette().background.base.text;
            c.a = 0.62;
            c
        }
    }
}

fn toned<'a>(s: impl text::IntoFragment<'a>, k: Tone) -> text::Text<'a> {
    text(s).style(move |t: &Theme| text::Style { color: Some(tone(t, k)) })
}

pub fn section<'a, M: 'a>(title: &'a str, body: impl Into<Element<'a, M>>) -> Element<'a, M> {
    container(column![text(title).size(16), body.into()].spacing(8))
        .padding(12)
        .width(Length::Fill)
        .style(container::bordered_box)
        .into()
}

pub fn mono<'a>(s: impl text::IntoFragment<'a>) -> text::Text<'a> {
    text(s).font(Font::MONOSPACE).size(12)
}

pub fn muted<'a>(s: impl text::IntoFragment<'a>) -> text::Text<'a> {
    toned(s, Tone::Muted).size(12)
}

pub fn error<'a>(s: impl text::IntoFragment<'a>) -> text::Text<'a> {
    toned(s, Tone::Bad).size(13)
}

pub fn warn<'a>(s: impl text::IntoFragment<'a>) -> text::Text<'a> {
    toned(s, Tone::Warn).size(13)
}

pub fn ok<'a>(s: impl text::IntoFragment<'a>) -> text::Text<'a> {
    toned(s, Tone::Good).size(13)
}

/// A status word coloured by what it means.
pub fn status<'a>(s: &str) -> text::Text<'a> {
    let k = match s {
        "ok" | "done" => Tone::Good,
        "running" | "queued" => Tone::Info,
        "invalid" | "cancelled" | "killed" => Tone::Warn,
        _ => Tone::Bad,
    };
    toned(s.to_string(), k).size(13)
}

pub fn labelled<'a, M: 'a>(label: &'a str, w: impl Into<Element<'a, M>>) -> Element<'a, M> {
    row![text(label).size(13).width(Length::Fixed(120.0)), w.into()].spacing(8).align_y(iced::Center).into()
}

pub fn small_button<'a, M: Clone + 'a>(label: &'a str, on: Option<M>) -> button::Button<'a, M> {
    button(text(label).size(13)).padding([4, 10]).on_press_maybe(on)
}

/// A column of lines, for a list of reasons or messages.
pub fn lines<'a, M: 'a>(items: impl IntoIterator<Item = Element<'a, M>>) -> Column<'a, M> {
    Column::with_children(items).spacing(2)
}

/// One option of a discovered choice: the value the pipeline takes and the
/// label experiment-schema gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub value: String,
    pub label: String,
}

impl std::fmt::Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label)
    }
}

/// The main memories experiment-schema lists. Without it (still loading, or
/// an image that predates it) only the runners' default is offered.
pub fn memory_choices(schema: Option<&ExperimentSchema>) -> Vec<Choice> {
    match schema {
        Some(s) if !s.main_memories.is_empty() => {
            s.main_memories.iter().map(|m| Choice { value: m.name.clone(), label: m.label.clone() }).collect()
        }
        _ => vec![Choice { value: "fixed".into(), label: "Fixed main memory".into() }],
    }
}

pub fn host_choices(schema: Option<&ExperimentSchema>) -> Vec<Choice> {
    match schema {
        Some(s) if !s.host_profiles.is_empty() => {
            s.host_profiles.iter().map(|h| Choice { value: h.name.clone(), label: h.label.clone() }).collect()
        }
        _ => vec![Choice { value: "cva6".into(), label: "Scalar CVA6 host".into() }],
    }
}

/// The engines a node can be pinned to; none without the schema.
pub fn engine_choices(schema: Option<&ExperimentSchema>) -> Vec<Choice> {
    schema
        .map(|s| s.engines.iter().map(|e| Choice { value: e.name.clone(), label: e.label.clone() }).collect())
        .unwrap_or_default()
}

/// `value` as one of `choices`, or as itself if it is not (yet) listed, so a
/// saved selection is never silently swapped for another.
pub fn chosen(choices: &[Choice], value: &str) -> Choice {
    choices
        .iter()
        .find(|c| c.value == value)
        .cloned()
        .unwrap_or_else(|| Choice { value: value.to_string(), label: value.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_come_from_the_schema_with_a_minimal_fallback() {
        let s: ExperimentSchema =
            serde_json::from_str(include_str!("../../tests/fixtures/experiment-schema.json")).unwrap();
        let mems = memory_choices(Some(&s));
        assert_eq!(mems.len(), 5);
        assert_eq!(chosen(&mems, "lpddr5").label, "LPDDR5 main memory");
        assert_eq!(host_choices(Some(&s)).iter().map(|c| c.value.as_str()).collect::<Vec<_>>(), ["ara", "cva6"]);
        assert_eq!(engine_choices(Some(&s)).len(), 3);

        assert_eq!(memory_choices(None).iter().map(|c| c.value.as_str()).collect::<Vec<_>>(), ["fixed"]);
        assert_eq!(host_choices(None)[0].value, "cva6");
        assert!(engine_choices(None).is_empty());
        assert_eq!(chosen(&memory_choices(None), "hyperram").label, "hyperram");
    }
}
