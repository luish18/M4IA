//! The Experimental Foundation's query surface, as Rust types.
//!
//! `gui_query.py experiment-schema` says what can be chosen (memories, host
//! profiles, engines, knob metadata) and `resolve-experiment` turns one
//! `ExperimentRequest` into the canonical experiment the runners will build.
//! The GUI only reads these: the Python resolver stays the one authority on
//! what a request means. Fields read here are pinned by
//! pipeline/tests/test_gui_query.py.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

// --- experiment-schema -------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ExperimentSchema {
    pub parameters: Vec<ParameterRow>,
    pub engines: Vec<EngineRow>,
    pub host_profiles: Vec<HostRow>,
    pub mapping_strategies: Vec<StrategyRow>,
    pub main_memories: Vec<MemoryRow>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ParameterRow {
    pub name: String,
    pub label: String,
    pub group: String,
    pub unit: Option<String>,
    pub description: String,
    pub default: Option<i64>,
    pub build_time: bool,
    pub screening_values: Vec<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EngineRow {
    pub name: String,
    pub label: String,
    /// "host" or "cluster".
    pub kind: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct HostRow {
    pub name: String,
    pub label: String,
    pub vector: bool,
    /// The GVSoC target this profile runs on.
    pub target: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StrategyRow {
    pub id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MemoryRow {
    /// The `--dram` / `dram` spelling.
    pub name: String,
    pub label: String,
    pub model: String,
    pub supports_overrides: bool,
}

impl ExperimentSchema {
    pub fn memory(&self, name: &str) -> Option<&MemoryRow> {
        self.main_memories.iter().find(|m| m.name == name)
    }

    pub fn param(&self, name: &str) -> Option<&ParameterRow> {
        self.parameters.iter().find(|p| p.name == name)
    }
}

// --- resolve-experiment ------------------------------------------------------

/// What `ExperimentRequest.from_dict` takes. `platform` and
/// `mapping_strategy` are left out so the backend applies its own defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ExperimentRequest {
    pub workload: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub design_overrides: BTreeMap<String, i64>,
    pub host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontend: Option<String>,
    pub serial: bool,
    pub power: bool,
    /// Always explicit; omitted and "fixed" resolve identically anyway.
    pub dram: String,
}

impl ExperimentRequest {
    pub fn new(workload: &str, host: &str, dram: &str) -> Self {
        ExperimentRequest { workload: workload.into(), host: host.into(), dram: dram.into(), ..Default::default() }
    }

    /// The execution controls as the runners apply them. Only KWS runs a
    /// front-end, so only KWS carries one (Snitch unless chosen) or `serial`
    /// -- the same as `sweep/run.py::resolve_cell_experiment`, so a preview
    /// resolves to the fingerprint the driver records.
    pub fn execution(mut self, app: Option<&str>, frontend: Option<&str>, serial: bool, power: bool) -> Self {
        let kws = app == Some("kws");
        self.frontend = kws.then(|| frontend.unwrap_or("snitch").to_string());
        self.serial = kws && serial;
        self.power = power;
        self
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ResolvedExperiment {
    pub resolved_fingerprint: String,
    pub hardware: Hardware,
    pub memory: ResolvedMemory,
    pub resources: Resources,
    pub host_profile: String,
    pub simulator: Simulator,
    pub workload: Workload,
    pub mapping: ResolvedMapping,
    pub execution: Execution,
    /// The backend's JSON as it came, for copying out.
    #[serde(skip)]
    pub raw: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Hardware {
    pub design: Map<String, Value>,
    pub design_slug: String,
    pub build_key: String,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ResolvedMemory {
    pub kind: String,
    pub label: String,
    pub model: String,
    pub fingerprint: String,
    pub overrides: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Resources {
    pub platform: PlatformResources,
    pub clusters: BTreeMap<String, ClusterResources>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct PlatformResources {
    pub total_modeled_cores: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ClusterResources {
    pub modeled_cores: u64,
    pub compute_cores: u64,
    pub spatz_vector: Option<SpatzVector>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SpatzVector {
    pub modeled_vector_lanes: u64,
    pub useful_vector_lanes: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Simulator {
    pub kind: String,
    pub target: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Workload {
    pub path: String,
    pub application: Option<String>,
    pub workload_fingerprint: String,
    pub node_count: u64,
    /// Op type -> count, in the backend's order.
    pub op_counts: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ResolvedMapping {
    pub strategy: String,
    pub pin: Option<String>,
    pub pin_semantics: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Execution {
    pub frontend: Option<String>,
    pub serial: bool,
    pub power: bool,
}

impl ResolvedExperiment {
    pub fn parse(text: &str) -> Result<ResolvedExperiment, String> {
        let mut r: ResolvedExperiment =
            serde_json::from_str(text).map_err(|e| format!("unexpected answer from resolve-experiment: {e}"))?;
        if r.resolved_fingerprint.is_empty() {
            return Err("resolve-experiment answered without a resolved_fingerprint".into());
        }
        r.raw = text.to_string();
        Ok(r)
    }
}

/// True when a bridge error only says this gui_query.py predates the
/// Foundation (an older Docker image): argparse refusing the subcommand.
pub fn bridge_too_old(error: &str) -> bool {
    error.contains("invalid choice")
}

/// `sha256:0123456789abcdef…` -> `sha256:0123456789ab`, for tables.
pub fn short(fp: &str) -> &str {
    let end = fp.find(':').map_or(0, |i| i + 1) + 12;
    fp.get(..end).unwrap_or(fp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_bridge_schema() {
        let s: ExperimentSchema =
            serde_json::from_str(include_str!("../tests/fixtures/experiment-schema.json")).unwrap();
        let mems: Vec<&str> = s.main_memories.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(mems, ["fixed", "lpddr4", "lpddr4x", "lpddr5", "hyperram"]);
        assert!(!s.memory("fixed").unwrap().supports_overrides);
        assert_eq!(s.host_profiles.iter().find(|h| h.name == "ara").unwrap().target, "hetero_ara");
        let engines: Vec<&str> = s.engines.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(engines, ["cva6", "snitch", "spatz"]);
        let lanes = s.param("SPATZ_NB_LANES").unwrap();
        assert_eq!((lanes.label.as_str(), lanes.unit.as_deref()), ("Spatz lanes", Some("lanes")));
        assert!(s.param("HOST_VLEN").unwrap().build_time);
        assert_eq!(s.mapping_strategies[0].id, "measured_rate_greedy");
    }

    #[test]
    fn parses_resolved_experiments() {
        let r = ResolvedExperiment::parse(include_str!("../tests/fixtures/resolved-kws-lpddr5.json")).unwrap();
        assert!(r.resolved_fingerprint.starts_with("sha256:"));
        assert_eq!((r.memory.kind.as_str(), r.memory.label.as_str()), ("lpddr5", "LPDDR5 main memory"));
        assert_eq!((r.host_profile.as_str(), r.simulator.target.as_str()), ("ara", "hetero_ara"));
        assert_eq!(r.hardware.design_slug, "baseline");
        assert_eq!(r.workload.application.as_deref(), Some("kws"));
        assert_eq!(r.mapping.pin.as_deref(), Some("spatz"));
        assert_eq!(r.execution.frontend.as_deref(), Some("snitch"));
        assert_eq!(r.resources.clusters["spatz"].spatz_vector.as_ref().unwrap().useful_vector_lanes, 32);
        assert_eq!(r.resources.platform.total_modeled_cores, 19);
        assert!(r.raw.contains("resolved_fingerprint"));

        let m = ResolvedExperiment::parse(include_str!("../tests/fixtures/resolved-mnist.json")).unwrap();
        assert_eq!(m.memory.kind, "fixed");
        assert_eq!(m.execution.frontend, None);
        assert!(ResolvedExperiment::parse(r#"{"error": "x"}"#).is_err());
    }

    #[test]
    fn requests_carry_kws_controls_only_for_kws() {
        let kws = ExperimentRequest::new("ops/kws", "cva6", "fixed").execution(Some("kws"), None, true, false);
        assert_eq!((kws.frontend.as_deref(), kws.serial), (Some("snitch"), true));
        let mnist =
            ExperimentRequest::new("ops/mnist", "cva6", "fixed").execution(Some("mnist"), Some("spatz"), true, true);
        assert_eq!((mnist.frontend.as_deref(), mnist.serial, mnist.power), (None, false, true));

        let json = serde_json::to_value(&mnist).unwrap();
        let keys: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["workload", "host", "serial", "power", "dram"], "unset optionals are omitted");
        let mut pinned = ExperimentRequest::new("ops/mnist", "ara", "lpddr5");
        pinned.pin = Some("spatz".into());
        pinned.design_overrides.insert("SPATZ_NB_LANES".into(), 8);
        let json = serde_json::to_value(&pinned).unwrap();
        assert_eq!(json["pin"], "spatz");
        assert_eq!(json["design_overrides"]["SPATZ_NB_LANES"], 8);
    }

    #[test]
    fn shortens_fingerprints_and_spots_an_old_bridge() {
        assert_eq!(short("sha256:0123456789abcdef0123"), "sha256:0123456789ab");
        assert_eq!(short("abc"), "abc");
        assert!(bridge_too_old("gui_query.py: error: argument cmd: invalid choice: 'resolve-experiment'"));
        assert!(!bridge_too_old("ValueError: unknown main memory 'ddr42'"));
    }
}
