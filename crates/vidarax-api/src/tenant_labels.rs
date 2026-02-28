use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct TenantLabelMaps {
    default: TenantLabelMap,
    tenants: HashMap<String, TenantLabelMap>,
}

#[derive(Debug, Clone, Default)]
struct TenantLabelMap {
    events: HashMap<String, String>,
    objects: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct LabelMapResult {
    pub label: String,
    pub used_fallback: bool,
}

#[derive(Debug, Deserialize)]
struct TenantLabelMapsFile {
    #[serde(default)]
    default: TenantLabelMapFile,
    #[serde(default)]
    tenants: HashMap<String, TenantLabelMapFile>,
}

#[derive(Debug, Deserialize, Default)]
struct TenantLabelMapFile {
    #[serde(default)]
    events: HashMap<String, String>,
    #[serde(default)]
    objects: HashMap<String, String>,
}

impl TenantLabelMaps {
    pub fn from_env() -> Result<Self, String> {
        let Some(path) = std::env::var("VIDARAX_TENANT_LABEL_MAPS_PATH").ok() else {
            return Ok(Self::default());
        };
        if path.trim().is_empty() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|err| format!("failed to read tenant label maps file '{path}': {err}"))?;
        let parsed: TenantLabelMapsFile = serde_json::from_str(&raw)
            .map_err(|err| format!("invalid tenant label maps json '{path}': {err}"))?;
        Ok(Self::from_file(parsed))
    }

    fn from_file(file: TenantLabelMapsFile) -> Self {
        let default = TenantLabelMap {
            events: file.default.events,
            objects: file.default.objects,
        };
        let tenants = file
            .tenants
            .into_iter()
            .map(|(tenant, map)| {
                (
                    tenant,
                    TenantLabelMap {
                        events: map.events,
                        objects: map.objects,
                    },
                )
            })
            .collect();
        Self { default, tenants }
    }

    pub fn map_event(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.map_label(tenant_id, label, |map| map.events.get(label))
    }

    pub fn map_object(&self, tenant_id: Option<&str>, label: &str) -> LabelMapResult {
        self.map_label(tenant_id, label, |map| map.objects.get(label))
    }

    fn map_label(
        &self,
        tenant_id: Option<&str>,
        label: &str,
        selector: impl Fn(&TenantLabelMap) -> Option<&String>,
    ) -> LabelMapResult {
        let Some(tenant_id) = tenant_id else {
            return selector(&self.default)
                .map(|mapped| LabelMapResult {
                    label: mapped.clone(),
                    used_fallback: false,
                })
                .unwrap_or_else(|| LabelMapResult {
                    label: label.to_string(),
                    used_fallback: false,
                });
        };

        if let Some(tenant) = self.tenants.get(tenant_id) {
            if let Some(mapped) = selector(tenant) {
                return LabelMapResult {
                    label: mapped.clone(),
                    used_fallback: false,
                };
            }
        }

        if let Some(mapped) = selector(&self.default) {
            return LabelMapResult {
                label: mapped.clone(),
                used_fallback: true,
            };
        }

        LabelMapResult {
            label: label.to_string(),
            used_fallback: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TenantLabelMapFile, TenantLabelMaps, TenantLabelMapsFile};
    use std::collections::HashMap;

    #[test]
    fn uses_tenant_override_when_available() {
        let mut tenants = HashMap::new();
        tenants.insert(
            "t1".to_string(),
            TenantLabelMapFile {
                events: HashMap::from([("scene_cut".to_string(), "scene.transition".to_string())]),
                objects: HashMap::new(),
            },
        );
        let maps = TenantLabelMaps::from_file(TenantLabelMapsFile {
            default: TenantLabelMapFile::default(),
            tenants,
        });
        let out = maps.map_event(Some("t1"), "scene_cut");
        assert_eq!(out.label, "scene.transition");
        assert!(!out.used_fallback);
    }

    #[test]
    fn falls_back_to_default_map() {
        let maps = TenantLabelMaps::from_file(TenantLabelMapsFile {
            default: TenantLabelMapFile {
                events: HashMap::from([("scene_cut".to_string(), "scene.transition".to_string())]),
                objects: HashMap::new(),
            },
            tenants: HashMap::new(),
        });
        let out = maps.map_event(Some("unknown"), "scene_cut");
        assert_eq!(out.label, "scene.transition");
        assert!(out.used_fallback);
    }
}
