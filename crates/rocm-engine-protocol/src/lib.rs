use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

pub const ENGINE_PROTOCOL_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineMethod {
    Detect,
    Install,
    Capabilities,
    ResolveModel,
    Launch,
    Healthcheck,
    Endpoint,
    Stop,
    Logs,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DevicePolicy {
    GpuRequired,
    GpuPreferred,
    CpuOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineRequestEnvelope {
    pub method: EngineMethod,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineErrorDetail {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResponseEnvelope {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineErrorDetail>,
}

impl EngineResponseEnvelope {
    pub fn success<T>(data: T) -> Self
    where
        T: Serialize,
    {
        Self {
            ok: true,
            data: Some(serde_json::to_value(data).expect("serializable response")),
            error: None,
        }
    }

    pub fn failure(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(EngineErrorDetail {
                code: code.into(),
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectRequest {
    pub runtime_id: Option<String>,
    pub device_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRequest {
    pub runtime_id: String,
    pub python_version: Option<String>,
    #[serde(default)]
    pub reinstall: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveModelRequest {
    pub model_ref: String,
    pub runtime_id: Option<String>,
    pub device_policy: Option<DevicePolicy>,
    pub recipe_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchRequest {
    pub service_id: String,
    pub env_id: Option<String>,
    pub runtime_id: Option<String>,
    pub model_ref: String,
    pub host: String,
    pub port: u16,
    pub device_policy: Option<DevicePolicy>,
    pub endpoint_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthcheckRequest {
    pub service_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointRequest {
    pub service_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopRequest {
    pub service_id: String,
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsRequest {
    pub service_id: String,
    pub tail_lines: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineDeviceAvailability {
    pub kind: String,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub cpu: bool,
    pub rocm_gpu: bool,
    pub multi_gpu: bool,
    pub openai_compatible: bool,
    pub tool_calling: bool,
    pub quantized_models: String,
    pub distributed_serving: bool,
    pub reasoning_parser: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectResponse {
    pub installed: bool,
    pub env_id: Option<String>,
    pub python_version: Option<String>,
    pub torch_version: Option<String>,
    pub transformers_version: Option<String>,
    pub available_devices: Vec<EngineDeviceAvailability>,
    pub capabilities: EngineCapabilities,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResponse {
    pub env_id: String,
    pub env_path: String,
    pub python_executable: String,
    pub installed_packages: Vec<String>,
    pub capabilities: EngineCapabilities,
    pub lock_hash: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveModelResponse {
    pub canonical_model_id: String,
    pub task: String,
    pub source: String,
    pub revision: String,
    pub loader: String,
    pub trust_remote_code: bool,
    pub chat_template_mode: String,
    pub dtype: String,
    pub device_policy: DevicePolicy,
    pub estimated_memory: String,
    pub launch_defaults: Value,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchResponse {
    pub service_id: String,
    pub pid: u32,
    pub endpoint_url: String,
    pub log_path: String,
    pub state_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthcheckResponse {
    pub status: String,
    pub model_loaded: bool,
    pub device: String,
    pub uptime_sec: u64,
    pub queue_depth: u32,
    pub last_error: Option<String>,
    pub tokens_per_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointResponse {
    pub endpoint_url: String,
    pub api_style: String,
    pub supported_routes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopResponse {
    pub stopped: bool,
    pub graceful: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub log_path: String,
    pub recent_lines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_roundtrip_preserves_method() {
        let envelope = EngineRequestEnvelope {
            method: EngineMethod::Detect,
            payload: serde_json::json!({ "runtime_id": "therock-release" }),
        };

        let serialized = serde_json::to_string(&envelope).unwrap();
        let parsed: EngineRequestEnvelope = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed.method, EngineMethod::Detect);
    }
}
