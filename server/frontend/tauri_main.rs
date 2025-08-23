use tauri::{command, State};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
struct MCPState {
    oauth_token: Option<String>,
}

#[command]
async fn execute_mcp_tool(tool: String, params: serde_json::Value, state: State<Mutex<MCPState>>) -> Result<String, String> {
    let state = state.lock().unwrap();
    if state.oauth_token.is_none() {
        return Err("Unauthorized: OAuth token required".into());
    }
    // Placeholder: Execute MCP tool via HTTP
    Ok(format!("Executed tool: {}", tool))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(MCPState { oauth_token: None }))
        .invoke_handler(tauri::generate_handler![execute_mcp_tool])
        .run(tauri::generate_context!())
        .expect("Error running Tauri app");
}
