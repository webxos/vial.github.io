Vial MCP Controller API Documentation
Overview
The Vial MCP Controller API provides endpoints for managing configurations, agents, and real-time collaboration.
Authentication

POST /auth/token: Obtain JWT token.
Request: username, password (form data)
Response: { "access_token": string, "token_type": "bearer" }



Core Endpoints

GET /health: Check system status.
Response: { "status": "healthy", "version": "2.9.3" }


POST /save-config: Save a visual configuration.
Request: { "name": string, "components": array, "connections": array }
Response: { "status": "saved", "config_id": string }


GET /load-config/{config_id}: Load a configuration.
Response: { "config": { "id": string, "name": string, "components": array, "connections": array } }



WebSocket

/ws: Real-time collaboration for cursor and config updates.
Payload: { "type": "cursor_update" | "config_update", "user_id": string, "position": { "x": number, "y": number } | "config": object }



Deployment

POST /deploy-config: Deploy configuration to GitHub Pages.
Request: { "config_id": string }
Response: { "status": "deployed", "url": string }


