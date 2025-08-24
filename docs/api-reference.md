# WebXOS Vial MCP API Reference

## Overview
The WebXOS Vial MCP server provides APIs for GIBS, NASA (APOD, EPIC, EONET), and SpaceX data, integrated with 3D visualization and Astropy processing.

## Endpoints

### POST /api/mcp/tools/gibs_data
Fetch GIBS WMTS tiles.
- **Headers**: `X-API-Key: <NASA_API_KEY>`
- **Body**: `{ "layer": "string", "time": "YYYY-MM-DD", "wallet_id": "string" }`
- **Response**: `{ "gibs": { "url": "string", "layer": "string", "time": "string" }, "wallet_id": "string" }`

### POST /api/mcp/tools/astronomy_data
Fetch APOD, EONET, and SpaceX data.
- **Headers**: `X-API-Key: <NASA_API_KEY>`
- **Body**: `{ "query": "string", "wallet_id": "string" }`
- **Response**: `{ "apod": {}, "eonet": {}, "spacex": {}, "wallet_id": "string" }`

## Authentication
- Requires NASA API key in `X-API-Key` header.
- Configure in `.env`: `NASA_API_KEY=your_key`.

## Example
```bash
curl -X POST http://localhost:8000/api/mcp/tools/gibs_data \
  -H "X-API-Key: your_nasa_key" \
  -d '{"layer": "MODIS_Terra_CorrectedReflectance_TrueColor", "time": "2023-01-01"}'

Monitoring

Metrics: /metrics/gibs (mcp_gibs_requests_total), /metrics/astronomy (mcp_astronomy_tasks_total).


