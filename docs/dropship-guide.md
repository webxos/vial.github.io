# Dropship Mode Guide: Vial MCP SDK

## Overview
Dropship mode is the unified testing interface for The Vial MCP SDK, integrating SVG diagram creation, SpaceX launch/Starlink data, Moon-Mars supply chain simulation, and real-time AR/VR streaming via OBS. It leverages NASA (GIBS, APOD, EONET), SpaceX, and Higress APIs, with `mcp_alchemist` (4x PyTorch models) coordinating agents.

## Usage
1. **Select Dropship Mode**:
   - In `index.html`, choose `DROPSHIP` from the mode selector.
   - Input: `route,time` (e.g., `moon-mars,2023-01-01`).
2. **API Call**:
   ```bash
   curl -X POST http://localhost:8000/api/mcp/tools/dropship_data \
     -H "X-API-Key: your_nasa_key" \
     -d '{"route": "moon-mars", "time": "2023-01-01", "wallet_id": "test-wallet"}'


Visualization:
3D popup globe in telescope.html using Three.js (GIBS textures).
OBS streaming: View live feeds at obs://live/moon-mars/2023-01-01.


Testing OBS:
Configure OBS to stream to the provided URL.
Test in telescope.html for AR/VR compatibility.


DAO Wallet:
Contributions tracked via .md wallets, linked to reputation.py.



Testing
pytest server/tests/test_dropship.py

Resources

GIBS Documentation
SpaceX API
Higress API
OBS Streaming


