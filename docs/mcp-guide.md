# MCP Integration Guide: Vial MCP SDK

## Overview
The Vial MCP SDK integrates the Model Context Protocol (MCP) into a planetary distribution system, hosted on Vercel at `vial.github.io`. It supports Three.js visualization, `mcp_alchemist` (4x PyTorch models), OBS streaming, and DAO wallets for testing supply chains and economic democracy.

## Modes
- **Dropship**: Unified mode for SVG diagrams, SpaceX/NASA/Higress data, and Moon-Mars supply chain simulation with 3D popup globe and OBS streaming.
- **Galaxycraft**: Space exploration game simulation.
- **Telescope**: Real-time AR/VR OBS feeds for astronomy data.

## MCP Alchemist
- **Function**: Coordinates agents using 4x PyTorch models.
- **API**: `/api/mcp/alchemist/coordinate`.

## Usage
1. **Dropship Mode**:
   ```bash
   curl -X POST https://vial.github.io/api/mcp/tools/dropship_data \
     -H "X-API-Key: your_nasa_key" \
     -d '{"route": "moon-mars", "time": "2023-01-01", "wallet_id": "test-wallet"}'


MCP Alchemist:curl -X POST https://vial.github.io/api/mcp/alchemist/coordinate \
  -H "X-API-Key: your_nasa_key" \
  -d '{"route": "moon-mars", "time": "2023-01-01", "wallet_id": "test-wallet"}'


OBS Streaming:
Configure OBS to stream to obs://live/moon-mars/2023-01-01.
View in telescope.html.



Vercel Hosting

DNS Setup:
Nameservers: ns1.vercel-dns.com, ns2.vercel-dns.com.
CNAME: www to 4d59d46a56f561ba.vercel-dns-017.com (TTL 60).


Deploy: vercel --prod.

Testing
pytest server/tests

Resources

GIBS Documentation
SpaceX API
Higress API
OBS Streaming
Vercel DNS


