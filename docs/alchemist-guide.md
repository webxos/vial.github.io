# MCP Alchemist Guide: Vial MCP SDK

## Overview
The `mcp_alchemist` coordinates supply chain agents in Dropship mode using 4x PyTorch models, integrated with the Vial MCP SDK on Vercel. It processes NASA (GIBS, APOD, EONET), SpaceX, and Higress API data, linking to `.md` DAO wallets for contribution tracking.

## Usage
1. **Access Alchemist API**:
   ```bash
   curl -X POST https://vial.github.io/api/mcp/alchemist/coordinate \
     -H "X-API-Key: your_nasa_key" \
     -d '{"route": "moon-mars", "time": "2023-01-01", "wallet_id": "test-wallet"}'


Integration:
Called within Dropship mode to coordinate agents.
Outputs agent coordination results and simulation data.


DAO Wallet:
Tracks contributions via reputation.py for future rewards.


Vercel Hosting:
Nameservers: ns1.vercel-dns.com, ns2.vercel-dns.com.
CNAME: www to 4d59d46a56f561ba.vercel-dns-017.com (TTL 60).
Deploy: vercel --prod.



Testing
pytest server/tests/test_alchemist.py

Resources

PyTorch Documentation
Vercel DNS


