# Deployment Guide: WebXOS 2025 Vial MCP SDK

## Overview
Deploy the WebXOS 2025 Vial MCP SDK on Vercel at `vial.github.io`, integrating MCP 2025-03-26, LangChain agents, and an 8-point SVG diagram interface. This guide covers setup, DNS configuration, and deployment.

## Prerequisites
- Node.js, Python 3.11, Vercel CLI, Docker.
- NASA API key, Higress API access.

## Setup
1. **Clone Repository**:
   ```bash
   git clone https://github.com/webxos/vial.github.io.git
   cd vial.github.io


Install Dependencies:pip install -r requirements.txt
npm install


Set Environment Variables:echo "NASA_API_KEY=your_key" >> .env
echo "GIBS_API_URL=https://gibs.earthdata.nasa.gov" >> .env
echo "HIGRESS_API_URL=https://higress.alibaba.com/api" >> .env
echo "ALIBABA_API_KEY=your_key" >> .env


Configure DNS for Vercel:
Nameservers: ns1.vercel-dns.com, ns2.vercel-dns.com.
CNAME: www to 4d59d46a56f561ba.vercel-dns-017.com (TTL 60).
Verify: dnschecker.org (24-48 hours).



Deployment

Test Locally:vercel dev


Deploy to Vercel:vercel --prod


Test SVG Diagram:
Assign roles, export/import network in index.html.


Test OBS Streaming:
Stream to obs://live/moon-mars/2023-01-01.
View in telescope.html.



Testing
pytest server/tests

Resources

Vercel DNS
OBS Streaming


