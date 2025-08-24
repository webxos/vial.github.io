# GIBS Integration Guide for WebXOS Vial MCP SDK

This guide details how to integrate NASA's Global Imagery Browse Services (GIBS) into the WebXOS Vial MCP SDK for real-time Earth imagery visualization.

## Overview
GIBS provides high-resolution Earth imagery via WMTS/WMS endpoints, supporting projections like EPSG:4326 (Geographic) and EPSG:3857 (Web Mercator). This SDK integrates GIBS with APOD, EPIC, EONET, and SpaceX APIs, using Three.js for 3D rendering and Astropy for data processing.

## Installation
1. Install dependencies:
   ```bash
   pip install requests httpx fastapi astropy crewai 'crewai[tools]' pytest
   npm install three


Set environment variables in .env:NASA_API_KEY=your_nasa_key
GIBS_API_URL=https://gibs.earthdata.nasa.gov



Usage

Fetch GIBS Tiles:
Use telescope.html console to query MODIS_Terra_CorrectedReflectance_TrueColor,2023-01-01.
API: POST /tools/gibs_data with { "layer": "MODIS_Terra_CorrectedReflectance_TrueColor", "time": "2023-01-01" }.


3D Visualization:
Tiles are rendered on a 3D sphere in telescope.html using gibs-visualizer.js.


Data Processing:
astronomy.py uses Astropy for FITS/coordinate processing.


Monitoring:
Prometheus tracks mcp_gibs_requests_total at /metrics/gibs.



Endpoints

POST /tools/gibs_data: Fetch GIBS WMTS tiles (requires NASA API key).
POST /tools/astronomy_data: Fetch APOD/EONET/SpaceX data.

Example Query
https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2023-01-01/250m/6/13/36.jpg

Testing
Run unit tests:
pytest server/tests/test_gibs.py

Resources

GIBS Documentation
OGC WMTS/WMS Specifications


