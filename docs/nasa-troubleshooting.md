NASA Troubleshooting Guide for WebXOS 2025 Vial MCP SDK
Overview
This guide addresses common issues with NASA tool integrations in the WebXOS 2025 Vial MCP SDK, ensuring smooth operation of satellite, telescope, and visualization features.
Common Issues & Resolutions

API Failures:

Symptom: 403 Forbidden or 429 Too Many Requests.
Cause: Invalid API key or exceeded quota.
Fix: Verify NASA_API_KEY in .env and implement rate limiting in nasa_api_client.py.


Visualization Errors:

Symptom: Black screen or missing orbits in nasa_orbit_viz.js.
Cause: Missing data or Three.js initialization failure.
Fix: Check network connectivity and ensure VIAL_API_URL is correct; test with curl http://localhost:8000/mcp/nasa_satellite/data.


Telescope Processing Issues:

Symptom: Empty processed_image in nasa_telescope.py.
Cause: Invalid image URL or CUDA unavailability.
Fix: Validate NASA APOD URL and ensure NVIDIA drivers are installed; run nvidia-smi to confirm.


Backup Failures:

Symptom: CronJob nasa-backup fails with permission errors.
Cause: Insufficient PVC storage or misconfigured mount path.
Fix: Increase storage: 10Gi in nasa-backup.yaml and verify BACKUP_PATH access.


Scaling Problems:

Symptom: HPA not scaling nasa-deployment.
Cause: Missing Prometheus metrics or misconfigured targets.
Fix: Ensure nasa-monitoring.yaml is applied and check webxos_requests_total in Prometheus.



Best Practices

Regularly test endpoints with pytest to catch issues early.
Monitor GPU usage with nvidia-smi during heavy NASA data processing.
Keep backups synchronized with kubectl get cronjob nasa-backup.

Contact Support
Join the WebXOS Community for assistance with unresolved issues.
