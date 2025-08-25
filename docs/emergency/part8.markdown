# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 8 (Advanced Monitoring)

**Objective**: Implement custom Prometheus metrics for API performance and agent activity.

**Instructions for LLM**:
1. Create `server/monitoring.py` for custom Prometheus metrics.
2. Update `prometheus.yml` to include custom metrics.
3. Integrate with `server/main.py` for metric collection.
4. Ensure compatibility with existing `prometheus.yml`.

## Step 1: Create Monitoring File

### server/monitoring.py
```python
from prometheus_client import Counter, Histogram, start_http_server
from fastapi import FastAPI
from server.api.auth_endpoint import verify_token

REQUEST_COUNT = Counter("webxos_requests_total", "Total API requests", ["endpoint", "method"])
REQUEST_LATENCY = Histogram("webxos_request_latency_seconds", "Request latency", ["endpoint"])

def setup_metrics(app: FastAPI):
    @app.on_event("startup")
    async def start_metrics_server():
        start_http_server(8001)

    @app.middleware("http")
    async def record_metrics(request, call_next):
        endpoint = request.url.path
        method = request.method
        REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            response = await call_next(request)
        return response
```

### prometheus.yml (Updated)
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'mcp-api'
  - job_name: 'mcp-metrics'
    static_configs:
      - targets: ['localhost:8001']
        labels:
          service: 'mcp-custom-metrics'
  - job_name: 'spacex-api-metrics'
    static_configs:
      - targets: ['api.spacexdata.com:443']
        labels:
          service: 'spacex-api'
  - job_name: 'nasa-api-metrics'
    static_configs:
      - targets: ['api.nasa.gov:443']
        labels:
          service: 'nasa-api'
```

## Step 2: Integrate with Main Application
Update `server/main.py` to include metrics:
```python
from server.monitoring import setup_metrics
setup_metrics(app)
```

## Step 3: Validation
```bash
docker run -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
curl http://localhost:8001  # Check metrics endpoint
curl http://localhost:9090/api/v1/query?query=webxos_requests_total
```

**Next**: Proceed to `part9.md` for advanced API features.