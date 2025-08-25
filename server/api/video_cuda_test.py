from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter
from server.security.oauth2 import validate_token
import subprocess

router = APIRouter(prefix="/cuda/test/video")

VIDEO_TESTS_RUN = Counter(
    'mcp_video_cuda_tests_total',
    'Total video CUDA tests executed',
    ['status']
)

@router.get("")
async def run_video_test(token: str = Depends(validate_token)):
    config = TestRunnerConfig()
    cmd = [
        "python3", "run_tests.py",
        "--dir", config.test_dir,
        "--output", config.output_dir,
        "--config", config.config_file,
        "--parallel", str(config.parallel_runs)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    VIDEO_TESTS_RUN.labels(status="success" if result.returncode == 0 else "failed").inc()
    return Response(generate_latest(), media_type="text/plain")
