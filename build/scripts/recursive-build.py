import yaml
import subprocess
import os
from pathlib import Path

def load_build_matrix(matrix_path: str = "build/configs/build-matrix.yaml") -> Dict:
    """Load build matrix configuration."""
    with open(matrix_path, 'r') as f:
        return yaml.safe_load(f)

def build_component(component: Dict, base_dir: str):
    """Build a single component using its Dockerfile."""
    dockerfile = component['dockerfile']
    tag = component['tag']
    cmd = ["docker", "build", "-f", dockerfile, "-t", tag, base_dir]
    subprocess.run(cmd, check=True)

def recursive_build():
    """Orchestrate recursive Docker builds for all components."""
    matrix = load_build_matrix()
    base_dir = str(Path(__file__).parent.parent.parent)
    
    for component in matrix['components']:
        print(f"Building {component['name']}...")
        build_component(component, base_dir)
    
    print("Building final MCP image...")
    subprocess.run([
        "docker", "build", "-f", "build/dockerfiles/mcp-final.Dockerfile",
        "-t", "webxos-mcp:latest", base_dir
    ], check=True)

if __name__ == "__main__":
    recursive_build()
