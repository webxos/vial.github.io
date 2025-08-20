import os
import json

class ResourceLoader:
    def __init__(self, base_dir: str = "resources"):
        self.base_dir = base_dir

    def load_resource(self, resource_name: str):
        path = os.path.join(self.base_dir, f"{resource_name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {"error": f"Resource {resource_name} not found"}

    def save_resource(self, resource_name: str, data: dict):
        path = os.path.join(self.base_dir, f"{resource_name}.json")
        with open(path, 'w') as f:
            json.dump(data, f)

resource_loader = ResourceLoader()
