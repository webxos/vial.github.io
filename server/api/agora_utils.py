```python
import base64
import hashlib
import urllib.parse
from typing import Optional

import requests
import yaml

def extract_substring(text: str, start_tag: str, end_tag: str, include_tags=True) -> Optional[str]:
    start_position = text.lower().find(start_tag.lower())
    end_position = text.lower().find(end_tag.lower(), start_position + len(start_tag))
    if start_position == -1 or end_position == -1:
        return None
    return text[start_position : end_position + len(end_tag)].strip() if include_tags else text[start_position + len(start_tag) : end_position].strip()

def compute_hash(s: str) -> str:
    m = hashlib.sha1()
    m.update(s.encode())
    return base64.b64encode(m.digest()).decode("ascii")

def extract_metadata(text: str) -> dict:
    metadata = extract_substring(text, "---", "---", include_tags=False)
    return yaml.safe_load(metadata) or {"name": "Unnamed protocol", "description": "No description", "multiround": False}

def encode_as_data_uri(text: str) -> str:
    return "data:text/plain;charset=utf-8," + urllib.parse.quote(text)

def download_and_verify_protocol(protocol_hash: str, protocol_source: str, timeout: int = 10000) -> Optional[str]:
    if protocol_source.startswith("data:"):
        if protocol_source.startswith("data:text/plain;charset=utf-8;base64,"):
            return base64.b64decode(protocol_source[len("data:text/plain;charset=utf-8;base64,") :]).decode("utf-8")
        elif protocol_source.startswith("data:text/plain;charset=utf-8,"):
            return urllib.parse.unquote(protocol_source[len("data:text/plain;charset=utf-8,") :])
        return None
    response = requests.get(protocol_source, timeout=timeout // 1000)  # Convert ms to s
    return response.text if response.status_code == 200 and compute_hash(response.text) == protocol_hash else None
```
