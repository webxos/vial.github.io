```python
import svgutils.transform as sg
from fastapi import HTTPException

def optimize_svg(svg_path: str) -> dict:
    """Optimize SVG file for rendering.
    
    Parameters:
        svg_path: Path to the SVG file
    
    Returns:
        Dictionary with optimization status.
    """
    try:
        fig = sg.fromfile(svg_path)
        fig.set_size(("800px", "600px"))  # Standardize size
        optimized_path = svg_path.replace(".svg", "_opt.svg")
        fig.save(optimized_path)
        return {"status": "optimized", "path": optimized_path, "timestamp": "06:42 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(optimize_svg("public/uploads/test.svg"))
```
