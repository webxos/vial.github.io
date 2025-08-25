import os
from wasmtime import Store, Module, Instance
from server.models.base import VideoFrame
from server.database.session import DatabaseManager

class WASMVideoProcessor:
    def __init__(self, wasm_path: str = "server/wasm/video_processor.wasm"):
        self.store = Store()
        self.module = Module.from_file(self.store.engine, wasm_path)
        self.instance = Instance(self.store, self.module, [])
        self.db_manager = DatabaseManager("sqlite:///./test.db")
    
    async def rasterize_svg(self, svg_path: str, width: int, height: int, output: str) -> Dict:
        """Rasterize SVG to video frame using WASM."""
        if not os.path.exists(svg_path):
            raise ValueError("SVG file not found")
        
        memory = self.instance.exports(self.store)["memory"]
        rasterize = self.instance.exports(self.store)["rasterize_svg"]
        
        # Allocate memory for SVG path
        svg_ptr = self.instance.exports(self.store)["alloc"](len(svg_path) + 1)
        memory.write(self.store, svg_ptr, svg_path.encode())
        
        # Call WASM function
        result_ptr = rasterize(self.store, svg_ptr, width, height)
        
        # Save frame metadata
        with self.db_manager.get_session() as session:
            frame = VideoFrame(svg_path=svg_path, duration=30, output_path=output, wallet_id="default")
            session.add(frame)
            session.commit()
            frame_id = frame.id
        
        return {"status": "success", "frame_id": frame_id, "output": output}
