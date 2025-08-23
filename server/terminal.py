import logging
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextArea
from httpx import AsyncClient

logger = logging.getLogger(__name__)
MCP_SERVER_URL = "http://127.0.0.1:8000"

class MCPTerminal(App):
    """Textual-based terminal for MCP Alchemist."""
    CSS_PATH = "terminal.css"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Enter command (e.g., 'process 0.5')")
        yield TextArea(id="output")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        output = self.query_one("#output", TextArea)
        command = event.value.split()
        try:
            if command[0] == "process":
                async with AsyncClient() as client:
                    response = await client.post(
                        f"{MCP_SERVER_URL}/alchemist/process",
                        json={"input": float(command[1]), "llm_provider": "anthropic"}
                    )
                    output.text = str(response.json())
        except Exception as e:
            logger.error(f"Terminal command failed: {str(e)}")
            output.text = f"Error: {str(e)}"

if __name__ == "__main__":
    app = MCPTerminal()
    app.run()
