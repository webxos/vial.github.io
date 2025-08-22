from server.security.auth import oauth2_scheme
from server.api.webxos_wallet import export_wallet
def setup_agent_tasks(app):
    app.state.oauth2_enabled = True
    app.state.token_hash = "test_hash"
    app.state.wallet = export_wallet("test", None)
    app.state.api_pipeline = lambda x: x
    app.state.token_merge = lambda x, y: x + y
