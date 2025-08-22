def setup_agent_tasks(app):
    app.state.oauth2_enabled = True
    app.state.token_hash = "test_hash"
    app.state.wallet = {"balance": 0}
    app.state.api_pipeline = lambda x: x
    app.state.token_merge = lambda x, y: x + y
