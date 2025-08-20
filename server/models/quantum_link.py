class QuantumLink:
    def __init__(self):
        self.links = {}

    def establish_link(self, node_a: str, node_b: str):
        link_id = f"{node_a}-{node_b}"
        self.links[link_id] = {"status": "active", "time": "01:35 PM EDT, Aug 20, 2025"}
        return {"link_id": link_id, "status": "established"}
