class PromptManager:
    def __init__(self):
        self.prompts = {}

    def add_prompt(self, name: str, command: str, callback):
        self.prompts[name] = {"command": command, "callback": callback}

    def execute_prompt(self, input_cmd: str):
        for name, data in self.prompts.items():
            if input_cmd.startswith(data["command"]):
                return data["callback"](input_cmd[len(data["command"]):].strip())
        return "Command not found. Use /help for available prompts."

prompt_manager = PromptManager()
