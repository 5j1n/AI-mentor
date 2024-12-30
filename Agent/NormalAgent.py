import Agent

class NormalAgent(Agent.Agent):
    def __init__(self):
        super().__init__()

    async def normalagent(self, question, continuous, history):
        response = ""
        async for partial_response in self.generate_response(question, None, continuous, history):
            response += partial_response
            yield partial_response        
        print(f"Question: {question}")
        print(f"Response: {response}\n")