from crewai import Crew, Agent, Task
from crewai_tools import SerpAPITool, ScrapeWebsiteTool
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

class AgenticSearch:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.researcher = Agent(
            role="Researcher",
            goal="Find and analyze 5+ sources for a given topic",
            tools=[SerpAPITool(), ScrapeWebsiteTool()]
        )
        self.writer = Agent(
            role="Writer",
            goal="Summarize findings into a coherent report"
        )
        self.critic = Agent(
            role="Critic",
            goal="Validate accuracy and flag biases"
        )

    async def run_search(self, topic: str, request_id: str = str(uuid.uuid4())) -> dict:
        try:
            research_task = Task(
                description=f"Research {topic} for 10 mins (5+ sources).",
                agent=self.researcher,
                timeout=600
            )
            write_task = Task(
                description=f"Summarize research findings for {topic}.",
                agent=self.writer
            )
            review_task = Task(
                description="Review for accuracy/bias.",
                agent=self.critic
            )
            crew = Crew(agents=[self.researcher, self.writer, self.critic], tasks=[research_task, write_task, review_task])
            result = await crew.kickoff_async(inputs={"topic": topic})
            await self.memory_manager.save_session(
                f"search_{topic}",
                {"topic": topic, "sources_analyzed": len(result.get("sources", [])), "summary": result.get("summary")},
                request_id
            )
            logger.info(f"Agentic search completed for {topic}", request_id=request_id)
            return {
                "topic": topic,
                "sources": result.get("sources", []),
                "summary": result.get("summary", ""),
                "validation": result.get("validation", "Passed"),
                "request_id": request_id
            }
        except Exception as e:
            logger.error(f"Agentic search error: {str(e)}", request_id=request_id)
            raise
