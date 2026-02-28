from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from dotenv import load_dotenv
import os

load_dotenv()

@CrewBase
class ResearchAndBlogCrew():
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    my_llm = LLM(
        model=os.getenv("MODEL", "groq/llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],
            llm=self.my_llm
        )
        
    @agent
    def blog_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["blog_writer"],
            llm=self.my_llm
        )
        
    @task
    def report_task(self) -> Task:
        return Task(config=self.tasks_config["report_task"])
        
    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_writing_task"],
            output_file="blogs/blog.md"
        )
        
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )