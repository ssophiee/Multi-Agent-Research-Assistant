import re
import os
import json
import requests
from dotenv import load_dotenv

from openai import OpenAI
from camel.toolkits import FunctionTool, SearchToolkit, ArxivToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import logging
import tiktoken

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def search_arxiv_tool(query: str, max_results: int = 10) -> str:
    """
    Searches ArXiv for papers matching the query.
    """
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    
    return response.text

def duckduckgo_wrapper(query: str, max_results: int = 10) -> str:
    """
    Searches DuckDuckGo for papers matching the query.

    Args:
    - query: The search query.
    - max_results: Max number of search results.

    Returns:
    A string containing the search results.
    """
    return SearchToolkit().search_duckduckgo(query=query, max_results=max_results)

class SearchAgent:
    def __init__(self):
        self.sys_msg = "You are the search agent who finds the most relevant and high quality resources for the given question."
        self.duck_tool = FunctionTool(duckduckgo_wrapper)
        self.arxiv_tool = FunctionTool(search_arxiv_tool)

        tools = [self.duck_tool, self.arxiv_tool]
        self.model = ChatAgent(system_message=self.sys_msg, tools=tools)

    def step(self, query):
        return self.model.step(query)
    
class SummarizerAgent:
    """ description ... """
    def __init__(self, model_name, api_key, temperature):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.memory = []

    def add_event(self, prompt: str):
        self.memory.append({"role": "user", "content": prompt})

    def add_response(self, response: str):
        self.memory.append({"role": "assistant", "content": response})

    def summarize(self, topic, data):

        prompt = f"""
        You are a research summarization assistant. Your task is to summarize the sources gathered for this topic: {topic}.

        === Recent News Articles & Papers ===
        {data}

        Write the most important information about the source and add links for each article and paper.

        !!! IMPORTANT Example as a reference how it should look EXACTLY:

        <b>Arxiv Papers:</b>

        1. <b>LLM Multi-Agent Systems: Challenges and Open Problems</b><br/>
        <b>Description:</b> This paper explores existing works on multi-agent systems, identifying challenges that are still open.<br/>
        <b>Link:</b> [Read the paper](https://arxiv.org/abs/2403.12345)
        2. 2. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Improvements
        <b>Description:</b> Explores the safety implications of using MCP with large language models (LLMs), detailing
        how MCP enhances system integration and functionality
        <b>Link:</b> [Read the paper](https://arxiv.org/abs/2504.03767)

        <b>News Articles:</b>

        1. <b>AI Trends 2025: AI Agents and Multi-Agent Systems with Victor Dibia - Spotify</b><br/>
        <b>Description:</b> Victor Dibia, a principal research software engineer at Microsoft Research, discusses key trends in AI agents.<br/>
        <b>Link:</b> [Listen on Spotify](https://news.example.com/2025/04/03/spotify-ep1)

        2. <b>Google's AI Co Scientist and the True Power of Multi-Agent Systems</b><br/>
        <b>Description:</b> Google's AI Co-Scientist model involves multiple agents working together to generate and validate scientific hypotheses.<br/>
        <b>Link:</b> [Listen on Spotify](https://news.example.com/2025/04/03/spotify-ep2)

        3. <b>OpenAI to Evolve into Multi-Agent Systems - Spotify</b><br/>
        <b>Description:</b> OpenAI has launched a new research team focusing on multi-agent systems. The episode explores this transition.<br/>
        <b>Link:</b> [Listen on Spotify](https://news.example.com/2025/04/03/spotify-ep3)

        4. <b>Cooperative Resilience in Multi-Agent Systems by Agentic Horizons</b><br/>
        <b>Description:</b> This episode introduces the concept of cooperative resilience, a metric for measuring the performance of agents working in teams.<br/>
        <b>Link:</b> [Listen on Spotify](https://news.example.com/2025/04/03/spotify-ep4)
        """

        self.add_event(prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.memory[-10:], 
            temperature=self.temperature
        ).choices[0].message.content.strip()
        self.add_response(response)

        return response

class QualityCheckerAgent:
  """ description ... """

  def __init__(self, model_name, api_key, temperature):
    self.client = OpenAI(api_key=api_key)
    self.model_name = model_name
    self.temperature = temperature
    self.search_result_approved = False
    self.memory = []
    self.search_history = []
    self.query_history = []

  def add_event(self, prompt):
      self.memory.append({"role": "user", "content": prompt})

  def add_response(self, response):
      self.memory.append({"role": "assistant", "content": response})

  def evaluate_quality(self, query, search_result):
    # in case quality checker doesnt return dictionay correctly, it will fall down to summary of the current search queries

    self.search_history.append(search_result)
    formatted_search_results = self.format_search_history() 
    prompt = f"""You are acting as a **strict** quality control agent responsible for validating research relevance and completeness.

    If the following research results are anything less than **comprehensive, highly relevant, and up-to-date**, your job is to **reject them** and request a refined search.
    At least 10 articles should be present in the response, otherwise answer is insufficient.

    Instructions:
    - Default to `"resubmit"` unless the results are clearly sufficient in every way
    - If any key areas are missing, underdeveloped, outdated, or weakly supported — return `"resubmit"`.
    - Only return `"summary"` if the research is clearly comprehensive, highly relevant, and no further search is needed.

    Return a dictionary with:
    - `updated_query`: a refined search prompt if additional research needed in different sub-topics, dont repeat previous queries
    - `status`: either `"summary"` or `"resubmit"`.

    Previous queries:
    {str(self.query_history)}
    Current query to imporve:
    {query}
    Search History:
    {formatted_search_results}

    Example (dictionary):
    {{"updated_query": "search for recent work on memory persistence in multi-agent summarization frameworks", "status": "resubmit"}}
    """

    self.add_event(prompt)

    try:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.memory[-2:], 
            temperature=self.temperature
        ).choices[0].message.content.strip()
        response = json.loads(response)
        if not isinstance(response, dict):
            raise ValueError("Response is not a dictionary")
        status = response['status']
        if status == "summary":
            self.search_result_approved = True
        self.add_response(str(response))
        return response
    except Exception as e:
        response = {"updated_query": "-", "status": "summary"}
        self.add_response(str(response))
        self.search_result_approved = True
        return response

  def format_search_history(self):
    full_search_results = " ".join(self.search_history)
    return full_search_results

  def load_search_results(self):
    return self.search_history

class CommunicationAgent:
  def __init__(self, quality_checker_agent, search_agent, summarizer_agent):
    self.quality_checker = quality_checker_agent
    self.searcher = search_agent
    self.summarizer = summarizer_agent
    self.memory = []

  def handle_query(self, query):
      self.memory.append({"role": "user", "content": query})

      print("🔍 Searching...")
      current_search = self.searcher.step(query).msgs[0].content
      self.memory.append({"role": "search", "content": current_search})

      # Reset checker state for this run
      self.quality_checker.search_result_approved = False
      self.quality_checker.search_history = []
      self.query_history = []


      self.query_history.append(query)
      while not self.quality_checker.search_result_approved:
          print("💯 Checking Quality...")
          self.quality_checker.search_history.append(current_search)
          quality_estimation = self.quality_checker.evaluate_quality(query, self.quality_checker.format_search_history())
          self.memory.append({"role": "quality_checker", "content": quality_estimation})

          if quality_estimation['status'] == 'resubmit':
              print("🔦 Decision to resubmit for search")
              updated_query = quality_estimation['updated_query']
              self.query_history.append(updated_query)
            
              current_search = self.searcher.step(updated_query).msgs[0].content
              print("📚 New search results received.")
          else:
              print("🧠 Summarizing...")
              trimmed_memory = trim_to_token_limit(self.quality_checker.format_search_history())
              summary = self.summarizer.summarize(query, trimmed_memory)
              self.memory.append({"role": "summarizer", "content": summary})
              break 

      return summary

# TO DO: this function is trimming incorrectly, fix it.
MAX_TOKENS = 16385
def trim_to_token_limit(text, model="gpt-3.5-turbo", max_tokens=MAX_TOKENS):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    else:
        trimmed = enc.decode(tokens[:max_tokens])
        return trimmed
    
def create_pdf(text, filename):
    """ Creates a PDF file from the given text."""
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    normal = styles['Normal']
    normal.spaceAfter = 6

    link_style = ParagraphStyle(
        'LinkStyle',
        parent=normal,
        fontSize=11,
        leading=14,
        textColor=colors.black
    )

    content = []

    url_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s]+)\)')

    for line in text.strip().split("\n"):
        if not line.strip():
            content.append(Spacer(1, 6))
            continue

        def replace_link(match):
            label, url = match.groups()
            return f'<a href="{url}" color="blue">{label}</a>'

        line = url_pattern.sub(replace_link, line)

        para = Paragraph(line, link_style)
        content.append(para)

    doc.build(content)

# Initialize the agents
    
MODEL_NAME = "gpt-3.5-turbo"
search_agent = SearchAgent()
summarizer = SummarizerAgent(MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.2)
quality_checker = QualityCheckerAgent(MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.2)
communication_agent = CommunicationAgent(quality_checker, search_agent, summarizer)

def research_pipeline(query):
    topic = f"Find recent papers on {query} (focus on arxiv papers, but also other sources)"
    result = communication_agent.handle_query(topic)
    return result
