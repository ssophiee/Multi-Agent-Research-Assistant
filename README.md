# Multi-Agent-Research-Assistant
A multi-agent system that simulates a researcher: querying search engines, collecting academic papers, and generating concise summaries using LLMs.
## Overview
This project is a multi-agent system designed to assist researchers in gathering and summarizing academic papers. The system consists of several agents, each with specific roles, including querying search engines, collecting papers, and generating summaries using large language models. The architecture is modular, allowing for easy integration of new agents or functionalities.
## Architecture
- **Search Agent**: This agent is responsible for querying search engines (DuckDuckGo and Arxiv) and retrieving relevant academic papers based on user-defined topic.
- **Quality Checker Agent**: This agent evaluates the quality of the retrieved papers and makes a decision to resubmit the improved topic to the search agent if the search was insufficient.
- **Summary Agent**: This agent generates concise summaries of the collected academic papers.
- **Communication Agent**: This component allows users to interact with the system, input topics, and receive the generated summaries.

## Installation
install the required packages using pip:
```bash
pip install -r requirements.txt
```
## Set up interface
1. Create a `.env` file in the root directory of the project.
2. Add the following lines to the `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
```
3. Run the main script:
```bash
uvicorn main:app --reload --port=8005 --host=0.0.0.0 
```
4. Open your web browser and navigate to `http://localhost:8005/` to access the API documentation and test the endpoints.

## Usage
1. Start the server using the command mentioned above.
2. Use it to submit a research topic.
3. The system will process the request, query the search engines, collect papers, and generate summaries.
4. The results will be returned in JSON format, including the summaries of the relevant papers.
