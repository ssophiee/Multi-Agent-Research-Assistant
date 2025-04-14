from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from fastapi.templating import Jinja2Templates

from camel.toolkits import FunctionTool
from camel.agents import ChatAgent

from agents import research_pipeline, create_pdf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

query_storage = {} 

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_search(request: Request, query: str = Form(...)):
    query_storage["last_query"] = f"You searched for: {query}"
    result = research_pipeline(query)
    query_storage["result"] = result
    return templates.TemplateResponse("index.html", {"request": request, "result": query_storage["result"]})

@app.get("/download")
async def download_file():
    pdf_filename = "summary.pdf"
    content = query_storage.get("result", "No content available.")
    create_pdf(content, pdf_filename)  
    buffer = BytesIO()
    with open(pdf_filename, "rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename={pdf_filename}"
    })
