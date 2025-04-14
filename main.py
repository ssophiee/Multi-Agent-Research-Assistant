from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# app.mount("/", StaticFiles(directory=""), name="static")

document_storage = {}  # In-memory "doc" to return

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_search(request: Request, query: str = Form(...)):
    document_storage["last_doc"] = f"You searched for: {query}"
    return templates.TemplateResponse("index.html", {"request": request, "result": document_storage["last_doc"]})

@app.get("/download")
async def download_file():
    content = document_storage.get("last_doc", "No content available.")
    buffer = BytesIO(content.encode("utf-8"))
    return StreamingResponse(buffer, media_type="text/plain", headers={
        "Content-Disposition": "attachment; filename=result.txt"
    })
