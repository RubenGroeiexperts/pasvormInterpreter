from fastapi import FastAPI, Request
import requests
from fastapi.responses import Response

app = FastAPI()

@app.post("/process-image")
async def process_image(request: Request):
    data = await request.json()
    image_url = data["image_url"]
    image_data = requests.get(image_url).content

    # Your image processing here
    svg_data = process_to_svg(image_data)

    return Response(content=svg_data, media_type="image/svg+xml")

def process_to_svg(image_bytes):
    # Dummy SVG generator
    return "<svg height='100' width='100'><circle cx='50' cy='50' r='40' stroke='black' stroke-width='3' fill='red' /></svg>"
