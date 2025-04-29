from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception("Failed to download image")

def project_onto_white(image):
    image = image.convert("RGBA")
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    white_bg.paste(image, (0, 0), image)
    return white_bg

def process_to_svg(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = project_onto_white(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    svg_elements = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_svg = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' style='fill:none;stroke:black;stroke-width:2' />"
        svg_elements.append(rect_svg)

    height, width = gray.shape
    svg_data = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>" + "".join(svg_elements) + "</svg>"
    return svg_data

@app.post("/process-image")
async def process_image(request: Request):
    data = await request.json()
    image_url = data["image_url"]
    image_data = requests.get(image_url).content
    svg_data = process_to_svg(image_data)
    return Response(content=svg_data, media_type="image/svg+xml")
