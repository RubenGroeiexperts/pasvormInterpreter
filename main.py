from fastapi import FastAPI, Request
import requests
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from xml.dom.minidom import Document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(request: Request):
    data = await request.json()
    image_url = data["image_url"]
    image_data = requests.get(image_url).content

    svg_data = process_to_svg(image_data)

    return Response(content=svg_data, media_type="image/svg+xml")

def remove_small_objects(image, min_area=500):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask

def extract_border_points(contour, min_area_threshold=500):
    area = cv2.contourArea(contour)
    if area < min_area_threshold:
        return []
    border_points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    num_points = len(border_points)
    if num_points > 300:
        indices = np.linspace(0, num_points - 1, 300, dtype=int)
        border_points = [border_points[i] for i in indices]
    elif num_points < 300 and num_points > 0:
        multiplier = 300 // num_points
        extra = 300 % num_points
        border_points = border_points * multiplier + border_points[:extra]
    return border_points

def extract_objects_from_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    white_bg.paste(image, (0, 0), image)
    image_np = np.array(white_bg)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cleaned = remove_small_objects(thresh, min_area=500)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        border_points = extract_border_points(cnt, min_area_threshold=500)
        objects.append({
            "bounding_box": (x, y, w, h),
            "mask": cnt,
            "border_points": border_points
        })
    return {
        "objects": objects,
        "width": image_cv.shape[1],
        "height": image_cv.shape[0]
    }

def process_to_svg(image_bytes):
    data = extract_objects_from_bytes(image_bytes)

    def shrink_and_reposition(coords):
        max_x = max(x for x, y in coords)
        max_y = max(y for x, y in coords)
        coordinates2 = [[x * ((max_x - 20) / max_x) + 10,
                        y * ((max_y - 20) / max_y) + 10] for x, y in coords]
        return coordinates2

    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('width', f"{data['width']}")
    svg.setAttribute('height', f"{data['height']}")
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    doc.appendChild(svg)

    for obj in data['objects']:
        points = obj['border_points']
        if not points:
            continue

        # Outer polyline
        points_str = " ".join(f"{x},{y}" for (x, y) in points)
        polyline = doc.createElement('polyline')
        polyline.setAttribute('points', points_str)
        polyline.setAttribute('fill', 'none')
        polyline.setAttribute('stroke', 'black')
        polyline.setAttribute('stroke-width', '1')
        svg.appendChild(polyline)

        # Shrinked inner shape
        x0, y0, w, h = obj['bounding_box']
        local_coords = [[x - x0, y - y0] for (x, y) in points]
        shrunken_local = shrink_and_reposition(local_coords)
        shrunken_global = [[x + x0, y + y0] for (x, y) in shrunken_local]
        shrink_points_str = " ".join(f"{x},{y}" for (x, y) in shrunken_global)

        shrink_polyline = doc.createElement('polyline')
        shrink_polyline.setAttribute('points', shrink_points_str)
        shrink_polyline.setAttribute('fill', 'none')
        shrink_polyline.setAttribute('stroke', 'red')
        shrink_polyline.setAttribute('stroke-width', '1')
        svg.appendChild(shrink_polyline)

    return doc.toxml()
