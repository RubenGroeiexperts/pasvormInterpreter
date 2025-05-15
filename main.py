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

def smooth_contour(coords, k=5):
    coords = np.array(coords, dtype=np.float32)
    coords = np.vstack([coords[-k:], coords, coords[:k]])
    kernel = np.ones((2 * k + 1, 1)) / (2 * k + 1)
    smoothed = np.convolve(coords[:, 0], kernel.ravel(), mode='valid'), \
               np.convolve(coords[:, 1], kernel.ravel(), mode='valid')
    return np.vstack(smoothed).T.tolist()


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

    def offset_inward(coords, offset_distance=5):
        coords = np.array(coords, dtype=np.float32)
        coords = np.vstack([coords, coords[0]])  # ensure closed loop

        vectors = np.roll(coords, -1, axis=0) - np.roll(coords, 1, axis=0)
        normals = np.zeros_like(vectors)
        normals[:, 0] = -vectors[:, 1]
        normals[:, 1] = vectors[:, 0]

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals /= norms

        inward_coords = coords[:-1] + offset_distance * normals[:-1]
        return inward_coords.tolist()

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

        # Uniform inward offset for inner border
        smoothed = smooth_contour(points, k=4)
        shrunken_global = offset_inward(smoothed, offset_distance=-5)

        shrink_points_str = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in shrunken_global)

        shrink_polyline = doc.createElement('polyline')
        shrink_polyline.setAttribute('points', shrink_points_str)
        shrink_polyline.setAttribute('fill', 'none')
        shrink_polyline.setAttribute('stroke', 'red')
        shrink_polyline.setAttribute('stroke-width', '1')
        svg.appendChild(shrink_polyline)

    return doc.toxml()
