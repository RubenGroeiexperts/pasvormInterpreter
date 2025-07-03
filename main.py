from fastapi import FastAPI, Request
import requests
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from xml.dom.minidom import Document
from skimage import measure
from math import hypot
from typing import List, Tuple

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}


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

    pad_size = 32
    image_np_padded = cv2.copyMakeBorder(
        image_np, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    image_cv = cv2.cvtColor(image_np_padded, cv2.COLOR_RGB2BGR)
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
        "height": image_cv.shape[0],
        "pad_size": pad_size
    }

def generate_inner_contour(image_shape, contour, offset_distance=8):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    contours = measure.find_contours(dist_transform, level=offset_distance)
    if not contours:
        return []
    best_contour = max(contours, key=len)
    return [(int(y), int(x)) for x, y in best_contour]

def remove_consecutive_duplicates(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [pt for i, pt in enumerate(points) if i == 0 or pt != points[i - 1]]

def remove_looping_tail(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    for i in range(1, len(points)):
        if points[i:] == points[:len(points)-i]:
            return points[:i]
    return points

def clean_polyline(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    points = remove_consecutive_duplicates(points)
    return remove_looping_tail(points)

def process_to_svg(image_bytes):
    data = extract_objects_from_bytes(image_bytes)
    pad = data.get("pad_size", 0)
    total_image_area = (data['width'] - 2 * pad) * (data['height'] - 2 * pad)

    num_objects = len(data['objects'])
    total_bbox_area = sum(w * h for (x, y, w, h) in (obj["bounding_box"] for obj in data['objects']))
    bbox_area_ratio = total_bbox_area / total_image_area if total_image_area > 0 else 0

    print(f"Found {num_objects} objects, bounding rect area ratio: {bbox_area_ratio:.2%}")

    if not (3 <= num_objects <= 9) or bbox_area_ratio < 0.4:
        doc = Document()
        svg = doc.createElement('svg')
        svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
        svg.setAttribute('width', '0')
        svg.setAttribute('height', '0')
        doc.appendChild(svg)
        return doc.toxml()

    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    doc.appendChild(svg)
    ref_x = (data['width'] - 2 * pad) * 0.25
    ref_y = 0
    ranked_objects = []
    for obj in data['objects']:
        x, y, w, h = obj['bounding_box']
        cx = x + w / 2
        cy = y + h / 2
        dist = hypot(cx - pad - ref_x, cy - pad - ref_y)
        ranked_objects.append((dist, obj))
    ranked_objects.sort(key=lambda t: t[0])
    all_points = []
    rank_0_a_polylines = []
    rank_0_b_polylines = []
    rank_0_polylines = []
    for rank, (_, obj) in enumerate(ranked_objects):
        border_points = clean_polyline([(x - pad, y - pad) for (x, y) in obj['border_points']])
        if not border_points:
            continue
        mask_shape = (data['height'], data['width'])
        red_points = clean_polyline([(x - pad, y - pad) for (x, y) in generate_inner_contour(mask_shape, obj['mask'], offset_distance=8)])
        if not red_points:
            continue
        if rank == 0:
            shifts = [(-15, -15, "rank_0_a"), (-30, -30, "rank_0_b")]
            for dx, dy, label in shifts:
                shifted_black = clean_polyline([(x + dx, y + dy) for (x, y) in border_points])
                shifted_red = clean_polyline([(x + dx, y + dy) for (x, y) in red_points])
                black_str = " ".join(f"{x},{y}" for (x, y) in shifted_black)
                red_str = " ".join(f"{x},{y}" for (x, y) in shifted_red)
                polyline_b = doc.createElement('polyline')
                polyline_b.setAttribute('points', black_str)
                polyline_b.setAttribute('stroke', 'black')
                polyline_b.setAttribute('fill', 'none')
                polyline_b.setAttribute('stroke-width', '1')
                polyline_b.setAttribute('class', label)
                polyline_r = doc.createElement('polyline')
                polyline_r.setAttribute('points', red_str)
                polyline_r.setAttribute('stroke', 'red')
                polyline_r.setAttribute('fill', 'none')
                polyline_r.setAttribute('stroke-width', '1')
                polyline_r.setAttribute('class', label)
                if label == "rank_0_a":
                    rank_0_a_polylines.extend([polyline_b, polyline_r])
                else:
                    rank_0_b_polylines.extend([polyline_b, polyline_r])
                all_points.extend(shifted_black)
                all_points.extend(shifted_red)
            black_str = " ".join(f"{x},{y}" for (x, y) in border_points)
            red_str = " ".join(f"{x},{y}" for (x, y) in red_points)
            polyline_black = doc.createElement('polyline')
            polyline_black.setAttribute('points', black_str)
            polyline_black.setAttribute('fill', 'none')
            polyline_black.setAttribute('stroke', 'black')
            polyline_black.setAttribute('stroke-width', '1')
            polyline_black.setAttribute('class', 'rank_0')
            polyline_red = doc.createElement('polyline')
            polyline_red.setAttribute('points', red_str)
            polyline_red.setAttribute('fill', 'none')
            polyline_red.setAttribute('stroke', 'red')
            polyline_red.setAttribute('stroke-width', '1')
            polyline_red.setAttribute('class', 'rank_0')
            rank_0_polylines.extend([polyline_black, polyline_red])
        else:
            black_str = " ".join(f"{x},{y}" for (x, y) in border_points)
            red_str = " ".join(f"{x},{y}" for (x, y) in red_points)
            black_polyline = doc.createElement('polyline')
            black_polyline.setAttribute('points', black_str)
            black_polyline.setAttribute('fill', 'none')
            black_polyline.setAttribute('stroke', 'black')
            black_polyline.setAttribute('stroke-width', '1')
            black_polyline.setAttribute('class', f'rank_{rank}')
            red_polyline = doc.createElement('polyline')
            red_polyline.setAttribute('points', red_str)
            red_polyline.setAttribute('fill', 'none')
            red_polyline.setAttribute('stroke', 'red')
            red_polyline.setAttribute('stroke-width', '1')
            red_polyline.setAttribute('class', f'rank_{rank}')
            svg.appendChild(black_polyline)
            svg.appendChild(red_polyline)
            all_points.extend(border_points)
            all_points.extend(red_points)

    for poly in rank_0_b_polylines + rank_0_a_polylines + rank_0_polylines:
        svg.appendChild(poly)

    min_x = min((x for x, _ in all_points), default=0)
    min_y = min((y for _, y in all_points), default=0)
    shift_x = -min_x if min_x < 0 else 0
    shift_y = -min_y if min_y < 0 else 0

    for poly in svg.getElementsByTagName('polyline'):
        coords = poly.getAttribute('points').split()
        shifted = []
        for pt in coords:
            x, y = map(int, pt.split(','))
            shifted.append(f"{x + shift_x},{y + shift_y}")
        poly.setAttribute('points', " ".join(shifted))

    all_x = [x + shift_x for x, _ in all_points]
    all_y = [y + shift_y for _, y in all_points]
    svg.setAttribute('width', str(max(all_x, default=0) + 1))
    svg.setAttribute('height', str(max(all_y, default=0) + 1))
    return doc.toxml()

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
