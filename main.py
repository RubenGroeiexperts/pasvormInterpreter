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
        "height": image_cv.shape[0],
        "original_image": gray
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

def process_rank_0_object(data, rank_0_obj):
    x, y, w, h = rank_0_obj['bounding_box']
    roi = data['original_image'][y:y+h, x:x+w]

    mask = np.zeros_like(roi)
    cv2.drawContours(mask, [rank_0_obj['mask'] - [x, y]], -1, 255, thickness=cv2.FILLED)
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

    roi_inverted = cv2.bitwise_not(roi_masked)
    _, binary_inv = cv2.threshold(roi_inverted, 127, 255, cv2.THRESH_BINARY)
    contours_inv, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_inv:
        return []

    largest = max(contours_inv, key=cv2.contourArea)
    border_points = [(int(pt[0][0] + x), int(pt[0][1] + y)) for pt in largest]
    return border_points

def process_to_svg(image_bytes):
    data = extract_objects_from_bytes(image_bytes)

    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('width', f"{data['width']}")
    svg.setAttribute('height', f"{data['height']}")
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    doc.appendChild(svg)

    ref_x = data['width'] * 0.25
    ref_y = 0
    ranked_objects = []
    for obj in data['objects']:
        x, y, w, h = obj['bounding_box']
        cx = x + w / 2
        cy = y + h / 2
        dist = hypot(cx - ref_x, cy - ref_y)
        ranked_objects.append((dist, obj))

    ranked_objects.sort(key=lambda t: t[0])

    all_points = []

    for rank, (_, obj) in enumerate(ranked_objects):
        border_points = obj['border_points']
        if not border_points:
            continue

        black_str = " ".join(f"{x},{y}" for (x, y) in border_points)
        black_polyline = doc.createElement('polyline')
        black_polyline.setAttribute('points', black_str)
        black_polyline.setAttribute('fill', 'none')
        black_polyline.setAttribute('stroke', 'black')
        black_polyline.setAttribute('stroke-width', '1')
        black_polyline.setAttribute('class', f'rank_{rank}')
        svg.appendChild(black_polyline)

        mask_shape = (data['height'], data['width'])
        red_points = generate_inner_contour(mask_shape, obj['mask'], offset_distance=8)
        if red_points:
            red_str = " ".join(f"{x},{y}" for (x, y) in red_points)
            red_polyline = doc.createElement('polyline')
            red_polyline.setAttribute('points', red_str)
            red_polyline.setAttribute('fill', 'none')
            red_polyline.setAttribute('stroke', 'red')
            red_polyline.setAttribute('stroke-width', '1')
            red_polyline.setAttribute('class', f'rank_{rank}')
            svg.appendChild(red_polyline)

        if rank == 0:
            shifts = [(-15, -15, "rank_0_a"), (-30, -30, "rank_0_b")]
            for dx, dy, label in shifts:
                shifted_black = [(x + dx, y + dy) for (x, y) in border_points]
                black_str = " ".join(f"{x},{y}" for (x, y) in shifted_black)
                polyline_b = doc.createElement('polyline')
                polyline_b.setAttribute('points', black_str)
                polyline_b.setAttribute('fill', 'none')
                polyline_b.setAttribute('stroke', 'black')
                polyline_b.setAttribute('stroke-width', '1')
                polyline_b.setAttribute('class', label)
                svg.appendChild(polyline_b)
                all_points.extend(shifted_black)

                shifted_red = [(x + dx, y + dy) for (x, y) in red_points]
                red_str = " ".join(f"{x},{y}" for (x, y) in shifted_red)
                polyline_r = doc.createElement('polyline')
                polyline_r.setAttribute('points', red_str)
                polyline_r.setAttribute('fill', 'none')
                polyline_r.setAttribute('stroke', 'red')
                polyline_r.setAttribute('stroke-width', '1')
                polyline_r.setAttribute('class', label)
                svg.appendChild(polyline_r)
                all_points.extend(shifted_red)

            # Green outline of largest object inside bounding box
            green_points = process_rank_0_object(data, obj)
            if green_points:
                green_str = " ".join(f"{x},{y}" for (x, y) in green_points)
                green_polyline = doc.createElement('polyline')
                green_polyline.setAttribute('points', green_str)
                green_polyline.setAttribute('fill', 'none')
                green_polyline.setAttribute('stroke', 'green')
                green_polyline.setAttribute('stroke-width', '1')
                green_polyline.setAttribute('class', 'rank_0_inner')
                svg.appendChild(green_polyline)
                all_points.extend(green_points)

        all_points.extend(border_points)
        all_points.extend(red_points)

    all_x = [x for x, y in all_points]
    all_y = [y for x, y in all_points]
    new_width = max(data['width'], max(all_x, default=0) + 1)
    new_height = max(data['height'], max(all_y, default=0) + 1)
    svg.setAttribute('width', str(new_width))
    svg.setAttribute('height', str(new_height))

    return doc.toxml()
