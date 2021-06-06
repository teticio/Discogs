import io
import cv2 as cv
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

app = FastAPI()

size = 300
sigma = 0.33
rho_step = 1
new_size = 150
max_lines = 10
threshold = 50
score_thickness = 1
theta_step = np.pi / 180
angle_tolerance = 10 * np.pi / 180


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [(x0, y0)]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def score_line(edges, point_1, point_2):
    return np.tensordot(
        edges,
        cv.line(np.zeros(edges.shape), point_1, point_2, (255, ),
                score_thickness, cv.LINE_AA)) / size


def score_square(top_left, top_right, bottom_right, bottom_left):
    return (min(
        top_right[0] - top_left[0], bottom_right[1] - top_right[1],
        bottom_right[0] - bottom_left[0], bottom_left[1] - top_left[1]) -
        abs(top_left[1] - top_right[1]) - abs(top_right[0] - bottom_right[0]) -
        abs(bottom_right[1] - bottom_left[1]) -
        abs(bottom_left[0] - top_left[0]) / size)


def crop(image):
    nparr = np.fromstring(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)

    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv.Canny(img, lower, upper)

    lines = cv.HoughLines(edges, rho_step, theta_step, threshold)
    horizontal_lines = [
        line for line in lines
        if np.pi / 2 - angle_tolerance < line[0][1] < np.pi / 2 +
        angle_tolerance or 3 * np.pi / 2 -
        angle_tolerance < line[0][1] < 3 * np.pi / 2 + angle_tolerance
    ]
    vertical_lines = [
        line for line in lines
        if line[0][1] < angle_tolerance or line[0][1] > 2 * np.pi -
        angle_tolerance or np.pi - angle_tolerance < line[0][1] < np.pi +
        angle_tolerance
    ]

    intersections = segmented_intersections(
        [horizontal_lines[:max_lines], vertical_lines[:max_lines]])

    top_left_corners = set(
        point[0] for point in intersections
        if 0 < point[0][0] < size / 2 and 0 < point[0][1] < size / 2)
    top_right_corners = set(
        point[0] for point in intersections
        if size > point[0][0] > size / 2 and 0 < point[0][1] < size / 2)
    bottom_left_corners = set(
        point[0] for point in intersections
        if 0 < point[0][0] < size / 2 and size > point[0][1] > size / 2)
    bottom_right_corners = set(
        point[0] for point in intersections
        if size > point[0][0] > size / 2 and size > point[0][1] > size / 2)

    top_lines = sorted([(score_line(edges, top_left_corner, top_right_corner),
                         top_left_corner, top_right_corner)
                        for top_left_corner in top_left_corners
                        for top_right_corner in top_right_corners],
                       reverse=True)

    right_lines = sorted(
        [(score_line(edges, top_right_corner, bottom_right_corner),
          top_right_corner, bottom_right_corner)
         for top_right_corner in top_right_corners
         for bottom_right_corner in bottom_right_corners],
        reverse=True)

    bottom_lines = sorted(
        [(score_line(edges, bottom_right_corner, bottom_left_corner),
          bottom_right_corner, bottom_left_corner)
         for bottom_right_corner in bottom_right_corners
         for bottom_left_corner in bottom_left_corners],
        reverse=True)

    left_lines = sorted(
        [(score_line(edges, bottom_left_corner,
                     top_left_corner), bottom_left_corner, top_left_corner)
         for bottom_left_corner in bottom_left_corners
         for top_left_corner in top_left_corners],
        reverse=True)

    quadrilaterals = sorted(
        [(score_square(top_line[1], right_line[1], bottom_line[1], left_line[1]) *
        (top_line[0] + right_line[0] + bottom_line[0] + left_line[0]),
        top_line[1], right_line[1], bottom_line[1], left_line[1])
        for top_line in top_lines
        for right_line in right_lines if right_line[1] == top_line[2]
        for bottom_line in bottom_lines if bottom_line[1] == right_line[2]
        for left_line in left_lines
        if left_line[1] == bottom_line[2] and left_line[2] == top_line[1]],
        reverse=True)

    if len(quadrilaterals) < 1:
        return None

    M = cv.getPerspectiveTransform(
        np.float32([
            [quadrilaterals[0][1][0], quadrilaterals[0][1][1]],
            [quadrilaterals[0][2][0], quadrilaterals[0][2][1]],
            [quadrilaterals[0][3][0], quadrilaterals[0][3][1]],
            [quadrilaterals[0][4][0], quadrilaterals[0][4][1]],
        ]),
        np.float32([[0, 0], [new_size, 0], [new_size, new_size], [0,
                                                                  new_size]]))

    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
    new_img = cv.warpPerspective(img, M, (size, size))[:new_size, :new_size, :]
    is_success, buffer = cv.imencode(".jpg", new_img)
    io_buf = io.BytesIO(buffer)
    return io_buf


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    image = await file.read()
    cropped = await run_in_threadpool(crop, image)
    if cropped is None:
        cropped = image
    return StreamingResponse(cropped, media_type="image/jpeg")