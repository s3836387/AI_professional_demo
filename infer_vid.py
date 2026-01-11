import base64
import json
import logging
import requests
import cv2
import time
import numpy as np
import threading

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava-llama3:8b"

VIDEO_PATH = "accident.mp4"

INFER_INTERVAL_SEC = 2.0
RESIZE_FOR_AI = (640, 360)

TARGET_W, TARGET_H = 640, 480
PANEL_WIDTH = 420
AI_PREVIEW_H = 180

WINDOW_NAME = "Road Anomaly Detection (Polygon + VLM)"

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# PROMPT
# =========================

PROMPT = """
INPUT DESCRIPTION:
You are given an image where ONLY the road region is visible.
Black areas represent regions outside the road and MUST BE IGNORED.
The visible area may be a masked or cropped region of interest focused on the roadway.
Do not assume black regions indicate darkness, night, or empty roads.

You are an expert road-incident detection system monitoring highways using CCTV.

Your task:
Determine whether the road is BLOCKED or PARTIALLY BLOCKED by any abnormal or hazardous object or event.

IMPORTANT DEFINITIONS:
- A FALLEN TREE means a tree, large branch, or trunk lying across the roadway or shoulder that obstructs traffic.
- FLOOD means standing or flowing water covering any drivable lane.
- ACCIDENT means crashed, overturned, or severely damaged vehicles affecting traffic flow.
- Even if no vehicles are visible, an object blocking the road IS an anomaly.

FOCUS CAREFULLY ON:
- Objects lying ACROSS lanes
- Natural debris after storms (trees, branches, mud)
- Water covering road markings
- Vehicles stopped in abnormal positions

If ANY part of the road is obstructed, anomaly MUST be true.
If you are uncertain, but a blockage is plausible, set anomaly=true with lower confidence.

OUTPUT ONLY VALID JSON. NO EXPLANATION.

JSON FORMAT:
{
  "anomaly": true | false,
  "anomaly_type": "fallen_tree" | "flood" | "accident" | "vehicle" | "construction" | "debris" | "landslide" | "fire" | "unknown" | "none",
  "confidence": 0.0-1.0
}
"""


# =========================
# SHARED STATE
# =========================

latest_frame = None
latest_ai_input = None
latest_result = None
latest_latency = None

lock = threading.Lock()
stop_flag = False
ROAD_POLYGON = None

# =========================
# HELPERS
# =========================

def encode_frame_b64(frame):
    frame = cv2.resize(frame, RESIZE_FOR_AI)
    ok, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buffer).decode("utf-8")

def safe_json_parse(text):
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s:e+1])
        raise ValueError("Invalid JSON")

def apply_polygon_mask(frame, polygon):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

# =========================
# MODEL WARM-UP
# =========================

def warmup_model():
    logging.info("Warming up model...")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [encode_frame_b64(dummy)],
        "stream": False,
        "options": {"temperature": 0}
    }

    t0 = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    logging.info("Warm-up done in %.2fs", time.perf_counter() - t0)

# =========================
# POLYGON DRAWING
# =========================

def draw_polygon(frame):
    points = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            logging.info("Point added: %s", (x, y))

    cv2.namedWindow("Draw Road Polygon")
    cv2.setMouseCallback("Draw Road Polygon", mouse_cb)

    logging.info("Draw polygon: click points | ENTER=confirm | BACKSPACE=undo")

    while True:
        vis = frame.copy()

        for p in points:
            cv2.circle(vis, p, 5, (0, 255, 255), -1)

        if len(points) > 1:
            cv2.polylines(
                vis,
                [np.array(points, np.int32)],
                False,
                (0, 255, 255),
                2
            )

        cv2.imshow("Draw Road Polygon", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(points) >= 3:
            cv2.destroyWindow("Draw Road Polygon")
            return np.array(points, dtype=np.int32)

        elif key == 8 and points:
            points.pop()

        elif key == 27:
            cv2.destroyWindow("Draw Road Polygon")
            raise SystemExit("Polygon drawing cancelled")

# =========================
# ASYNC INFERENCE WORKER
# =========================

def infer_worker():
    global latest_result, latest_latency, latest_ai_input

    logging.info("Inference worker started")
    next_run = time.monotonic()

    while not stop_flag:
        if time.monotonic() < next_run:
            time.sleep(0.05)
            continue

        next_run += INFER_INTERVAL_SEC

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        try:
            masked = apply_polygon_mask(frame, ROAD_POLYGON)

            with lock:
                latest_ai_input = masked.copy()

            img_b64 = encode_frame_b64(masked)

            payload = {
                "model": MODEL_NAME,
                "prompt": PROMPT,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0}
            }

            t0 = time.perf_counter()
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()

            result = safe_json_parse(r.json()["response"])
            latency = time.perf_counter() - t0

            with lock:
                latest_result = result
                latest_latency = latency

            logging.info("AI updated in %.2fs → %s", latency, result)

        except Exception as e:
            logging.error("Inference error: %s", e)

# =========================
# UI PANEL
# =========================

def render_panel(h, w, result, latency, ai_input):
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    cv2.putText(panel, "AI OUTPUT", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y = 70
    color = (0, 255, 0)

    if result:
        lines = [
            f"Anomaly    : {result['anomaly']}",
            f"Type       : {result['anomaly_type']}",
            f"Confidence : {result['confidence']:.2f}",
            f"Latency    : {latency:.2f}s"
        ]
    else:
        lines = ["Waiting for inference..."]
        color = (0, 180, 255)

    for line in lines:
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 30

    # AI INPUT PREVIEW
    preview_top = h - AI_PREVIEW_H - 10
    cv2.putText(panel, "MODEL INPUT", (20, preview_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if ai_input is not None:
        preview = cv2.resize(ai_input, (w - 20, AI_PREVIEW_H))
        panel[preview_top:preview_top + AI_PREVIEW_H, 10:10 + preview.shape[1]] = preview
    else:
        cv2.putText(panel, "No input yet",
                    (20, preview_top + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    return panel

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    t = threading.Thread(target=infer_worker, daemon=True)
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        with lock:
            latest_frame = frame

        vis = frame.copy()
        cv2.polylines(vis, [ROAD_POLYGON], True, (0, 255, 255), 2)

        with lock:
            result = latest_result
            latency = latest_latency
            ai_input = latest_ai_input

        panel = render_panel(TARGET_H, PANEL_WIDTH, result, latency, ai_input)

        canvas = np.zeros((TARGET_H, TARGET_W + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[:, :TARGET_W] = vis
        canvas[:, TARGET_W:] = panel

        cv2.imshow(WINDOW_NAME, canvas)

        if cv2.waitKey(delay) & 0xFF == 27:
            break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    logging.info("Starting system")

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Cannot read video")

    first_frame = cv2.resize(first_frame, (TARGET_W, TARGET_H))

    ROAD_POLYGON = draw_polygon(first_frame)
    logging.info("Polygon confirmed: %s", ROAD_POLYGON.tolist())

    warmup_model()
    run_pipeline()
