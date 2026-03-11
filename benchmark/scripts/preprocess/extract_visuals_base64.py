import json
import requests
import base64
import os
import argparse
import cv2
import numpy as np


def image_to_png_base64(img_source):
    """
    Process any image (local path or URL) and convert to PNG format base64 encoding

    Args:
        img_source: Image source, can be local path or URL

    Returns:
        tuple: (base64_string, success_flag)
               Returns ("data:image/png;base64,xxx", True) on success
               Returns ({"error": "error message"}, False) on failure
    """
    try:
        # 1. Get image data
        if img_source.startswith("http"):
            # Download image from URL
            resp = requests.get(img_source, timeout=30, verify=False)
            resp.raise_for_status()
            img_data = np.frombuffer(resp.content, np.uint8)
        else:
            # Read from local file
            with open(img_source, "rb") as f:
                img_data = np.frombuffer(f.read(), np.uint8)

        # 2. Decode image using OpenCV
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("OpenCV failed to decode image")

        # 3. Handle different image formats
        # Convert grayscale to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Convert RGBA to BGR (with white background)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            bgr = img[:, :, :3]
            white_bg = np.ones_like(bgr) * 255
            img = (
                bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])
            ).astype(np.uint8)
        # BGR format (3 channels) use directly
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass

        # 4. Encode to PNG format
        success, buffer = cv2.imencode(".png", img)

        if not success:
            raise ValueError("Failed to encode image to PNG")

        # 5. Convert to base64
        png_base64 = base64.b64encode(buffer).decode("utf-8")
        result = f"data:image/png;base64,{png_base64}"

        return result, True

    except Exception as e:
        print(f"❌ {str(e)}")
        return {"error": str(e)}, False


def extract_visuals_base64(report_root_dir, query_id, eval_system_name):
    VISUAL_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/visuals.json"
    REPORT_PATH = f"{report_root_dir}/{eval_system_name}/{query_id}/report.md"
    UPDATE_REPORT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/report_updated.md"
    )
    OUTPUT_PATH = (
        f"{report_root_dir}/{eval_system_name}/{query_id}/visuals_with_base64.json"
    )

    try:
        with open(VISUAL_PATH, "r", encoding="utf-8") as f:
            visuals = json.load(f)
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}")
        return

    for item in visuals:
        content = item.get("content", "")
        if not content:
            item["img_base64"] = {"error": "no content found"}
            continue

        # Handle image path (URL or local path)
        if content.startswith("http"):
            img_source = content
        else:
            img_source = os.path.join(os.path.dirname(VISUAL_PATH), content)

        # Convert to PNG format base64
        img_base64, success = image_to_png_base64(img_source)

        if success:
            item["img_base64"] = img_base64
        else:
            # Conversion failed, remove image reference from report
            item["img_base64"] = img_base64
            report_content = report_content.replace(content, "")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(visuals, f, ensure_ascii=False, indent=2)
    with open(UPDATE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(
        f"✅ Visual elements Base64 encoding completed, results saved to {OUTPUT_PATH}"
    )

    return visuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root_dir",
        type=str,
        required=True,
        help="Root directory path for reports",
    )
    parser.add_argument(
        "--query_id", type=str, required=True, help="Evaluation report ID"
    )
    parser.add_argument(
        "--eval_system_name", type=str, required=True, help="Evaluation system name"
    )
    args = parser.parse_args()
    extract_visuals_base64(args.report_root_dir, args.query_id, args.eval_system_name)
