#!/usr/bin/env python3
"""
scripts/test_predict.py
=======================
Send a local image to the Fruit Ripeness API and print the JSON result.

Usage:
  python scripts/test_predict.py path/to/fruit.jpg
  python scripts/test_predict.py path/to/fruit.jpg --url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test the /predict endpoint")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: File not found: {img_path}")
        sys.exit(1)

    endpoint = f"{args.url.rstrip('/')}/predict"
    print(f"Uploading '{img_path}' to {endpoint} …")

    with open(img_path, "rb") as fh:
        response = requests.post(
            endpoint,
            files={"file": (img_path.name, fh, "image/jpeg")},
            timeout=60,
        )

    print(f"\nHTTP {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)

    if response.status_code == 200 and data.get("success"):
        top = data["predictions"][0]
        pct = top["confidence"] * 100
        print(f"\n✅  Top prediction: {top['label']} ({pct:.1f}% confidence)")
        print(f"   Processing time: {data['meta']['processing_time_ms']:.1f} ms")
    else:
        print("\n❌  Prediction failed. See response above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
