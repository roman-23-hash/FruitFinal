"""
app/gate.py
===========
Color-spectrum based guava gate using HSV analysis.

Guavas fall in the green-to-yellow color spectrum (Hue 25-85).
If less than 20% of the image pixels fall in this range,
the image is rejected as "Not a Guava" before model inference.

No external models or dependencies needed — pure OpenCV.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV range for guava color spectrum (green to yellow)
LOWER_GUAVA = np.array([25, 40, 40])
UPPER_GUAVA = np.array([85, 255, 255])

# Minimum percentage of pixels that must match guava color
DEFAULT_MATCH_PCT_THRESHOLD = 20.0


class GuavaGate:
    """
    Rejects non-guava images based on HSV color spectrum analysis.

    Guavas (green/yellow) have Hue values between 25-85.
    Red apples, oranges, bananas, random objects etc. will mostly
    fall outside this range and be rejected.
    """

    def __init__(self, threshold: float = DEFAULT_MATCH_PCT_THRESHOLD, enabled: bool = True):
        # threshold here is a PERCENTAGE (0-100), not 0-1
        self.threshold = threshold
        self.enabled = enabled
        self.available = True  # always available — pure OpenCV

        if not enabled:
            logger.info("Guava gate disabled — all images will pass through.")
        else:
            logger.info(
                "Guava color gate ready. Min guava-color pixel match: %.1f%%", threshold
            )

    def check_from_output(self, guava_guard_array) -> tuple:
        """
        Kept for API compatibility — not used in color-based gate.
        Always returns pass-through so predict.py can call this safely.
        The real check happens in check_color() called from predict.py.
        """
        return True, 1.0, "Color gate — see check_color()"

    def check_color(self, img_rgb: np.ndarray) -> tuple:
        """
        Check if image contains guava-colored pixels (green/yellow spectrum).

        Args:
            img_rgb: np.ndarray (H, W, 3) uint8 RGB image

        Returns:
            (is_guava: bool, match_pct: float, message: str)

            match_pct is 0-100 (percentage of pixels matching guava color)
        """
        if not self.enabled:
            return True, 100.0, "Gate disabled — all images pass"

        # Convert RGB → BGR → HSV (OpenCV uses BGR)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Create mask for guava color range
        guava_mask = cv2.inRange(hsv, LOWER_GUAVA, UPPER_GUAVA)

        # Calculate percentage of matching pixels
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        matching_pixels = np.count_nonzero(guava_mask)
        match_pct = (matching_pixels / total_pixels) * 100.0

        is_guava = match_pct >= self.threshold

        if is_guava:
            message = f"Guava confirmed ({match_pct:.1f}% green/yellow pixels)"
        else:
            message = (
                                f"Not a guava"

                # f"Not a guava — only {match_pct:.1f}% green/yellow pixels "
                # f"(need ≥{self.threshold:.0f}%)"
            )

        logger.info("Color gate: %s", message)
        return is_guava, match_pct, message