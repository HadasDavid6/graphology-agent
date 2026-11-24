"""Image analyzer using LLM vision capabilities."""

import base64
from typing import Optional


def encode_image_to_base64(image_data: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_data).decode("utf-8")


def infer_media_type(filename: str) -> str:
    """Infer media type from filename."""
    ext = filename.lower().split(".")[-1]
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


class ImageAnalyzer:
    """Analyzes images using LLM vision capabilities."""

    @staticmethod
    def prepare_image_message(image_data: bytes, filename: str, prompt: str) -> dict:
        """
        Prepare image message for LLM.

        Args:
            image_data: Raw image bytes
            filename: Original filename (for media type inference)
            prompt: Text prompt to accompany image

        Returns:
            Message dict with image content
        """
        b64 = encode_image_to_base64(image_data)
        media_type = infer_media_type(filename)

        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }

    @staticmethod
    def get_description(prompt: str = "") -> str:
        """Get default system context for handwriting analysis."""
        base = (
            "You are a professional graphology expert analyzing handwritten text. "
            "Based on the handwriting image provided, analyze and describe: "
            "1. Writing style and characteristics\n"
            "2. Personality traits indicated by the handwriting\n"
            "3. Pressure and size patterns\n"
            "4. Slant and alignment\n"
            "Ground your analysis ONLY in observable features from the image. "
            "Do NOT make medical, psychological, or diagnostic claims."
        )
        if prompt:
            return f"{base}\n\nUser's specific question: {prompt}"
        return base
