"""QR Code Decoder Module for QRShield.

Handles QR code detection and decoding from image files (PNG, JPG, JPEG),
byte streams, and NumPy arrays using OpenCV and Pyzbar.

Features:
- Dual-engine decoding: OpenCV QRCodeDetector + Pyzbar
- File format and file size validation
- Corrupted image detection
- Multi-QR code detection in a single image
- Payload type classification (HTTPS, HTTP, Dangerous URI schemes, Plain Text)
- Safe static extraction (never navigates, downloads, or executes embedded content)
"""

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import config

logger = logging.getLogger("qrshield.qr_decoder")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Check Pyzbar availability (handles Windows DLL loader errors gracefully)
PYZBAR_AVAILABLE = False
PYZBAR_ERROR_MSG = ""

try:
    # Attempt to import pyzbar
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError) as exc:
    PYZBAR_AVAILABLE = False
    PYZBAR_ERROR_MSG = str(exc)
    logger.info(f"Pyzbar native library not loaded ({exc}). Using OpenCV QRCodeDetector as primary engine.")

# Known dangerous URI schemes
DANGEROUS_SCHEMES = {
    "javascript",
    "data",
    "file",
    "vbscript",
    "blob",
    "about",
    "chrome",
    "ms-msdt",
    "search-ms",
    "shell",
    "jar",
    "filesystem",
}

# Recognized specialized/unsupported schemes
SPECIALIZED_SCHEMES = {
    "mailto",
    "tel",
    "smsto",
    "sms",
    "wifi",
    "geo",
    "market",
    "intent",
    "facetime",
    "skype",
    "ftp",
    "sftp",
    "ssh",
}


@dataclass
class DecodedQR:
    """Represents a single decoded QR code item."""

    raw_payload: str
    payload_type: str  # 'URL_HTTPS', 'URL_HTTP', 'DANGEROUS_SCHEME', 'SPECIALIZED_SCHEME', 'PLAIN_TEXT', 'EMPTY'
    is_url: bool
    scheme: Optional[str] = None
    bounding_box: Optional[List[Tuple[int, int]]] = None
    decoder_engine: str = "OpenCV"

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary representation."""
        return {
            "raw_payload": self.raw_payload,
            "payload_type": self.payload_type,
            "is_url": self.is_url,
            "scheme": self.scheme,
            "decoder_engine": self.decoder_engine,
        }


@dataclass
class QRScanResult:
    """Result of a QR code scan operation on an image."""

    success: bool
    qr_count: int
    items: List[DecodedQR] = field(default_factory=list)
    error_message: Optional[str] = None
    engine_used: str = "OpenCV"

    @property
    def primary_payload(self) -> Optional[str]:
        """Return raw payload of the first detected QR code, if any."""
        return self.items[0].raw_payload if self.items else None

    @property
    def primary_type(self) -> Optional[str]:
        """Return payload type of the first detected QR code, if any."""
        return self.items[0].payload_type if self.items else None


def classify_payload(payload: str) -> Tuple[str, bool, Optional[str]]:
    """Classify decoded payload into HTTP/HTTPS URL, Dangerous URI, Contact, Wi-Fi, Payment, or Plain Text.

    Does not send network requests or execute payload content.

    Args:
        payload: The raw decoded text from the QR code.

    Returns:
        Tuple[str, bool, Optional[str]]: (payload_type, is_url, scheme)
    """
    clean = (payload or "").strip()
    if not clean:
        return "EMPTY", False, None

    lower_payload = clean.lower()

    # 1. Contact Details: vCard and meCard
    if clean.upper().startswith("BEGIN:VCARD") or clean.upper().startswith("MECARD:"):
        return "SPECIALIZED_SCHEME", False, "vcard"

    # 2. Wi-Fi Configuration
    if clean.upper().startswith("WIFI:"):
        return "SPECIALIZED_SCHEME", False, "wifi"

    # 3. Cryptocurrency & Payment Payloads
    payment_schemes = ("upi", "bitcoin", "ethereum", "solana", "lightning", "bitcoincash")
    for ps in payment_schemes:
        if lower_payload.startswith(f"{ps}:") or lower_payload.startswith(f"{ps}://"):
            return "SPECIALIZED_SCHEME", False, ps

    if "paypal.me/" in lower_payload:
        # PayPal.me web payment link
        return "PAYMENT_PAYLOAD", True, "https" if lower_payload.startswith("https") else "http"

    # 4. Email, Phone, SMS, Geo
    if lower_payload.startswith("mailto:"):
        return "SPECIALIZED_SCHEME", False, "mailto"
    if lower_payload.startswith(("tel:", "sms:", "smsto:")):
        return "SPECIALIZED_SCHEME", False, lower_payload.split(":")[0]
    if lower_payload.startswith("geo:"):
        return "SPECIALIZED_SCHEME", False, "geo"

    # 5. Check if string starts with dangerous schemes
    for dangerous_scheme in DANGEROUS_SCHEMES:
        if lower_payload.startswith(f"{dangerous_scheme}:"):
            return "DANGEROUS_SCHEME", False, dangerous_scheme

    # 6. Check for HTTP / HTTPS URLs
    try:
        parsed = urlparse(clean)
        scheme = (parsed.scheme or "").lower()

        if scheme in ("http", "https"):
            # Ensure netloc is present or host-like structure exists
            if parsed.netloc or ("." in clean.split("/")[0]):
                ptype = "URL_HTTPS" if scheme == "https" else "URL_HTTP"
                return ptype, True, scheme
            return "PLAIN_TEXT", False, None

        if scheme in DANGEROUS_SCHEMES:
            return "DANGEROUS_SCHEME", False, scheme

        if scheme in SPECIALIZED_SCHEMES:
            return "SPECIALIZED_SCHEME", False, scheme

        if scheme and ":" in clean[:20] and not clean.startswith(" "):
            # Any other arbitrary URI scheme (e.g., customapp://)
            return "SPECIALIZED_SCHEME", False, scheme

    except Exception:
        pass

    # Handle domain without http/https prefix (e.g. "www.example.com/test")
    if lower_payload.startswith("www.") and "." in clean:
        return "URL_HTTP", True, "http"

    return "PLAIN_TEXT", False, None


class QRDecoder:
    """Production-grade QR code decoder using OpenCV and Pyzbar."""

    def __init__(self, max_size_mb: Optional[float] = None) -> None:
        self.max_size_mb = max_size_mb if max_size_mb is not None else getattr(config, "MAX_UPLOAD_SIZE_MB", 5.0)
        self.supported_extensions = set(getattr(config, "SUPPORTED_IMAGE_EXTENSIONS", [".png", ".jpg", ".jpeg"]))
        self.opencv_detector = cv2.QRCodeDetector()

    def validate_file(
        self,
        file_input: Union[bytes, io.BytesIO, str, Path, Any],
        filename: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[bytes]]:
        """Validate file format, size, and integrity.

        Args:
            file_input: File path, bytes, or file-like object.
            filename: Optional filename for extension checking.

        Returns:
            Tuple[bool, Optional[str], Optional[bytes]]: (Valid boolean, Error message, Extracted byte data).
        """
        # Determine filename and extract bytes
        inferred_name = filename
        raw_bytes: Optional[bytes] = None

        if hasattr(file_input, "name") and not inferred_name:
            inferred_name = file_input.name

        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not inferred_name:
                inferred_name = path.name
            if not path.exists():
                return False, f"File does not exist: {file_input}", None
            try:
                raw_bytes = path.read_bytes()
            except Exception as e:
                return False, f"Unable to read file: {e}", None

        elif isinstance(file_input, bytes):
            raw_bytes = file_input

        elif hasattr(file_input, "read") or hasattr(file_input, "getvalue"):
            try:
                if hasattr(file_input, "getvalue"):
                    raw_bytes = file_input.getvalue()
                else:
                    raw_bytes = file_input.read()
                    if hasattr(file_input, "seek"):
                        file_input.seek(0)
            except Exception as e:
                return False, f"Unable to read file stream: {e}", None
        else:
            return False, "Unsupported file input type.", None

        if not raw_bytes or len(raw_bytes) == 0:
            return False, "Uploaded file is empty (0 bytes).", None

        # Check file extension if filename is available
        if inferred_name:
            ext = Path(inferred_name).suffix.lower()
            if ext not in self.supported_extensions:
                return (
                    False,
                    f"Unsupported image format '{ext}'. Supported formats: PNG, JPG, JPEG.",
                    None,
                )

        # Check file size limit
        file_size_mb = len(raw_bytes) / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            return (
                False,
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit ({self.max_size_mb:.1f} MB).",
                None,
            )

        # Verify image integrity via PIL and OpenCV
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                pil_img.verify()
        except (UnidentifiedImageError, OSError, Exception):
            return False, "Corrupted image file: could not parse image header.", None

        # Confirm OpenCV can decode the image buffer
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None or cv_img.size == 0:
            return False, "Corrupted or unreadable image data.", None

        return True, None, raw_bytes

    def load_cv_image(
        self,
        image_input: Union[bytes, io.BytesIO, str, Path, np.ndarray, Image.Image, Any],
        filename: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Load and convert an image input into a valid BGR OpenCV NumPy array.

        Args:
            image_input: Path, bytes, PIL Image, Streamlit UploadedFile, or NumPy array.
            filename: Optional filename for extension verification.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: (OpenCV image matrix or None, Error message or None).
        """
        # If already a numpy array
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return None, "Empty image array provided."
            # Convert RGBA/RGB to BGR if needed
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR), None
            elif len(image_input.shape) == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR), None
                return image_input, None
            return None, "Invalid image array dimensions."

        # If a PIL Image
        if isinstance(image_input, Image.Image):
            rgb = image_input.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR), None

        # Validate bytes or file-like object
        valid, err_msg, byte_data = self.validate_file(image_input, filename=filename)
        if not valid or byte_data is None:
            return None, err_msg

        np_arr = np.frombuffer(byte_data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            return None, "Failed to decode image buffer."
        return cv_img, None

    def _decode_with_opencv(self, cv_img: np.ndarray) -> List[DecodedQR]:
        """Decode QR codes using OpenCV's QRCodeDetector."""
        results: List[DecodedQR] = []
        try:
            # First try multi-QR detection
            retval, decoded_info, points, _ = self.opencv_detector.detectAndDecodeMulti(cv_img)
            if retval and decoded_info:
                for idx, text in enumerate(decoded_info):
                    if text and text.strip():
                        ptype, is_url, scheme = classify_payload(text)
                        bbox = None
                        if points is not None and len(points) > idx:
                            bbox = [(int(p[0]), int(p[1])) for p in points[idx]]
                        results.append(
                            DecodedQR(
                                raw_payload=text,
                                payload_type=ptype,
                                is_url=is_url,
                                scheme=scheme,
                                bounding_box=bbox,
                                decoder_engine="OpenCV",
                            )
                        )
                if results:
                    return results

            # Fallback to single QR detector if detectAndDecodeMulti returned empty
            val, pts, _ = self.opencv_detector.detectAndDecode(cv_img)
            if val and val.strip():
                ptype, is_url, scheme = classify_payload(val)
                bbox = [(int(p[0]), int(p[1])) for p in pts[0]] if pts is not None and len(pts) > 0 else None
                results.append(
                    DecodedQR(
                        raw_payload=val,
                        payload_type=ptype,
                        is_url=is_url,
                        scheme=scheme,
                        bounding_box=bbox,
                        decoder_engine="OpenCV",
                    )
                )
        except Exception as exc:
            logger.debug(f"OpenCV decode exception: {exc}")

        return results

    def _decode_with_pyzbar(self, cv_img: np.ndarray) -> List[DecodedQR]:
        """Decode QR codes using Pyzbar if available."""
        if not PYZBAR_AVAILABLE:
            return []

        results: List[DecodedQR] = []
        try:
            # Pyzbar works with BGR or grayscale numpy arrays
            decoded_objects = pyzbar.decode(cv_img, symbols=[pyzbar.ZBarSymbol.QRCODE])
            for obj in decoded_objects:
                try:
                    text = obj.data.decode("utf-8", errors="replace")
                except Exception:
                    text = str(obj.data)

                if text and text.strip():
                    ptype, is_url, scheme = classify_payload(text)
                    bbox = None
                    if hasattr(obj, "polygon") and obj.polygon:
                        bbox = [(p.x, p.y) for p in obj.polygon]
                    results.append(
                        DecodedQR(
                            raw_payload=text,
                            payload_type=ptype,
                            is_url=is_url,
                            scheme=scheme,
                            bounding_box=bbox,
                            decoder_engine="Pyzbar",
                        )
                    )
        except Exception as exc:
            logger.debug(f"Pyzbar decode exception: {exc}")

        return results

    def decode(
        self,
        image_input: Union[bytes, io.BytesIO, str, Path, np.ndarray, Image.Image, Any],
        filename: Optional[str] = None,
    ) -> QRScanResult:
        """Decode QR codes from an image input without navigating to or executing payloads.

        Args:
            image_input: Uploaded file, file path, bytes, or image array.
            filename: Optional filename for validation.

        Returns:
            QRScanResult: Scan outcomes including all decoded items and error messages.
        """
        cv_img, err_msg = self.load_cv_image(image_input, filename=filename)
        if cv_img is None:
            return QRScanResult(success=False, qr_count=0, items=[], error_message=err_msg)

        # 1. Try Pyzbar if available
        pyzbar_items = self._decode_with_pyzbar(cv_img)
        
        # 2. Try OpenCV QRCodeDetector
        opencv_items = self._decode_with_opencv(cv_img)

        # Merge results, prioritizing non-empty items and de-duplicating by raw text
        merged_items: List[DecodedQR] = []
        seen_payloads = set()

        for item in pyzbar_items + opencv_items:
            if item.raw_payload not in seen_payloads:
                seen_payloads.add(item.raw_payload)
                merged_items.append(item)

        if not merged_items:
            return QRScanResult(
                success=True,
                qr_count=0,
                items=[],
                error_message="No QR code found in the image. Ensure the image is clear and well-lit.",
                engine_used="Pyzbar+OpenCV" if PYZBAR_AVAILABLE else "OpenCV",
            )

        primary_engine = merged_items[0].decoder_engine
        return QRScanResult(
            success=True,
            qr_count=len(merged_items),
            items=merged_items,
            error_message=None,
            engine_used=primary_engine,
        )


# Global singleton instance
_decoder_instance = QRDecoder()


def decode_qr(
    image_input: Union[bytes, io.BytesIO, str, Path, np.ndarray, Image.Image, Any],
    filename: Optional[str] = None,
) -> QRScanResult:
    """Convenience helper to decode QR codes using the default QRDecoder."""
    return _decoder_instance.decode(image_input, filename=filename)


def get_decoder_status() -> Dict[str, Any]:
    """Retrieve operational status of the QR decoding engines."""
    return {
        "opencv_available": True,
        "pyzbar_available": PYZBAR_AVAILABLE,
        "pyzbar_error": PYZBAR_ERROR_MSG if not PYZBAR_AVAILABLE else None,
        "primary_engine": "Pyzbar & OpenCV (Dual Engine)" if PYZBAR_AVAILABLE else "OpenCV (Native)",
    }
