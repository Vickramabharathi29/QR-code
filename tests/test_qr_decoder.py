"""Unit tests for modules/qr_decoder.py.

Generates test QR images locally in-memory using the qrcode library
without accessing external networks or websites.
"""

import io
from typing import Tuple

import numpy as np
import pytest
import qrcode
from PIL import Image

from modules.qr_decoder import (
    QRDecoder,
    classify_payload,
    decode_qr,
    get_decoder_status,
)


def create_qr_bytes(payload: str, img_format: str = "PNG") -> bytes:
    """Helper to generate a QR code image as bytes in-memory."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


def create_blank_image_bytes(width: int = 250, height: int = 250) -> bytes:
    """Helper to create a solid white image containing no QR code."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_multi_qr_bytes(payload1: str, payload2: str) -> bytes:
    """Helper to stitch two QR codes horizontally into one image."""
    img1 = qrcode.make(payload1).convert("RGB")
    img2 = qrcode.make(payload2).convert("RGB")
    
    total_width = img1.width + img2.width + 60
    total_height = max(img1.height, img2.height) + 40
    
    combined = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    combined.paste(img1, (20, 20))
    combined.paste(img2, (img1.width + 40, 20))
    
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: Valid QR Code with HTTPS URL
# ---------------------------------------------------------------------------
def test_decode_valid_https_url():
    """Verify decoding of a valid QR code containing an HTTPS URL."""
    test_url = "https://secure-portal.example.org/login?session=abc"
    image_bytes = create_qr_bytes(test_url)

    result = decode_qr(image_bytes, filename="test_qr.png")

    assert result.success is True
    assert result.qr_count >= 1
    assert result.error_message is None

    primary = result.items[0]
    assert primary.raw_payload == test_url
    assert primary.is_url is True
    assert primary.payload_type == "URL_HTTPS"
    assert primary.scheme == "https"


# ---------------------------------------------------------------------------
# Test 2: Valid QR Code with Plain Text
# ---------------------------------------------------------------------------
def test_decode_plain_text():
    """Verify decoding of a QR code containing plain non-URL text."""
    plain_text = "Conference Room A Wi-Fi Guest Pass: 987654"
    image_bytes = create_qr_bytes(plain_text)

    result = decode_qr(image_bytes, filename="plain_text.jpg")

    assert result.success is True
    assert result.qr_count >= 1
    primary = result.items[0]
    assert primary.raw_payload == plain_text
    assert primary.is_url is False
    assert primary.payload_type == "PLAIN_TEXT"


# ---------------------------------------------------------------------------
# Test 3: Dangerous URI Schemes (javascript:, data:, file:)
# ---------------------------------------------------------------------------
def test_classify_dangerous_uri_schemes():
    """Verify detection of dangerous URI schemes without executing them."""
    js_payload = "javascript:alert('XSS_ATTACK')"
    js_bytes = create_qr_bytes(js_payload)
    result_js = decode_qr(js_bytes, filename="js_exploit.png")
    assert result_js.success is True
    assert result_js.items[0].payload_type == "DANGEROUS_SCHEME"
    assert result_js.items[0].is_url is False

    # Direct classification checks
    ptype, is_url, scheme = classify_payload("data:text/html;base64,PHNjcmlwdD4=")
    assert ptype == "DANGEROUS_SCHEME"
    assert is_url is False

    ptype, is_url, scheme = classify_payload("file:///etc/shadow")
    assert ptype == "DANGEROUS_SCHEME"
    assert is_url is False


# ---------------------------------------------------------------------------
# Test 4: Specialized URI Schemes (WIFI, tel, mailto)
# ---------------------------------------------------------------------------
def test_classify_specialized_schemes():
    """Verify detection of specialized non-HTTP schemes."""
    ptype, is_url, scheme = classify_payload("WIFI:S:OfficeNet;T:WPA;P:secret123;;")
    assert ptype == "SPECIALIZED_SCHEME"
    assert is_url is False

    ptype, is_url, scheme = classify_payload("tel:+18005550199")
    assert ptype == "SPECIALIZED_SCHEME"
    assert is_url is False

    ptype, is_url, scheme = classify_payload("mailto:security@example.com")
    assert ptype == "SPECIALIZED_SCHEME"
    assert is_url is False


# ---------------------------------------------------------------------------
# Test 5: Image Without a QR Code
# ---------------------------------------------------------------------------
def test_image_without_qr_code():
    """Verify handling of a valid image that contains no QR code."""
    blank_bytes = create_blank_image_bytes(300, 300)
    result = decode_qr(blank_bytes, filename="blank.png")

    assert result.success is True
    assert result.qr_count == 0
    assert len(result.items) == 0
    assert "No QR code found" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Test 6: Invalid / Corrupted Image
# ---------------------------------------------------------------------------
def test_invalid_corrupted_image():
    """Verify handling of corrupted or non-image byte streams."""
    corrupted_data = b"This is not a real PNG or JPEG file. Just garbage bytes."
    result = decode_qr(corrupted_data, filename="corrupt.png")

    assert result.success is False
    assert result.qr_count == 0
    assert "Corrupted" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Test 7: Empty File
# ---------------------------------------------------------------------------
def test_empty_file():
    """Verify handling of an empty 0-byte file upload."""
    result = decode_qr(b"", filename="empty.png")
    assert result.success is False
    assert "empty" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# Test 8: Unsupported File Extension
# ---------------------------------------------------------------------------
def test_unsupported_file_extension():
    """Verify rejection of unsupported extensions (e.g., .gif, .pdf, .exe)."""
    valid_qr_bytes = create_qr_bytes("https://example.com")
    result = decode_qr(valid_qr_bytes, filename="document.pdf")

    assert result.success is False
    assert "Unsupported image format" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Test 9: File Size Limit Exceeded
# ---------------------------------------------------------------------------
def test_oversized_file():
    """Verify enforcement of file size limits."""
    decoder = QRDecoder(max_size_mb=0.0001)  # Set tiny limit (~104 bytes)
    valid_qr_bytes = create_qr_bytes("https://example.com")  # Typically 500-600 bytes

    result = decoder.decode(valid_qr_bytes, filename="sample.png")
    assert result.success is False
    assert "exceeds maximum allowed limit" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Test 10: Multiple QR Codes in One Image
# ---------------------------------------------------------------------------
def test_multiple_qr_codes():
    """Verify detection when multiple QR codes are present in a single image."""
    payload1 = "https://service-one.example.com"
    payload2 = "https://service-two.example.com"
    multi_bytes = create_multi_qr_bytes(payload1, payload2)

    result = decode_qr(multi_bytes, filename="multi.png")

    assert result.success is True
    # Should detect both QR codes (or at least 1 without crashing)
    assert result.qr_count >= 1
    found_payloads = [item.raw_payload for item in result.items]
    assert any(p in (payload1, payload2) for p in found_payloads)


# ---------------------------------------------------------------------------
# Test 11: Decoder Engine Status Check
# ---------------------------------------------------------------------------
def test_get_decoder_status():
    """Verify decoder status reports operational engines."""
    status = get_decoder_status()
    assert "opencv_available" in status
    assert status["opencv_available"] is True
    assert "pyzbar_available" in status
    assert "primary_engine" in status
