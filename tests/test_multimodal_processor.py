from pathlib import Path

from PIL import Image, ImageDraw

from multimodal_processor import MultimodalProcessor


def test_image_ocr_uses_preprocessed_screenshot_fallback(tmp_path, monkeypatch):
    image_path = tmp_path / "screenshot.png"
    img = Image.new("RGB", (900, 220), "#202124")
    draw = ImageDraw.Draw(img)
    draw.text((30, 50), "Screenshot test text 12345", fill="#e8eaed")
    img.save(image_path)

    processor = MultimodalProcessor()
    monkeypatch.setattr(processor, "_ollama_vision", lambda path, errors: "")

    seen_paths: list[Path] = []

    def fake_rapidocr(path, errors, label="RapidOCR"):
        seen_paths.append(Path(path))
        if label == "RapidOCR preprocessed #1":
            return "Screenshot test text 12345"
        errors.append(f"{label} ran but found no readable text")
        return ""

    monkeypatch.setattr(processor, "_rapidocr_image", fake_rapidocr)

    text = processor._extract_image(image_path)

    assert text == "Screenshot test text 12345"
    assert len(seen_paths) >= 2
    assert seen_paths[0] == image_path
    assert not any(path.exists() for path in seen_paths[1:])


def test_preprocess_image_for_ocr_creates_multiple_variants(tmp_path):
    image_path = tmp_path / "screenshot.png"
    img = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "Small UI text", fill="black")
    img.save(image_path)

    processor = MultimodalProcessor()
    errors: list[str] = []
    variants = processor._preprocess_image_for_ocr(image_path, errors)

    try:
        assert not errors
        assert len(variants) == 4
        assert all(path.exists() for path in variants)
    finally:
        for path in variants:
            path.unlink(missing_ok=True)


def test_video_uses_frame_ocr_when_audio_is_unavailable(monkeypatch, tmp_path):
    video_path = tmp_path / "screen_recording.mp4"
    video_path.write_bytes(b"not a real video; helpers are mocked")

    processor = MultimodalProcessor()
    monkeypatch.setattr(processor, "_get_video_duration", lambda path: 5.0)
    monkeypatch.setattr(
        processor,
        "_video_to_audio",
        lambda path: (_ for _ in ()).throw(RuntimeError("no audio track")),
    )
    monkeypatch.setattr(
        processor,
        "_sample_frame_ocr",
        lambda path, duration: "Visible button label 123",
    )

    text = processor._extract_video(video_path)

    assert "Visible frame text:" in text
    assert "Visible button label 123" in text
    assert "no audio track" not in text


def test_video_raises_when_audio_and_frames_fail(monkeypatch, tmp_path):
    video_path = tmp_path / "empty_video.mp4"
    video_path.write_bytes(b"not a real video; helpers are mocked")

    processor = MultimodalProcessor()
    monkeypatch.setattr(processor, "_get_video_duration", lambda path: 5.0)
    monkeypatch.setattr(
        processor,
        "_video_to_audio",
        lambda path: (_ for _ in ()).throw(RuntimeError("no audio track")),
    )
    monkeypatch.setattr(processor, "_sample_frame_ocr", lambda path, duration: "")

    try:
        processor._extract_video(video_path)
    except RuntimeError as exc:
        assert "No searchable video content" in str(exc)
        assert "no audio track" in str(exc)
    else:
        raise AssertionError("Expected video extraction to fail")
