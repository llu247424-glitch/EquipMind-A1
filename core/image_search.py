from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import IMAGE_DIR
from .retriever import expand_query


@dataclass
class ImageCase:
    case_id: str
    device: str
    fault: str
    image_path: str
    description: str
    suggestion: str
    hash: str
    difference_hash: str
    color_histogram: list[float]


def average_hash(img: Image.Image, size: int = 8) -> str:
    gray = img.convert("L").resize((size, size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ''.join('1' if p > avg else '0' for p in pixels)
    return f"{int(bits, 2):0{size*size//4}x}"


def difference_hash(img: Image.Image, size: int = 8) -> str:
    gray = img.convert("L").resize((size + 1, size))
    pixels = list(gray.getdata())
    bits = []
    for y in range(size):
        row = pixels[y * (size + 1):(y + 1) * (size + 1)]
        bits.extend("1" if row[x] > row[x + 1] else "0" for x in range(size))
    return f"{int(''.join(bits), 2):0{size*size//4}x}"


def compact_color_histogram(img: Image.Image, bins: int = 16) -> list[float]:
    rgb = img.convert("RGB").resize((128, 128))
    hist = rgb.histogram()
    values: list[float] = []
    for channel in range(3):
        channel_hist = hist[channel * 256:(channel + 1) * 256]
        grouped = [sum(channel_hist[i * (256 // bins):(i + 1) * (256 // bins)]) for i in range(bins)]
        total = max(sum(grouped), 1)
        values.extend(v / total / 3 for v in grouped)
    return values


def histogram_intersection(a: list[float], b: list[float]) -> float:
    return sum(min(x, y) for x, y in zip(a, b))


def _open_image(source) -> Image.Image:
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        return Image.open(BytesIO(data))
    return Image.open(source)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def ensure_sample_images() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    def font(size: int, bold: bool = False):
        for p in font_candidates[1 if bold else 0:] + font_candidates[:1]:
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = font(24, True)
    small_font = font(16)
    tiny_font = font(13)

    def base(title: str, subtitle: str):
        img = Image.new("RGB", (720, 460), "#F8FAFC")
        d = ImageDraw.Draw(img)
        # card background
        d.rounded_rectangle((28, 28, 692, 432), radius=24, fill="white", outline="#CBD5E1", width=2)
        d.text((52, 46), title, fill="#0F172A", font=title_font)
        d.text((52, 82), subtitle, fill="#475569", font=small_font)
        d.line((52, 112, 668, 112), fill="#E2E8F0", width=2)
        return img, d

    def save_air_compressor(path: Path):
        img, d = base("空压机轴承温升异常", "图像输入：轴承端温度热点、冷却风道检查")
        # compressor body
        d.rounded_rectangle((120, 210, 500, 300), radius=24, fill="#DCEBFF", outline="#2563EB", width=4)
        d.rectangle((190, 168, 425, 220), fill="#BFDBFE", outline="#2563EB", width=3)
        d.ellipse((82, 270, 162, 350), fill="#E2E8F0", outline="#334155", width=3)
        d.ellipse((450, 270, 530, 350), fill="#E2E8F0", outline="#334155", width=3)
        d.rectangle((94, 346, 518, 362), fill="#64748B")
        # hotspot
        d.ellipse((420, 245, 510, 335), fill="#F97316", outline="#DC2626", width=4)
        d.ellipse((445, 270, 485, 310), fill="#FACC15")
        d.line((485, 245, 610, 165), fill="#DC2626", width=4)
        d.rounded_rectangle((560, 130, 660, 178), radius=12, fill="#FEE2E2", outline="#DC2626", width=2)
        d.text((576, 143), "87°C", fill="#991B1B", font=small_font)
        d.text((126, 382), "匹配案例：检查润滑油、冷却器与轴承游隙", fill="#334155", font=small_font)
        img.save(path)

    def save_pump(path: Path):
        img, d = base("离心泵机械密封泄漏", "图像输入：密封端渗漏痕迹、泵体周边积液")
        d.ellipse((150, 170, 370, 390), fill="#DBEAFE", outline="#1D4ED8", width=4)
        d.rectangle((360, 230, 560, 300), fill="#BFDBFE", outline="#1D4ED8", width=4)
        d.ellipse((205, 225, 315, 335), fill="white", outline="#1D4ED8", width=4)
        d.rectangle((92, 252, 150, 282), fill="#93C5FD", outline="#1D4ED8", width=3)
        d.rectangle((558, 250, 635, 280), fill="#93C5FD", outline="#1D4ED8", width=3)
        # leakage drops near seal
        for x, y in [(390, 305), (425, 318), (450, 303)]:
            d.ellipse((x, y, x+18, y+28), fill="#38BDF8", outline="#0284C7")
            d.line((x+9, y+28, x+6, y+54), fill="#0284C7", width=3)
        d.text((132, 392), "匹配案例：检查机械密封、轴套磨损与入口压力", fill="#334155", font=small_font)
        img.save(path)

    def save_cabinet(path: Path):
        img, d = base("配电柜保护告警", "图像输入：面板告警灯、回路负载与连接点温升")
        d.rounded_rectangle((180, 145, 520, 385), radius=18, fill="#F1F5F9", outline="#334155", width=4)
        d.line((350, 145, 350, 385), fill="#CBD5E1", width=3)
        for x in [230, 280, 400, 450]:
            d.rectangle((x, 190, x+46, 260), fill="#E2E8F0", outline="#475569", width=2)
            d.ellipse((x+13, 205, x+33, 225), fill="#FBBF24", outline="#B45309")
        # alarm lamp
        d.ellipse((326, 292, 374, 340), fill="#EF4444", outline="#991B1B", width=3)
        d.text((310, 350), "ALARM", fill="#991B1B", font=tiny_font)
        d.text((126, 406), "匹配案例：核查负载电流、断路器状态和绝缘电阻", fill="#334155", font=small_font)
        img.save(path)

    def save_inverter(path: Path):
        img, d = base("变频器 OC 过流告警", "图像输入：变频器报警界面、电机回路过流信息")
        d.rounded_rectangle((210, 130, 510, 390), radius=24, fill="#EEF2FF", outline="#4F46E5", width=4)
        d.rectangle((250, 170, 470, 245), fill="#111827", outline="#4F46E5", width=3)
        d.text((303, 189), "OC", fill="#F87171", font=ImageFont.truetype(font_candidates[1], 40) if Path(font_candidates[1]).exists() else title_font)
        for y in [275, 315, 355]:
            d.line((250, y, 470, y), fill="#818CF8", width=8)
        d.line((510, 270, 610, 270), fill="#4F46E5", width=5)
        d.ellipse((610, 238, 675, 303), fill="#E0E7FF", outline="#4F46E5", width=3)
        d.text((120, 407), "匹配案例：检查电机绝缘、负载卡滞和加减速参数", fill="#334155", font=small_font)
        img.save(path)

    def save_fan(path: Path):
        img, d = base("风机振动偏大", "图像输入：机壳振动、叶轮积灰或地脚松动迹象")
        d.ellipse((190, 150, 500, 380), fill="#DCFCE7", outline="#16A34A", width=4)
        cx, cy = 345, 265
        for angle in [0, 120, 240]:
            import math
            rad = math.radians(angle)
            x = cx + int(100 * math.cos(rad))
            y = cy + int(75 * math.sin(rad))
            d.polygon([(cx, cy), (x, y), (cx + int(28*math.cos(rad+1.2)), cy + int(28*math.sin(rad+1.2)))], fill="#86EFAC", outline="#15803D")
        d.ellipse((318, 238, 372, 292), fill="white", outline="#15803D", width=4)
        # vibration waves
        for off in [0, 16, 32]:
            d.arc((140-off, 205-off, 550+off, 340+off), 200, 340, fill="#F59E0B", width=4)
        d.rectangle((250, 378, 440, 397), fill="#64748B")
        d.text((144, 407), "匹配案例：检查叶轮积灰、轴承状态和地脚螺栓", fill="#334155", font=small_font)
        img.save(path)

    renderers = {
        "bearing_overheat.png": save_air_compressor,
        "pump_leak.png": save_pump,
        "cabinet_alarm.png": save_cabinet,
        "inverter_overcurrent.png": save_inverter,
        "fan_vibration.png": save_fan,
    }
    for filename, fn in renderers.items():
        target = IMAGE_DIR / filename
        if not target.exists():
            fn(target)


def load_image_cases() -> list[ImageCase]:
    ensure_sample_images()
    raw = [
        ("IMG-001", "空压机", "轴承温升异常", IMAGE_DIR / "bearing_overheat.png", "红外/现场照片显示轴承区域温度偏高。", "停机确认润滑状态、轴承游隙和冷却通道，复测温度曲线。"),
        ("IMG-002", "离心泵", "泵体机械密封泄漏", IMAGE_DIR / "pump_leak.png", "泵壳或密封端有液体渗漏痕迹。", "检查机械密封、轴套磨损和入口压力，必要时更换密封组件。"),
        ("IMG-003", "配电柜", "保护告警/过载", IMAGE_DIR / "cabinet_alarm.png", "柜门或面板出现告警状态。", "核查负载电流、断路器状态和绝缘电阻，按停送电票执行。"),
        ("IMG-004", "变频器", "过流 OC 告警", IMAGE_DIR / "inverter_overcurrent.png", "变频器面板显示 OC/过流告警。", "检查电机绝缘、负载卡滞、加减速时间和输出电缆。"),
        ("IMG-005", "风机", "振动偏大", IMAGE_DIR / "fan_vibration.png", "风机机壳振动明显或地脚螺栓松动。", "检查叶轮积灰、轴承、地脚螺栓和联轴器对中。"),
    ]
    cases: list[ImageCase] = []
    for case_id, device, fault, path, desc, sug in raw:
        with Image.open(path) as img:
            ah = average_hash(img)
            dh = difference_hash(img)
            hist = compact_color_histogram(img)
        cases.append(ImageCase(case_id, device, fault, str(path), desc, sug, ah, dh, hist))
    return cases


def _text_similarity(text_hint: str, cases: list[ImageCase]) -> list[float]:
    hint = expand_query(text_hint or "").strip()
    if not hint:
        return [0.0 for _ in cases]
    documents = [f"{c.device} {c.fault} {c.description} {c.suggestion}" for c in cases]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents + [hint])
    return cosine_similarity(matrix[-1], matrix[:-1]).ravel().tolist()


def _image_confidence(score: float, text_score: float, device_match: float) -> str:
    if score >= 0.82 or (score >= 0.72 and (text_score >= 0.35 or device_match > 0)):
        return "高"
    if score >= 0.58:
        return "中"
    return "低"


def search_similar_image(
    uploaded_file,
    top_k: int = 3,
    *,
    text_hint: str = "",
    device_hint: str = "全部",
) -> list[dict[str, Any]]:
    """Fuse visual similarity with optional field text/device hints.

    Without hints, this behaves as a pure local image retrieval engine. With a
    symptom description or device hint, the ranking becomes a real cross-modal
    match between image features and case metadata.
    """
    with _open_image(uploaded_file) as raw_img:
        img = raw_img.convert("RGB")
        q_ahash = average_hash(img)
        q_dhash = difference_hash(img)
        q_hist = compact_color_histogram(img)

    cases = load_image_cases()
    text_scores = _text_similarity(text_hint, cases)
    has_text = bool((text_hint or "").strip())
    has_device = bool(device_hint and device_hint != "全部")
    rows = []
    for case, text_score in zip(cases, text_scores):
        a_distance = hamming_hex(q_ahash, case.hash)
        d_distance = hamming_hex(q_dhash, case.difference_hash)
        a_score = 1 - a_distance / 64
        d_score = 1 - d_distance / 64
        color_score = histogram_intersection(q_hist, case.color_histogram)
        image_score = 0.45 * a_score + 0.35 * d_score + 0.20 * color_score
        device_match = 1.0 if has_device and device_hint in case.device else 0.0
        if has_text or has_device:
            score = 0.68 * image_score + 0.24 * text_score + 0.08 * device_match
        else:
            score = image_score
        item = asdict(case)
        item["distance"] = round(0.45 * a_distance + 0.35 * d_distance, 2)
        item["score"] = round(score, 3)
        item["image_score"] = round(image_score, 3)
        item["shape_score"] = round(0.45 * a_score + 0.35 * d_score, 3)
        item["color_score"] = round(color_score, 3)
        item["text_score"] = round(float(text_score), 3)
        item["device_match"] = device_match
        item["confidence"] = _image_confidence(score, float(text_score), device_match)
        modes = ["图像特征"]
        if has_text:
            modes.append("故障描述")
        if has_device:
            modes.append("设备类型")
        item["fusion_mode"] = " + ".join(modes)
        rows.append(item)
    return sorted(rows, key=lambda x: (x["score"], x["text_score"], x["image_score"]), reverse=True)[:top_k]
