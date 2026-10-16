import os
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="ClipMotionfy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://altoonemore-cloud.github.io",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE = Path("/tmp/clipmotionfy")
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(OUTPUTS)), name="files")

MAX_UPLOAD = 200 * 1024 * 1024
MIN_CLIP = 20
MAX_CLIP = 60

# Model AI untuk transkripsi audio.
# Pertama kali dipakai, model akan diunduh oleh server.
model = WhisperModel("tiny", device="cpu", compute_type="int8")


def ffmpeg(args):
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:])


@app.get("/")
def home():
    return {
        "app": "ClipMotionfy",
        "status": "online",
        "features": ["upload", "whisper", "ffmpeg", "mp4"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/clip")
async def create_clips(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(400, "Nama file tidak valid.")

    job = uuid.uuid4().hex
    source = UPLOADS / f"{job}.mp4"
    job_dir = OUTPUTS / job
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        size = 0
        with source.open("wb") as target:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(
                        413,
                        "Batas ukuran video adalah 200 MB."
                    )
                target.write(chunk)

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode != 0:
            raise HTTPException(400, "File bukan video yang valid.")

        duration = float(probe.stdout.strip())
        if duration < MIN_CLIP:
            raise HTTPException(
                400,
                "Video harus berdurasi setidaknya 20 detik."
            )

        segments, info = model.transcribe(
            str(source),
            vad_filter=True,
            beam_size=1,
        )
        segments = list(segments)

        # Kelompokkan ucapan menjadi kandidat berdurasi 20–60 detik.
        groups = []
        current = []
        start = None

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            if start is None:
                start = float(segment.start)

            current.append(segment)
            elapsed = float(segment.end) - start

            if elapsed >= MAX_CLIP:
                groups.append((start, float(segment.end), current))
                current = []
                start = None

        if current and start is not None:
            end = float(current[-1].end)
            if end - start >= MIN_CLIP:
                groups.append((start, end, current))

        # Jika ucapan tidak cukup panjang, gunakan potongan waktu
        # sebagai fallback agar video tetap bisa menghasilkan klip.
        if not groups:
            cursor = 0.0
            while cursor + MIN_CLIP <= duration and len(groups) < 10:
                end = min(cursor + MAX_CLIP, duration)
                groups.append((cursor, end, []))
                cursor += MAX_CLIP

        results = []

        for index, (start, end, group) in enumerate(groups[:10], 1):
            if end - start < MIN_CLIP:
                continue

            filename = f"clip_{index:02d}.mp4"
            output = job_dir / filename

            ffmpeg([
                "-ss", str(start),
                "-i", str(source),
                "-t", str(end - start),
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "26",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(output),
            ])

            transcript = " ".join(
                segment.text.strip() for segment in group
            )
            url = f"/files/{job}/{filename}"

            results.append({
                "id": index,
                "title": f"Clip {index}",
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "transcript": transcript,
                "preview_url": url,
                "download_url": url,
            })

        if not results:
            raise HTTPException(
                422,
                "Tidak ada kandidat klip. Coba video yang lebih panjang."
            )

        return {
            "job_id": job,
            "language": info.language,
            "clips": results,
            "note": (
                "Klip dibuat dari kelompok ucapan atau potongan waktu. "
                "Pemilihan momen viral berbasis AI lanjutan belum tersedia."
            ),
        }

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except subprocess.TimeoutExpired:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(504, "Pemrosesan terlalu lama.")
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Pemrosesan gagal: {exc}")
    finally:
        await video.close()
        source.unlink(missing_ok=True)
