from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, ffmpeg
from transformers import pipeline
from opencc import OpenCC

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延遲初始化 ASR
asr = None

@app.on_event("startup")
async def startup_event():
    print("🚀 Server started, waiting for first ASR request...")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    lang: str = Form("traditional")
):
    global asr
    if asr is None:
        print("⏳ Loading Whisper model...")
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)

    # 暫存檔案
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # 抽音訊
    audio_path = temp_path.replace(".mp4", ".mp3")
    ffmpeg.input(temp_path).output(audio_path).run()

    # Whisper 語音轉文字
    result = asr(audio_path)
    transcript = result["text"]

    # 繁體 / 簡體轉換
    cc = OpenCC("t2s") if lang == "simplified" else OpenCC("s2t")
    transcript = cc.convert(transcript)

    # 模擬 GPT 摘要
    summary = "、".join(transcript.split("。")[:5]) + "。"

    # 清理暫存檔
    os.remove(temp_path)
    os.remove(audio_path)

    return {
        "transcript": transcript,
        "summary": summary,
        "lang": lang
    }
