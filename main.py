from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, ffmpeg
from transformers import pipeline
from opencc import OpenCC

# 初始化 FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    lang: str = Form("traditional")  # 新增語言參數，預設繁體
):
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
    if lang == "simplified":
        cc = OpenCC("t2s")  # 繁 -> 簡
        transcript = cc.convert(transcript)
    else:
        cc = OpenCC("s2t")  # 簡 -> 繁
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
