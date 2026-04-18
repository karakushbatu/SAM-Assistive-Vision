# SAM Assistive Vision Backend

Bu repo, gorme engelli kullanicilar icin gelistirilen mobil uygulamanin backend tarafidir. Android istemcisi ayri gelistirilecek olsa da backend bugun itibariyla tek basina test edilebilir durumdadir.

## Ne Yapiyor

Backend iki ana modu destekler:

1. `scene`
   - Goruntuyu analiz eder
   - Kisa Turkce aciklama uretir
   - MP3 ses ciktisi doner

2. `ocr`
   - Goruntudeki yaziyi okumaya calisir
   - OCR metni cikarir
   - Gerekirse Turkce ozet uretir
   - MP3 ses ciktisi doner

Ek olarak audio+image envelope ile STT de desteklenir.

## Aktif Pipeline

### Scene mode
```text
image
  -> FastSAM-s
  -> BLIP-large
  -> qwen2.5:7b
  -> Edge TTS
```

### OCR mode
```text
image
  -> FastSAM-s
  -> PaddleOCR
  -> qwen2.5:7b
  -> Edge TTS
```

### Audio + image
```text
audio+image
  -> faster-whisper STT
  -> intent routing
  -> scene or ocr branch
```

## Kullanilan Teknolojiler

- FastAPI
- Uvicorn
- faster-whisper
- FastSAM-s / ultralytics
- BLIP-large / transformers
- Ollama
- Edge TTS
- PaddleOCR

## Gereksinimler

- Windows uzerinde test edildi
- NVIDIA GPU onerilir
- CUDA aktif PyTorch kurulumu onerilir
- Ollama kurulu olmali
- Lokal modeller hazir olmali:
  - `qwen2.5:7b`
- `FastSAM-s.pt` proje icinde mevcut olmali

Opsiyonel OCR fallback:
- `glm-ocr:latest`

## Hizli Kurulum

### 1. Environment hazirla
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local_cuda.ps1
```

### 2. `.env` olustur
```powershell
Copy-Item .env.example .env
```

### 3. Backend baslat
```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_backend.ps1
```

Alternatif:
```powershell
.\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Onerilen `.env` Profili

```env
MODEL_DEVICE=cuda
HF_LOCAL_FILES_ONLY=true

MOCK_STT=false
MOCK_SAM=false
MOCK_BLIP=false
MOCK_OLLAMA=false
MOCK_TTS=false

STT_MODEL=tiny
SAM_MAX_CROPS=2
BLIP_MAX_CAPTIONS=2
BLIP_MAX_NEW_TOKENS=24

SCENE_OLLAMA_MODEL=qwen2.5:7b
OCR_SUMMARY_MODEL=qwen2.5:7b
OLLAMA_NUM_PREDICT=64
OLLAMA_TEMPERATURE=0.1
OLLAMA_KEEP_ALIVE=30m

OCR_BACKEND=paddleocr
PADDLEOCR_LANG=tr
PADDLEOCR_VERSION=PP-OCRv5

TTS_STREAMING=true
```

## API Uclari

- `GET /health`
- `GET /contract`
- `WS /ws/vision`

## WebSocket Girdi Formati

### Legacy image
- Ham `JPEG/PNG` binary

### Audio envelope
```text
[4-byte little-endian audio_length][WAV audio bytes][JPEG/PNG image bytes]
```

## WebSocket Cikti Akisi

1. JSON text event
2. MP3 binary veya MP3 chunk stream
3. Streaming aciksa final `audio_end` text event

## Test Komutlari

### 1. Hemen smoke test
```powershell
.\.venv\Scripts\python tests\test_protocol.py
```

### 2. Health kontrol
```powershell
curl http://127.0.0.1:8000/health
```

Beklenen:
- `ready=true`

### 3. Android istemci simulasyonu
```powershell
.\.venv\Scripts\python tests\ws_client_demo.py --host 127.0.0.1 --port 8000 --image tests/sample_images/yulaf.jpg --frames 1
```

### 4. Audio + image ile STT testi
```powershell
.\.venv\Scripts\python tests\ws_client_demo.py --host 127.0.0.1 --port 8000 --image tests/sample_images/yulaf.jpg --audio path\\to\\query.wav --frames 1
```

### 5. Direct pipeline benchmark
```powershell
.\.venv\Scripts\python tests\test_pipeline_live.py --mode scene
.\.venv\Scripts\python tests\test_pipeline_live.py --mode ocr
.\.venv\Scripts\python tests\test_pipeline_live.py --mode scene --skip-tts
```

### 6. Model benchmark
```powershell
.\.venv\Scripts\python tests\benchmark_model_candidates.py
```

## Guncel Olculen Sonuclar

### Scene mode - 2026-04-16
- `ilaç.jpg`: toplam yaklasik `7250 ms`
- `ss1_test1.jpg`: toplam yaklasik `1907 ms`
- `yulaf.jpg`: toplam yaklasik `2062 ms`
- ortalama: yaklasik `3739 ms`

Bu hatta backend stabil ve demo icin kullanilabilir.

### OCR mode - 2026-04-16
- PaddleOCR backend kod tabanina eklendi ve aktif backend olarak baglandi.
- Ancak mevcut Windows lokal runtime testlerinde:
  - OCR metni ornek goruntulerde bos dondu
  - PaddleOCR GPU yerine CPU fallback yapti
  - toplam sure ciddi bicimde artti

Son test:
- `ilaç.jpg`: OCR branch toplam yaklasik `22625 ms`
- ortalama OCR branch: yaklasik `16005 ms`

Sonuc:
- PaddleOCR entegrasyonu kodda mevcut
- Fakat mevcut lokal Windows stack icin uretim default'u olmaya hazir degil

## Pratik OCR Notu

Bugun calisan bir lokal demo OCR gerekiyorsa gecici fallback:

```env
OCR_BACKEND=ollama_vision
OCR_MODEL=glm-ocr:latest
```

Bu fallback daha yavas olsa da mevcut ornek veri uzerinde daha anlamli sonuc verdi.

## LLM Degerlendirmesi

2026-04-16 itibariyla yerel benchmark sonucu:

- `qwen2.5:7b`
  - hala en tutarli Turkce kalite
  - scene branch icin en guvenli secim

- `qwen3:4b`
  - yerelde benchmark edildi
  - daha hizli gorunse de mevcut Ollama akisinda bos veya dusuk kaliteli cikti verdi
  - bu nedenle default modele gecilmedi

## STT Aktif mi?

Evet. STT backendde aktif.

Calisma kosulu:
- istemci sadece image gonderirse STT devreye girmez
- istemci audio+image envelope gonderirse faster-whisper STT calisir

Yani test ederken STT'yi aktif kullanabilirsin. Bunun icin `--audio` ile bir WAV dosyasi gondermen yeterli.

## Android Entegrasyonu Yuksek Seviye

1. Android kamera goruntusunu alir
2. Gerekirse mikrofon sesini WAV olarak hazirlar
3. `WS /ws/vision` baglantisi acar
4. Binary payload gonderir
5. JSON sonucu alir
6. MP3 sonucu alir ve oynatir

Android tarafi icin sozlesme:
- `GET /contract`

## Bugun Icin Net Durum

- Scene pipeline calisiyor ve test edildi
- STT backend aktif
- PaddleOCR entegrasyonu yapildi ama mevcut Windows lokal ortamda performans ve kalite problemi var
- `qwen2.5:7b` hala en guvenli default model
- `qwen3:4b` benchmark edildi ama mevcut akis icin tercih edilmedi
