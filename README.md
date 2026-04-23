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
  -> glm-ocr:latest
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

OCR_BACKEND=ollama_vision
OCR_MODEL=glm-ocr:latest

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

### 5. Interaktif terminal demo
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo_terminal.ps1
```

Bu script senden:
- bir gorsel secmeni
- istersen mikrofondan soru sormanı veya WAV vermeni
- sonra backend sonucunu terminalde gormeni
ister.

### 5b. Tek komutla tam demo
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_demo.ps1
```

Bu script:
- backend'i ayaga kaldirir
- `/health` cevap verene kadar bekler
- sonra interaktif terminal demo client'ini acar
- demo bitince backend process'ini kapatir

Not:
- Edge TTS disa cikamiyorsa pipeline artik dusmez
- text sonuc yine gelir ve sessiz fallback MP3 doner

### 6. Direct pipeline benchmark
```powershell
.\.venv\Scripts\python tests\test_pipeline_live.py --mode scene
.\.venv\Scripts\python tests\test_pipeline_live.py --mode ocr
.\.venv\Scripts\python tests\test_pipeline_live.py --mode scene --skip-tts
```

### 7. Model benchmark
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

### OCR mode - guncel pratik karar
- Aktif demo/backend varsayilani tekrar `ollama_vision`
- Sebep:
  - mevcut Windows lokal runtime'da PaddleOCR pratik sonuc vermedi
  - `glm-ocr:latest` daha guvenilir demo OCR davranişi verdi

## Pratik OCR Notu

Bugun calisan lokal demo OCR:

```env
OCR_BACKEND=ollama_vision
OCR_MODEL=glm-ocr:latest
```

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
2. Kullanici bir tetikleme tusuna basar
3. Uygulama o andaki en guncel frame'i alir
4. Ayni interaction icinde kisa bir mikrofon kaydi alir
5. Gerekirse mikrofon sesini WAV olarak hazirlar
6. `WS /ws/vision` baglantisi acar
7. Binary payload gonderir
8. JSON sonucu alir
9. MP3 sonucu alir ve oynatir

Final mobil UX icin tercih edilen model:

```text
Model A
  -> kamera preview acik
  -> kullanici tetikleme tusuna basar
  -> uygulama son frame'i yakalar
  -> 2-4 saniyelik ses sorgusu alir
  -> image + audio tek request olarak backend'e gider
```

Bu, surekli video stream gondermekten daha hafif ve daha kararlidir.

## TTS Notu

Su an aktif TTS motoru `Edge TTS` ve lokal demo icin yeterlidir.
Guncel varsayilan local ses profili:

```env
TTS_VOICE=tr-TR-AhmetNeural
TTS_RATE=+10%
```

Ancak daha dogal, daha insansi ve daha duygulu ses icin production yonu su olmalidir:

1. `Edge TTS`
   - local/demo fallback
   - sifir ek servis maliyeti
   - kalite sinirli

2. `Cartesia Sonic-2 / Sonic-3`
   - dusuk latency odakli
   - Turkish destegi bulunan modern streaming TTS secenegi

3. `ElevenLabs`
   - en dogal ve en etkileyici ses kalitesi icin guclu aday
   - daha pahali ve harici servis bagimliligi getirir

Kisa karar:
- demo ve local runtime: `Edge TTS`
- gercek urun hissi icin sonraki adim: harici premium TTS provider

Android tarafi icin sozlesme:
- `GET /contract`

## Android Emulator Test Shell

Repo icinde Android Studio ile acilabilecek kucuk bir test shell var:

```text
android-test-shell/
```

Amaci:
- emulator icinde gorsel secmek
- opsiyonel WAV secmek
- backend'e WebSocket ile gondermek
- JSON ve MP3 sonucunu gormek

Varsayilan emulator backend adresi:

```text
ws://10.0.2.2:8000/ws/vision
```

Kullanim akisi:

1. Android Studio ile `android-test-shell/` klasorunu ac
2. Bir Android Emulator baslat
3. Backend'i bu repodan `scripts/start_backend.ps1` ile calistir
4. Shell icinde gorsel sec
5. Istersen WAV sec
6. `Scene Gonder` veya `Audio+Image Gonder` ile backend'i test et

Not:
- Bu shell su an minimal bir integration client'tir
- Canli mikrofon kaydi yerine WAV secimi kullanir
- Amac Android entegrasyonunu erken dogrulamaktir, production UI yapmak degil

## Bugun Icin Net Durum

- Scene pipeline calisiyor ve test edildi
- STT backend aktif
- PaddleOCR entegrasyonu yapildi ama mevcut Windows lokal ortamda performans ve kalite problemi var
- `qwen2.5:7b` hala en guvenli default model
- `qwen3:4b` benchmark edildi ama mevcut akis icin tercih edilmedi
