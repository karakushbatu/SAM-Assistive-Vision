# Project Brief: SAM Assistive Vision Backend

## Project Goal
A real-time visual assistant backend for visually impaired users.
The Android frontend captures frames and sends them via WebSocket to this FastAPI backend,
which runs a sequential AI pipeline and returns a natural-language scene description.

## Hardware
- Development: NVIDIA RTX 5060 Ti (16GB VRAM), Windows 11
- Deployment target: Cloud GPU server (Linux)
- Dockerization with NVIDIA GPU passthrough is mandatory from Day 1

## Core Tech Stack
- **Runtime**: Python 3.11+
- **API Framework**: FastAPI + Uvicorn
- **Transport**: WebSocket (binary image frames in, JSON text out)
- **Containerization**: Docker + Docker Compose (nvidia runtime)

## AI Pipeline (Sequential)
1. **Segmentation**: FastSAM or MobileSAM — identifies objects/regions in frame
2. **Classification**: CLIP or ResNet — labels each segmented region
3. **Context Generation**: Ollama (local LLM server) + Llama-3-8B — produces natural language description

## Development Strategy: Mock First
- Phase 1 (current): Mock pipeline using `asyncio.sleep()` + fake returns
- Phase 2: Integrate FastSAM / MobileSAM
- Phase 3: Integrate CLIP / ResNet classifier
- Phase 4: Integrate Ollama + Llama-3
- Phase 5: Docker GPU deployment + cloud migration

## Key Constraints
- All code, variables, comments, file names, and docs: English
- All mentor explanations and discussions: Turkish
