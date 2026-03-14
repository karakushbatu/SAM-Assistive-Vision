# Active Context

## Current Phase
**Phase 1 — Mock Pipeline & WebSocket Boilerplate**

## What Was Just Done
- Initialized the entire project from scratch
- Created cline_docs/ Memory Bank
- Scaffolded the full backend directory structure
- Wrote FastAPI WebSocket entry point (main.py)
- Wrote mock AI pipeline (services/ai_pipeline.py)
- Wrote requirements.txt
- Wrote Dockerfile + docker-compose.yml with NVIDIA GPU support

## Immediate Next Steps (for user)
1. Install Docker Desktop (with WSL2 backend) if not already installed
2. Install Python 3.11+ if not already installed
3. Create a Python virtual environment and install requirements
4. Run the FastAPI server locally (without Docker first) to verify WebSocket works
5. Test with a WebSocket client (e.g., websocat or a simple Python test script)

## Active Decisions / Open Questions
- FastSAM vs MobileSAM: not decided yet (deferred to Phase 2)
- Ollama model: planning to use llama3:8b (to be confirmed in Phase 4)
- Android frontend: not started yet (separate project)

## Known Issues
- None at this stage (fresh project)
