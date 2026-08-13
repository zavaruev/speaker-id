# Speaker ID

Docker Compose project hosting a FastAPI speaker-identification service (WeSpeaker CAMPPlus, 512-dim embeddings). The only live code lives in `speaker_id_service/`; everything else at the root is docs or legacy.

## Layout

- `speaker_id_service/` — the service: `docker-compose.yaml`, `enroll_client.sh`, plus the app in `speaker_id/` (FastAPI `app.py`, CAMPPlus model code, Dockerfile). **Read `speaker_id_service/AGENTS.md` — it has the full architecture, API table, and audio pipeline; this file only covers what it omits.**
- `DESIGN.md` — Apple design-system spec for the `/enroll` UI. That UI is an inline HTML string inside `speaker_id/app.py` (lines ~207–917); UI changes happen there, not in a template dir.
- `setup_speaker_id.sh` — legacy installer (README: "kept for reference"). Do not use; it generates an outdated speechbrain-based app.

## Run

```bash
cd speaker_id_service
docker compose up --build
```

- Requires NVIDIA GPU host (CUDA 11.8, Pascal+) + `nvidia-container-toolkit`; compose reserves `count: all` GPUs. Container may be rebuilt by the tooling.
- Server on `:8001`; readiness = log line `CAMPPlus model successfully loaded!`. First start downloads the 63 MB checkpoint from HuggingFace into the bind-mounted `./models/speaker_id/` (models/ and speakers/ are gitignored — recreate dirs if missing).
- `docker compose exec speaker_id bash` to enter the container.

## Tests

No pytest config, no pytest/torch in `requirements.txt` — pytest is not installed in the image.

Test code is duplicated across **three** locations (all variants of `test_campplus_model.py` / `test_pooling_layers.py` exist in several):

- `speaker_id_service/speaker_id/tests/` — main suite (`test_app.py`, `test_path_traversal.py`, `test_security.py`, `test_campplus_model.py`, `test_pooling_layers.py`)
- `speaker_id_service/tests/test_campplus_model.py` — imports via `../speaker_id`
- `tests/test_campplus_model.py` (repo root) — imports `speaker_id_service.speaker_id.campplus_model`

Run the suite inside the container (only `speaker_id/` contents are copied into the image, so the two outer test dirs aren't available there):

```bash
docker compose exec speaker_id pip install pytest pytest-asyncio httpx2
docker compose exec speaker_id python -m pytest tests -q
```

(`httpx2` is required by starlette's `TestClient`; `pytest-asyncio` for the async `convert_to_wav` tests — neither is in `requirements.txt`.)

App tests (`test_app.py`, `test_security.py`, `test_path_traversal.py`) mock `torch`/`torchaudio` into `sys.modules` *before* importing `app` — don't remove that mocking or they'll try to load the real model. Since `convert_to_wav` is now async, those tests patch it with `unittest.mock.AsyncMock` — preserve that. **Important:** `test_app.py` must restore the real modules in `sys.modules` right after `import app` (and `test_security.py` must scope its `patch()`s to the import) — otherwise later pooling/model tests import the mocked `torch` and fail with `StopIteration`. Model/pooling tests need real torch, so they only run where torch is installed (container or a CUDA-capable host env).

## Gotchas

- `/enroll` requires an API key: header `X-API-Key`, value = env `API_KEY` (fallback `default_secret_key` — weak, and `docker-compose.yaml` does not set it). The inline UI sends it from a form field; `enroll_client.sh` does **not** and will fail with 401.
- `user_id` is restricted to `^[a-zA-Z0-9_-]+$` (after `os.path.basename`); filenames are sanitized with `os.path.basename`, `.`/`..` and exceeding `MAX_FILES = 50` files → 400. There are dedicated path-traversal/security tests — preserve this behavior.
- 50 MB upload cap (HTTP 413), min audio 4000 samples (~0.25 s), confidence < 0.4 → `"unknown"`.
- Enrolled voices are `.npy` files in the mounted `./speakers/` volume; enrollment is atomic (tmp + rename) and rebuilds an in-memory cosine-similarity cache.
- GPU inference has a CPU fallback on `RuntimeError` — keep it.
- Upload temp files use `tempfile.NamedTemporaryFile`, file writes via `anyio.open_file`, `convert_to_wav` runs `asyncio.create_subprocess_exec`, cleanup via `run_in_threadpool(_safe_remove, ...)` — keep these async patterns (no blocking `open`/`os.remove`/`subprocess.run` in request paths).
- TLS is opt-in via env `SSL_KEYFILE`/`SSL_CERTFILE` in the `uvicorn.run` block (no certs → plain HTTP).
