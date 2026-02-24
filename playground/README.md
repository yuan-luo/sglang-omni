# Playground

This directory contains the SGLang-Omni frontend demo and filesystem API.

## Components

1. Frontend static page (`index.html` + `app.js` + `styles.css`)
2. Filesystem API (`playground/fs_api.py`, default port `8001`) (used to list the filesystem of the server that hosts sglang omni so that you can upload data from it)

## How to Start

### Start frontend + FS API together (recommended)

Run from repository root:

```bash
./scripts/start_playground_and_fs.sh
```

Default ports:
- Frontend: `3000`
- FS API: `8001`

You can pass custom ports:

```bash
./scripts/start_playground_and_fs.sh 3001 9001
```

### Start separately

Frontend:

```bash
python -m http.server 3000 --directory playground
```

FS API:

```bash
python -m playground.fs_api --host 0.0.0.0 --port 8001
```

