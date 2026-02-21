#!/bin/bash
echo "Starting NLP Inference API..."
uvicorn src.inference:app --host 0.0.0.0 --port 8000
