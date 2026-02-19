#!/bin/bash
# Run the prediction service

cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the service
echo "Starting TrueFlux Prediction Service on port 8001..."
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
