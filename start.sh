#!/bin/bash
# Quick start script for OptiChain AI Streamlit App

echo "=================================="
echo "OptiChain AI - Starting Services"
echo "=================================="

# Check if model exists
if [ ! -d "./models/best_model" ]; then
    echo ""
    echo "⚠️  WARNING: Model not found at ./models/best_model"
    echo "Please train and save the model first using the Jupyter notebook."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting services..."
echo ""

# Start all services
docker-compose up -d

echo ""
echo "✅ Services started successfully!"
echo ""
echo "Access points:"
echo "  - Jupyter Lab:  http://localhost:8888"
echo "  - Streamlit:    http://localhost:8501"
echo "  - MongoDB:      mongodb://localhost:27017"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f streamlit"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
