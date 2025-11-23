# OptiChain AI - Quick Start Script (PowerShell)
# Run this to start all services

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "OptiChain AI - Starting Services" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
if (-Not (Test-Path ".\models\best_model")) {
    Write-Host "⚠️  WARNING: Model not found at .\models\best_model" -ForegroundColor Yellow
    Write-Host "Please train and save the model first using the Jupyter notebook." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Starting Docker services..." -ForegroundColor Green
Write-Host ""

# Start all services
docker-compose up -d

Write-Host ""
Write-Host "✅ Services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "  - Jupyter Lab:  http://localhost:8888" -ForegroundColor White
Write-Host "  - Streamlit:    http://localhost:8501" -ForegroundColor White
Write-Host "  - MongoDB:      mongodb://localhost:27017" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View Streamlit logs:  docker-compose logs -f streamlit" -ForegroundColor White
Write-Host "  View all logs:        docker-compose logs -f" -ForegroundColor White
Write-Host "  Stop services:        docker-compose down" -ForegroundColor White
Write-Host "  Restart Streamlit:    docker-compose restart streamlit" -ForegroundColor White
Write-Host ""
Write-Host "Opening Streamlit in browser in 5 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Start-Process "http://localhost:8501"
