#!/bin/bash

# Deployment script for ChurnGuard AI

echo "🚀 Deploying ChurnGuard AI to production..."

# Build Docker images
echo "📦 Building Docker images..."
docker-compose build

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start new containers
echo "🎯 Starting new containers..."
docker-compose up -d

# Run database migrations
echo "📊 Running database migrations..."
docker-compose exec web python -c "
from backend.main import init_database
init_database()
print('✅ Database initialized')
"

# Health check
echo "🏥 Performing health check..."
sleep 5
curl -f http://localhost/api/health || echo "❌ Health check failed"

echo "✅ Deployment complete! Visit http://yourdomain.com"