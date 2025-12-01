#!/bin/bash
# Script to launch Native WebRTC (UDP) mode

echo "🚀 Stopping existing services..."
docker-compose down 2>/dev/null
docker stop stt-vad-service 2>/dev/null
docker rm stt-vad-service 2>/dev/null

echo "🔨 Building Native WebRTC container..."
docker-compose -f docker-compose.native.yml build

echo "🟢 Starting Native WebRTC service..."
docker-compose -f docker-compose.native.yml up -d

echo "⏳ Waiting for startup..."
sleep 5

echo "📊 Service Logs:"
docker-compose -f docker-compose.native.yml logs -f




