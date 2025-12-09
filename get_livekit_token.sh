#!/bin/bash

# Fetch a LiveKit token from the orchestrator
# Usage: ./get_livekit_token.sh [room_name] [participant_name]

ROOM_NAME=${1:-"room-1"}
PARTICIPANT_NAME=${2:-"test-user"}
ORCHESTRATOR_URL="http://localhost:2004"

echo "Fetching token for Room: $ROOM_NAME, Participant: $PARTICIPANT_NAME..."

RESPONSE=$(curl -s "$ORCHESTRATOR_URL/token?room_name=$ROOM_NAME&participant_name=$PARTICIPANT_NAME")

# Check if curl was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to connect to orchestrator at $ORCHESTRATOR_URL"
    exit 1
fi

# Check if jq is installed for parsing JSON
if command -v jq &> /dev/null; then
    TOKEN=$(echo $RESPONSE | jq -r '.token')
    if [ "$TOKEN" == "null" ]; then
        echo "Error: Failed to retrieve token. Response: $RESPONSE"
        exit 1
    fi
else
    # Fallback if jq is not installed (simple grep/sed)
    TOKEN=$(echo $RESPONSE | grep -o '"token":"[^"]*' | sed 's/"token":"//')
fi

echo ""
echo "✅ Token retrieved successfully!"
echo "---------------------------------------------------"
echo "$TOKEN"
echo "---------------------------------------------------"
echo ""
echo "Use this token to connect at: https://livekit.io/connection-tester"
echo "LiveKit URL: ws://localhost:7880"

