# Fix Docker Network iptables Issue

## Problem
Docker is failing to create networks due to missing iptables chain `DOCKER-ISOLATION-STAGE-2`.

## Solution Options

### Option 1: Restart Docker Desktop (Easiest)
1. Open Docker Desktop
2. Click the gear icon (Settings)
3. Click "Troubleshoot"
4. Click "Restart Docker Desktop"
5. Wait for Docker to restart
6. Try building again

### Option 2: Fix iptables Manually (Requires sudo)
Run these commands in your terminal:

```bash
# Create the missing iptables chain
sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-2

# Add default rule
sudo iptables -t filter -A DOCKER-ISOLATION-STAGE-2 -j RETURN

# Verify
sudo iptables -t filter -L DOCKER-ISOLATION-STAGE-2
```

### Option 3: Use Existing Network (Temporary Workaround)
The TARA services are configured to use the existing `services_leibniz-network` network.

If the network still can't be found, try:

```bash
# Check if network exists
docker context use desktop-linux
docker network ls | grep leibniz

# If it exists, services should work
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d
```

### Option 4: Reset Docker Networks (Nuclear Option)
⚠️ **Warning**: This will remove all custom Docker networks!

```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove all networks (except defaults)
docker network prune -f

# Restart Docker Desktop
# Then rebuild services
```

## Current Status
The `docker-compose-tara.yml` has been updated to use the existing `services_leibniz-network` network as a workaround.

## Next Steps
1. Try restarting Docker Desktop first (Option 1)
2. If that doesn't work, use Option 2 to fix iptables
3. Then rebuild: `docker-compose -p tara-microservice -f docker-compose-tara.yml up -d --build`




