# ⚠️ Docker Network Fix Required

## Problem
Docker cannot create networks because the iptables chain `DOCKER-ISOLATION-STAGE-2` is missing. This is a Docker daemon corruption issue.

## ✅ Quick Fix (Recommended)

**Restart Docker Desktop:**
1. Open Docker Desktop application
2. Click the gear icon (⚙️ Settings)
3. Go to "Troubleshoot" tab
4. Click "Restart Docker Desktop"
5. Wait for Docker to fully restart (~30 seconds)
6. Try again: `docker-compose -p tara-microservice -f docker-compose-tara.yml up -d`

## 🔧 Alternative Fix (If restart doesn't work)

Run these commands in your terminal (requires sudo password):

```bash
# Fix the missing iptables chain
sudo iptables -t filter -N DOCKER-ISOLATION-STAGE-2
sudo iptables -t filter -A DOCKER-ISOLATION-STAGE-2 -j RETURN

# Verify it was created
sudo iptables -t filter -L DOCKER-ISOLATION-STAGE-2

# Then restart Docker Desktop
```

## 📋 What Was Changed

The `docker-compose-tara.yml` file has been updated to:
- Remove the `version` field (obsolete)
- Use the existing `services_leibniz-network` network (to avoid creating new networks)

## 🚀 After Fixing

Once Docker is restarted, you can build and start services:

```bash
cd /home/prometheus/leibniz_agent/services
docker context use desktop-linux
docker-compose -p tara-microservice -f docker-compose-tara.yml up -d --build
```

## 📝 Note

This is a Docker daemon issue, not a problem with our configuration. The iptables chains get corrupted sometimes, especially after system updates or Docker Desktop updates. Restarting Docker Desktop usually fixes it.

