# 🚀 Render Deployment Guide for vital-DSP

## Quick Setup

### 1. **Repository Configuration**
- **Root Directory**: `src`
- **Dockerfile Path**: `src/Dockerfile`
- **Docker Command**: `python vitalDSP_webapp/run_webapp.py`

### 2. **Environment Variables**
Add these in Render dashboard:
```
PYTHONPATH=/app
PORT=8000
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### 3. **Health Check**
- **Health Check Path**: `/api/health`
- This endpoint returns `{"status": "healthy"}` for monitoring

### 4. **Instance Type**
- **Free Tier**: 512 MB RAM, 0.1 CPU (good for testing)
- **Starter**: $7/month, 512 MB RAM, 0.5 CPU (recommended for production)

## 🔧 Configuration Details

### Dockerfile Features
- ✅ Python 3.9 slim base image
- ✅ System dependencies (gcc, g++, curl)
- ✅ Optimized layer caching
- ✅ Health check endpoint
- ✅ Proper port exposure
- ✅ Uploads directory creation

### Application Structure
```
src/
├── vitalDSP_webapp/
│   ├── run_webapp.py          # Main entry point
│   ├── app.py                 # FastAPI + Dash app
│   ├── requirements.txt       # Webapp dependencies
│   └── api/endpoints.py       # Health check endpoint
├── vitalDSP/                  # Core library
└── Dockerfile                 # Render-optimized Dockerfile
```

### Health Check Endpoint
- **URL**: `https://your-app.onrender.com/api/health`
- **Response**: `{"status": "healthy", "timestamp": "..."}`
- **Purpose**: Render monitors this for service health

## 🚀 Deployment Steps

1. **Connect Repository**
   - Connect your GitHub repo to Render
   - Select `vital-DSP` repository

2. **Configure Service**
   - **Name**: `vital-DSP`
   - **Root Directory**: `src`
   - **Dockerfile Path**: `src/Dockerfile`
   - **Docker Command**: `python vitalDSP_webapp/run_webapp.py`

3. **Set Environment Variables**
   - Add the environment variables listed above

4. **Configure Health Check**
   - **Health Check Path**: `/api/health`

5. **Deploy**
   - Click "Deploy web service"
   - Wait for build to complete (~5-10 minutes)

## 🔍 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check Dockerfile path is `src/Dockerfile`
   - Verify all dependencies are in requirements.txt

2. **App Won't Start**
   - Check Docker command is correct
   - Verify PORT environment variable

3. **Health Check Fails**
   - Ensure `/api/health` endpoint is accessible
   - Check logs for startup errors

### Logs
- View logs in Render dashboard
- Check both build logs and runtime logs
- Look for Python import errors or port binding issues

## 📊 Monitoring

### Health Check
- Render pings `/api/health` every 30 seconds
- Service restarts if health check fails
- Monitor in Render dashboard

### Performance
- Free tier: Spins down after 15 minutes of inactivity
- Paid tiers: Always running
- Monitor CPU and memory usage

## 🔄 Updates

### Auto-Deploy
- Enabled by default
- Deploys on every push to main branch
- Can be disabled for manual deployments

### Manual Deploy
- Go to Render dashboard
- Click "Manual Deploy"
- Select branch to deploy

## 💡 Tips

1. **Use Free Tier for Testing**
   - Perfect for development and testing
   - No cost, but spins down when idle

2. **Upgrade for Production**
   - Starter tier ($7/month) for production use
   - Better performance and reliability

3. **Monitor Resource Usage**
   - Check logs for memory/CPU issues
   - Upgrade if needed

4. **Environment Variables**
   - Store secrets in Render environment variables
   - Don't commit sensitive data to repo

## 🎯 Expected Result

After successful deployment:
- ✅ App accessible at `https://your-app.onrender.com`
- ✅ Health check working at `/api/health`
- ✅ All features functional
- ✅ File uploads working
- ✅ Real-time processing available

## 📞 Support

If you encounter issues:
1. Check Render logs
2. Verify configuration settings
3. Test locally first
4. Check this guide for common solutions