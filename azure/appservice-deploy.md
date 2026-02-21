# Azure App Service Deployment Guide

## Steps

1. Create Azure App Service (Python 3.10)
2. Deploy using Azure CLI:

   az login
   az webapp up --runtime "PYTHON:3.10"

3. Set Startup Command:

   bash azure/startup.sh

4. Configure environment variables in Azure Portal:
   MODEL_PATH=models/spam_model
