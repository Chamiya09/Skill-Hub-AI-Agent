#!/usr/bin/env bash

# ==============================================================================
# Script Name: deploy-ai-agent.sh
# Description: Production deployment script for Python FastAPI AI Agent
#              targeting Azure Container Apps (Consumption Tier / Scale-to-Zero).
# ==============================================================================

set -euo pipefail

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
RESOURCE_GROUP="${RESOURCE_GROUP:-TalentMatch-RG}"
LOCATION="${LOCATION:-eastasia}"
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-cae-skillhub-prod}"
APP_NAME="${APP_NAME:-ca-skillhub-ai-agent}"

GITHUB_USERNAME="${GITHUB_USERNAME:-chamiya09}"
IMAGE_TAG="${1:-${IMAGE_TAG:-latest}}"
IMAGE_NAME="ghcr.io/${GITHUB_USERNAME}/skill-hub-ai-agent:${IMAGE_TAG}"

# Resource sizing optimized for LangChain/LangGraph execution (Scale-to-Zero)
CPU="0.5"
MEMORY="1.0Gi"
TARGET_PORT=8000
MIN_REPLICAS=0
MAX_REPLICAS=1

# Default backend URL in Azure Container Apps
DEFAULT_BACKEND_URL="https://ca-skillhub-backend.bravebay-18c4a18e.eastasia.azurecontainerapps.io"
BACKEND_API_URL="${BACKEND_API_URL:-$DEFAULT_BACKEND_URL}"

# Secrets (Prefer passing via environment variables)
# Example: export GROQ_API_KEY="gsk_..."
GROQ_API_KEY="${GROQ_API_KEY:-}"
GROQ_MODEL="${GROQ_MODEL:-llama-3.3-70b-versatile}"
GHCR_TOKEN="${GHCR_PAT:-${GITHUB_TOKEN:-}}"

# ==========================================
# 1. PREREQUISITES CHECK
# ==========================================
echo "---------------------------------------------------------"
echo "🚀 Skill Hub AI Agent: Azure Container App Deployment"
echo "---------------------------------------------------------"

if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI ('az') is not installed."
    echo "   Install from: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

echo "🔍 Checking Azure authentication status..."
if ! az account show &> /dev/null; then
    echo "⚠️ Not logged in. Initiating Azure login..."
    az login --use-device-code
fi

CURRENT_SUB=$(az account show --query "name" -o tsv)
echo "✅ Authenticated to subscription: ${CURRENT_SUB}"

# Ensure containerapp extension is installed and up-to-date
az extension add --name containerapp --upgrade --yes > /dev/null 2>&1 || true

# Register required providers
az provider register --namespace Microsoft.App --wait > /dev/null 2>&1 || true

# ==========================================
# 2. RESOURCE GROUP & ENVIRONMENT VERIFICATION
# ==========================================
echo "🔎 Checking Resource Group: ${RESOURCE_GROUP}..."
if ! az group exists --name "${RESOURCE_GROUP}" | grep -q "true"; then
    echo "📁 Creating resource group ${RESOURCE_GROUP} in ${LOCATION}..."
    az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output table
else
    echo "✅ Resource group ${RESOURCE_GROUP} already exists."
fi

echo "🌐 Verifying Container Apps Managed Environment: ${ENVIRONMENT_NAME}..."
ENV_EXISTS=$(az containerapp env show \
    --name "${ENVIRONMENT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "name" -o tsv 2>/dev/null || true)

if [ -z "${ENV_EXISTS}" ]; then
    echo "⚙️ Creating Container Apps Environment: ${ENVIRONMENT_NAME}..."
    az containerapp env create \
        --name "${ENVIRONMENT_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --output table
    echo "✅ Environment created."
else
    echo "✅ Environment ${ENVIRONMENT_NAME} ready."
fi

# ==========================================
# 3. CONTAINER REGISTRY AUTHENTICATION (GHCR)
# ==========================================
if [ -z "${GHCR_TOKEN}" ]; then
    echo "⚠️ WARNING: GHCR_PAT or GITHUB_TOKEN is not set."
    echo "   If your GHCR package is private, the deployment may fail with 403 Forbidden."
    echo "   Export your GitHub PAT with 'read:packages' scope: export GHCR_PAT='ghp_...'"
fi

# ==========================================
# 4. DEPLOY CONTAINER APP
# ==========================================
echo "🚀 Deploying Container App: ${APP_NAME}..."
echo "   • Image:       ${IMAGE_NAME}"
echo "   • Port:        ${TARGET_PORT}"
echo "   • Resources:   ${CPU} vCPU / ${MEMORY}"
echo "   • Scale:       Min ${MIN_REPLICAS} / Max ${MAX_REPLICAS}"
echo "   • Backend URL: ${BACKEND_API_URL}"

APP_EXISTS=$(az containerapp show \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "name" -o tsv 2>/dev/null || true)

if [ -z "${APP_EXISTS}" ]; then
    echo "📦 Creating new Container App '${APP_NAME}'..."

    REGISTRY_ARGS=()
    if [ -n "${GHCR_TOKEN}" ]; then
        REGISTRY_ARGS=(--registry-server "ghcr.io" --registry-username "${GITHUB_USERNAME}" --registry-password "${GHCR_TOKEN}")
    fi

    # Configure secrets if provided
    SECRET_ARGS=()
    ENV_VARS_ARGS=(
        "PORT=${TARGET_PORT}"
        "BACKEND_API_URL=${BACKEND_API_URL}"
        "GROQ_MODEL=${GROQ_MODEL}"
    )

    if [ -n "${GROQ_API_KEY}" ]; then
        SECRET_ARGS=(--secrets "groq-api-key=${GROQ_API_KEY}")
        ENV_VARS_ARGS+=("GROQ_API_KEY=secretref:groq-api-key")
    else
        echo "⚠️ Note: GROQ_API_KEY was not supplied. Set it via Azure Portal or pass export GROQ_API_KEY='gsk_...'"
    fi

    az containerapp create \
        --name "${APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --environment "${ENVIRONMENT_NAME}" \
        --image "${IMAGE_NAME}" \
        --target-port "${TARGET_PORT}" \
        --ingress external \
        --cpu "${CPU}" \
        --memory "${MEMORY}" \
        --min-replicas "${MIN_REPLICAS}" \
        --max-replicas "${MAX_REPLICAS}" \
        "${REGISTRY_ARGS[@]}" \
        "${SECRET_ARGS[@]}" \
        --env-vars "${ENV_VARS_ARGS[@]}" \
        --output table
else
    echo "🔄 Container App already exists. Updating container image and environment..."

    if [ -n "${GHCR_TOKEN}" ]; then
        echo "🔑 Refreshing GHCR registry credentials..."
        az containerapp registry set \
            --name "${APP_NAME}" \
            --resource-group "${RESOURCE_GROUP}" \
            --server "ghcr.io" \
            --username "${GITHUB_USERNAME}" \
            --password "${GHCR_TOKEN}" \
            --output none
    fi

    if [ -n "${GROQ_API_KEY}" ]; then
        az containerapp secret set \
            --name "${APP_NAME}" \
            --resource-group "${RESOURCE_GROUP}" \
            --secrets "groq-api-key=${GROQ_API_KEY}" \
            --output none
    fi

    UPDATE_ENV_VARS=(
        "PORT=${TARGET_PORT}"
        "BACKEND_API_URL=${BACKEND_API_URL}"
        "GROQ_MODEL=${GROQ_MODEL}"
    )
    if [ -n "${GROQ_API_KEY}" ]; then
        UPDATE_ENV_VARS+=("GROQ_API_KEY=secretref:groq-api-key")
    fi

    az containerapp update \
        --name "${APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${IMAGE_NAME}" \
        --cpu "${CPU}" \
        --memory "${MEMORY}" \
        --min-replicas "${MIN_REPLICAS}" \
        --max-replicas "${MAX_REPLICAS}" \
        --set-env-vars "${UPDATE_ENV_VARS[@]}" \
        --output table
fi

# ==========================================
# 5. POST-DEPLOYMENT VERIFICATION
# ==========================================
echo "---------------------------------------------------------"
echo "🌐 Retrieving Application FQDN..."
APP_FQDN=$(az containerapp show \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null || true)

if [ -n "${APP_FQDN}" ]; then
    echo "✅ AI Agent deployed successfully!"
    echo "🔗 Base URL:   https://${APP_FQDN}"
    echo "🩺 Health:     https://${APP_FQDN}/health"
    echo "📖 OpenAPI:    https://${APP_FQDN}/docs"
else
    echo "⚠️ Warning: Ingress FQDN could not be retrieved."
fi
echo "---------------------------------------------------------"
