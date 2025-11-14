#!/bin/bash

##############################################################################
# SensorDrift Data Engineering Pipeline - One-Command Startup
#
# This script starts the complete data engineering pipeline including:
# - PostgreSQL with TimescaleDB
# - Redis
# - MinIO (S3-compatible storage)
# - Apache Kafka
# - Apache Airflow
# - Grafana
# - Metabase
# - Flask Application
# - Jupyter Lab
##############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║        SensorDrift Data Engineering Pipeline            ║"
echo "║     Gas Monitoring & Detection System v2.0              ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Determine Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${YELLOW}🔧 Using Docker Compose command: $DOCKER_COMPOSE${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p airflow/dags airflow/logs airflow/plugins notebooks grafana/provisioning grafana/dashboards data model init_scripts streaming

# Check if data files exist
echo -e "${BLUE}📊 Checking data files...${NC}"
if [ ! -f "data/combined_sensor_data_clean.csv" ]; then
    echo -e "${YELLOW}⚠️  Warning: Real data file not found. Some features may not work.${NC}"
    echo -e "${YELLOW}   Expected: data/combined_sensor_data_clean.csv${NC}"
fi

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping any existing containers...${NC}"
$DOCKER_COMPOSE down --remove-orphans 2>/dev/null || true

# Build images
echo -e "${BLUE}🔨 Building Docker images...${NC}"
$DOCKER_COMPOSE build

# Start infrastructure services first (PostgreSQL, Redis, Kafka, MinIO)
echo -e "${GREEN}🚀 Starting infrastructure services...${NC}"
$DOCKER_COMPOSE up -d postgres redis zookeeper kafka minio

echo -e "${YELLOW}⏳ Waiting for infrastructure services to be healthy (30 seconds)...${NC}"
sleep 30

# Initialize database
echo -e "${BLUE}💾 Database should be initialized automatically via init scripts${NC}"

# Start remaining services
echo -e "${GREEN}🚀 Starting all remaining services...${NC}"
$DOCKER_COMPOSE up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for all services to start (60 seconds)...${NC}"
sleep 60

# Check service health
echo -e "${BLUE}🏥 Checking service health...${NC}"

services=("postgres:5432" "redis:6379" "kafka:9092" "minio:9000" "flask-app:5001" "grafana:3000" "airflow-webserver:8081" "jupyter:8888" "metabase:3001")
service_names=("PostgreSQL" "Redis" "Kafka" "MinIO" "Flask App" "Grafana" "Airflow" "Jupyter Lab" "Metabase")

for i in "${!services[@]}"; do
    service="${services[$i]}"
    name="${service_names[$i]}"
    container="${service%%:*}"

    if docker ps --format '{{.Names}}' | grep -q "sensordrift_${container}"; then
        echo -e "${GREEN}✅ $name is running${NC}"
    else
        echo -e "${YELLOW}⚠️  $name may not be ready yet${NC}"
    fi
done

# Display access information
echo -e "${BLUE}"
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              🎉 Pipeline Started Successfully!           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}📊 Access URLs:${NC}"
echo -e "  ${BLUE}Flask Dashboard:${NC}     http://localhost:5001"
echo -e "  ${BLUE}Apache Airflow:${NC}      http://localhost:8081  (admin/admin)"
echo -e "  ${BLUE}Grafana:${NC}             http://localhost:3000  (admin/admin)"
echo -e "  ${BLUE}Metabase:${NC}            http://localhost:3001"
echo -e "  ${BLUE}Jupyter Lab:${NC}         http://localhost:8888"
echo -e "  ${BLUE}MinIO Console:${NC}       http://localhost:9001  (minioadmin/minioadmin)"

echo ""
echo -e "${GREEN}🔧 Service Ports:${NC}"
echo -e "  ${BLUE}PostgreSQL:${NC}          localhost:5432"
echo -e "  ${BLUE}Redis:${NC}               localhost:6379"
echo -e "  ${BLUE}Kafka:${NC}               localhost:9092"

echo ""
echo -e "${YELLOW}📝 Quick Start Guide:${NC}"
echo -e "  1. Open Flask Dashboard: http://localhost:5001"
echo -e "  2. View Airflow DAGs: http://localhost:8081"
echo -e "  3. Check Data Quality in Grafana: http://localhost:3000"
echo -e "  4. Run EDA in Jupyter Lab: http://localhost:8888"

echo ""
echo -e "${BLUE}🎯 Airflow DAGs Available:${NC}"
echo -e "  • 01_data_ingestion_pipeline     - Ingests data from Kafka"
echo -e "  • 02_data_transformation_pipeline - Cleans and transforms data"
echo -e "  • 03_data_quality_monitoring      - Monitors data quality"

echo ""
echo -e "${YELLOW}⚙️  Useful Commands:${NC}"
echo -e "  ${BLUE}View logs:${NC}           $DOCKER_COMPOSE logs -f [service_name]"
echo -e "  ${BLUE}Stop pipeline:${NC}       $DOCKER_COMPOSE down"
echo -e "  ${BLUE}Restart service:${NC}     $DOCKER_COMPOSE restart [service_name]"
echo -e "  ${BLUE}View services:${NC}       $DOCKER_COMPOSE ps"

echo ""
echo -e "${GREEN}✨ Pipeline is ready! Start exploring your data!${NC}"
echo ""

# Optional: Open browser windows
read -p "Do you want to open the dashboards in your browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🌐 Opening dashboards...${NC}"

    # Detect OS and open browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open http://localhost:5001 2>/dev/null || true
        open http://localhost:8081 2>/dev/null || true
        open http://localhost:3000 2>/dev/null || true
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open http://localhost:5001 2>/dev/null || true
        xdg-open http://localhost:8081 2>/dev/null || true
        xdg-open http://localhost:3000 2>/dev/null || true
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows
        start http://localhost:5001 2>/dev/null || true
        start http://localhost:8081 2>/dev/null || true
        start http://localhost:3000 2>/dev/null || true
    fi
fi

echo -e "${GREEN}🎊 Happy Data Engineering!${NC}"
