# 🚀 How to Run SensorDrift - Complete Guide

## 🎯 Quick Start (3 Steps!)

### Step 1: Prerequisites Check

Ensure you have:
- ✅ Docker Desktop installed and running
- ✅ At least 8GB RAM available
- ✅ At least 10GB free disk space
- ✅ Internet connection

**Check Docker:**
```bash
docker --version
docker-compose --version
```

### Step 2: Navigate to Project
```bash
cd /home/user/SensorDrift
```

### Step 3: Start Everything!
```bash
./start-pipeline.sh
```

**That's it!** The pipeline will start automatically. Wait 2-3 minutes for all services to initialize.

---

## 🌐 Access Your Services

Once started, open these URLs:

### Primary Interfaces
1. **Flask Dashboard** - http://localhost:5001
   - Main application interface
   - Gas detection, forecasting, analytics

2. **Apache Airflow** - http://localhost:8080
   - Login: `admin` / `admin`
   - View and trigger data pipelines (DAGs)

3. **Grafana** - http://localhost:3000
   - Login: `admin` / `admin`
   - Real-time monitoring dashboards

### Additional Tools
4. **Metabase** - http://localhost:3001
   - Business intelligence and ad-hoc queries
   - Complete initial setup on first visit

5. **Jupyter Lab** - http://localhost:8888
   - Data exploration and analysis
   - No password required

6. **MinIO Console** - http://localhost:9001
   - Login: `minioadmin` / `minioadmin`
   - S3-compatible object storage

---

## 📊 Running the Data Pipeline

### Automatic Pipeline Execution

The pipeline runs automatically on schedule:
- **Data Ingestion**: Every 30 minutes
- **Data Transformation**: Every 15 minutes
- **Data Quality Checks**: Every hour

### Manual Pipeline Execution

#### Option 1: Via Airflow UI (Recommended)
1. Open http://localhost:8080
2. Login with `admin` / `admin`
3. Enable the DAGs (toggle switch on the left)
4. Click "Trigger DAG" button (play icon) to run manually
5. Watch execution in real-time

#### Option 2: Via Command Line
```bash
# Trigger data ingestion
docker exec sensordrift_airflow_webserver \
  airflow dags trigger 01_data_ingestion_pipeline

# Trigger transformation
docker exec sensordrift_airflow_webserver \
  airflow dags trigger 02_data_transformation_pipeline

# Trigger quality checks
docker exec sensordrift_airflow_webserver \
  airflow dags trigger 03_data_quality_monitoring
```

---

## 🎓 Complete Workflow Demo

### For Presentation/Grading

**1. Start the System (2 minutes)**
```bash
./start-pipeline.sh
# Wait for "Pipeline Started Successfully!" message
```

**2. Show Airflow Orchestration (5 minutes)**
- Open http://localhost:8080
- Login with admin/admin
- Show 3 DAGs:
  - 01_data_ingestion_pipeline
  - 02_data_transformation_pipeline
  - 03_data_quality_monitoring
- Enable and trigger "01_data_ingestion_pipeline"
- Show Graph View of task execution
- Explain each task:
  - produce_sensor_data: Streams from UCI dataset to Kafka
  - consume_and_store_data: Stores in PostgreSQL
  - verify_ingestion: Validates data count
  - log_pipeline_execution: Logs to monitoring table

**3. Show Data Flow (5 minutes)**

**Check Kafka streaming:**
```bash
# List Kafka topics
docker exec sensordrift_kafka \
  kafka-topics --list --bootstrap-server localhost:9092

# View messages (sample)
docker exec sensordrift_kafka \
  kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic sensor-readings --from-beginning --max-messages 5
```

**Check PostgreSQL storage:**
```bash
# Connect to database
docker exec -it sensordrift_postgres \
  psql -U sensordrift -d gas_monitoring

# Run queries
SELECT COUNT(*) FROM raw_data.sensor_readings;
SELECT COUNT(*) FROM processed_data.cleaned_sensor_readings;
SELECT * FROM analytics.gas_distribution_stats;

# Exit
\q
```

**4. Show Data Transformation (3 minutes)**
- Trigger "02_data_transformation_pipeline" in Airflow
- Show tasks:
  - clean_and_validate_data: Removes outliers, handles nulls
  - detect_duplicates: Flags duplicate readings
  - calculate_aggregates: Creates hourly statistics
  - log_transformation: Records metrics

**5. Show Data Quality (3 minutes)**
- Trigger "03_data_quality_monitoring" in Airflow
- Show quality checks:
  - check_completeness: Validates data completeness
  - check_freshness: Ensures data is recent
  - check_consistency: Validates data ranges
  - generate_quality_report: Summary report

**6. Show Analytics & Visualization (5 minutes)**

**Flask Dashboard:**
- Open http://localhost:5001
- **Detection Tab**: Run real-time gas detection
- **Forecasting Tab**: Show time-series predictions
- **Analytics Tab**: Display distribution charts
- **Export Features**: Generate PDF report

**Grafana:**
- Open http://localhost:3000
- Login with admin/admin
- Create simple dashboard:
  - Add panel
  - Select PostgreSQL datasource
  - Query: `SELECT * FROM analytics.gas_distribution_stats`
  - Save dashboard

**7. Show Machine Learning (3 minutes)**
- In Flask dashboard, click "Compare Models"
- Show predictions from 3 models:
  - Real Data Only Model (99.57%)
  - Synthetic Data Only Model (99.68%)
  - Optimal Mix Model (99.73%) ⭐
- Explain consensus prediction

**8. Show Data Lake (2 minutes)**
- Open http://localhost:9001 (MinIO)
- Login with minioadmin/minioadmin
- Show bucket structure (if configured)
- Explain raw data storage strategy

**9. Show EDA (2 minutes)**
- Open http://localhost:8888 (Jupyter Lab)
- Create new notebook
- Connect to database:
```python
import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host='postgres',
    database='gas_monitoring',
    user='sensordrift',
    password='sensordrift_password'
)

df = pd.read_sql("SELECT * FROM analytics.gas_distribution_stats", conn)
df
```
- Show quick visualization

---

## 📊 Key Metrics to Show

### Data Ingestion Metrics
```bash
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT
     COUNT(*) as total_records,
     COUNT(DISTINCT gas_type) as unique_gases,
     MIN(timestamp) as first_reading,
     MAX(timestamp) as latest_reading
   FROM raw_data.sensor_readings;"
```

### Data Quality Metrics
```bash
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT
     metric_name,
     metric_value,
     is_within_threshold,
     timestamp
   FROM monitoring.data_quality_metrics
   ORDER BY timestamp DESC
   LIMIT 10;"
```

### Pipeline Execution Metrics
```bash
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT
     pipeline_name,
     status,
     records_processed,
     duration_seconds,
     execution_timestamp
   FROM monitoring.pipeline_executions
   ORDER BY execution_timestamp DESC
   LIMIT 10;"
```

---

## 🧪 Testing Checklist

Before presentation, verify:

### ✅ All Services Running
```bash
docker-compose ps
# Should show 12 containers, all "Up" or "healthy"
```

### ✅ Database Initialized
```bash
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c "\dt raw_data.*; \dt processed_data.*; \dt analytics.*;"
# Should show multiple tables
```

### ✅ Airflow DAGs Visible
- Open http://localhost:8080
- Should see 3 DAGs listed

### ✅ Flask Dashboard Working
- Open http://localhost:5001
- All tabs should load without errors

### ✅ Grafana Accessible
- Open http://localhost:3000
- Should login successfully

---

## 🛑 How to Stop

### Stop All Services
```bash
docker-compose down
```

### Stop and Remove All Data
```bash
docker-compose down -v
# WARNING: This deletes all data!
```

### Stop Specific Service
```bash
docker-compose stop [service-name]
# Example: docker-compose stop jupyter
```

---

## 🔄 How to Restart

### Restart Everything
```bash
docker-compose down
./start-pipeline.sh
```

### Restart Specific Service
```bash
docker-compose restart [service-name]
# Example: docker-compose restart flask-app
```

---

## 📝 Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f flask-app
docker-compose logs -f airflow-scheduler
docker-compose logs -f kafka
```

### Check Resource Usage
```bash
docker stats
```

### Execute Commands in Containers
```bash
# PostgreSQL
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring

# Redis
docker exec -it sensordrift_redis redis-cli

# Flask Python shell
docker exec -it sensordrift_flask python
```

---

## 🐛 Troubleshooting

### Issue: Services Won't Start
```bash
# Check Docker is running
docker ps

# Check system resources
docker system df

# Restart Docker Desktop
# Increase RAM to 8GB in Docker Desktop settings
```

### Issue: Port Already in Use
```bash
# Find process using port (example: 5432)
lsof -i :5432

# Kill process or change port in docker-compose.yml
```

### Issue: Database Connection Failed
```bash
# Check PostgreSQL logs
docker logs sensordrift_postgres

# Restart PostgreSQL
docker-compose restart postgres

# Wait 30 seconds and try again
sleep 30
```

### Issue: Airflow DAGs Not Showing
```bash
# Check scheduler logs
docker logs sensordrift_airflow_scheduler

# Check DAG files exist
ls -la airflow/dags/

# Restart Airflow
docker-compose restart airflow-webserver airflow-scheduler
```

---

## 📊 Grading Rubric Demonstration

### Component 1: Data Ingestion (3 marks)
**Show:**
- Kafka streaming (kafka-console-consumer)
- PostgreSQL ingestion (SELECT COUNT(*) queries)
- CSV file source (ls data/)
- Airflow ingestion DAG execution

### Component 2: Data Transformation (3 marks)
**Show:**
- Cleaning pipeline (duplicate detection, outlier removal)
- Feature engineering (concentration_ppm, risk_score)
- Multi-source joins (real + synthetic)
- Transformation DAG execution

### Component 3: Data Storage (3 marks)
**Show:**
- PostgreSQL with TimescaleDB (\dt commands)
- MinIO console (http://localhost:9001)
- Redis (docker exec redis-cli)
- Schema design (Bronze/Silver/Gold layers)

### Component 4: Orchestration (3 marks)
**Show:**
- 3 Airflow DAGs
- Task dependencies (Graph View)
- Retry logic (DAG configuration)
- Monitoring logs (monitoring.pipeline_executions)

### Component 5: Visualization (3 marks)
**Show:**
- Flask dashboard (http://localhost:5001)
- Grafana dashboards (http://localhost:3000)
- Metabase queries (http://localhost:3001)
- Jupyter notebooks (http://localhost:8888)

### Component 6: Presentation (3 marks)
**Show:**
- One-command setup (./start-pipeline.sh)
- Complete documentation (PROJECT_GUIDE.md)
- Clear architecture diagram
- Live demo of end-to-end flow

### Component 7: Problem Definition (2 marks)
**Explain:**
- UCI Gas Sensor Dataset
- Real-world application (industrial safety)
- 6 hazardous gases monitored
- ML accuracy (99.73%)

---

## 🎯 Tips for Best Demo

1. **Start Early**: Run `./start-pipeline.sh` before presentation
2. **Test Everything**: Go through checklist above
3. **Have Backup**: Take screenshots of working dashboards
4. **Know Your Data**: Understand the UCI dataset
5. **Explain Clearly**: Describe each component's purpose
6. **Show Metrics**: Display actual data, not just interfaces
7. **Be Confident**: You built a production-ready system!

---

## 📞 Need Help?

If something doesn't work:

1. Check logs: `docker-compose logs [service]`
2. Restart service: `docker-compose restart [service]`
3. Check documentation: `cat PROJECT_GUIDE.md`
4. Review quick reference: `cat QUICK_REFERENCE.md`

---

## 🎉 Success Indicators

Your system is working perfectly when:

✅ All 12 containers are running
✅ Airflow shows 3 DAGs
✅ Flask dashboard displays data
✅ Grafana connects to PostgreSQL
✅ Database contains sensor readings
✅ Pipeline executions logged
✅ No errors in logs
✅ All URLs accessible

---

**You're ready to demonstrate a world-class Data Engineering project! 🚀**

**Good luck with your presentation!** 🎓
