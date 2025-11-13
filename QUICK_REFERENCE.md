# 🚀 SensorDrift - Quick Reference Card

## One-Line Commands

### Start Everything
```bash
./start-pipeline.sh
```

### Stop Everything
```bash
docker-compose down
```

### Stop and Delete All Data
```bash
docker-compose down -v
```

### Restart Specific Service
```bash
docker-compose restart [service-name]
```

---

## Service URLs

| Service | URL | Login |
|---------|-----|-------|
| Flask Dashboard | http://localhost:5001 | - |
| Airflow | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Metabase | http://localhost:3001 | Setup required |
| Jupyter Lab | http://localhost:8888 | No password |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

---

## Useful Docker Commands

### View All Services
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f flask-app
docker-compose logs -f airflow-scheduler
docker-compose logs -f kafka
```

### Execute Commands in Container
```bash
# PostgreSQL
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring

# Redis
docker exec -it sensordrift_redis redis-cli

# Flask
docker exec -it sensordrift_flask bash
```

---

## Database Queries

### Check Data Ingestion
```sql
-- Connect to PostgreSQL
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring

-- Count raw readings
SELECT COUNT(*) FROM raw_data.sensor_readings;

-- Count processed readings
SELECT COUNT(*) FROM processed_data.cleaned_sensor_readings;

-- Recent readings by gas type
SELECT gas_type, COUNT(*) as count
FROM processed_data.cleaned_sensor_readings
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY gas_type
ORDER BY count DESC;

-- View hourly statistics
SELECT * FROM analytics.hourly_statistics
ORDER BY hour_timestamp DESC
LIMIT 10;
```

---

## Airflow Commands

### Trigger DAG from CLI
```bash
docker exec -it sensordrift_airflow_webserver \
  airflow dags trigger 01_data_ingestion_pipeline
```

### List All DAGs
```bash
docker exec -it sensordrift_airflow_webserver \
  airflow dags list
```

### View DAG Runs
```bash
docker exec -it sensordrift_airflow_webserver \
  airflow dags list-runs -d 01_data_ingestion_pipeline
```

---

## Kafka Commands

### List Topics
```bash
docker exec -it sensordrift_kafka \
  kafka-topics --list --bootstrap-server localhost:9092
```

### Describe Topic
```bash
docker exec -it sensordrift_kafka \
  kafka-topics --describe --topic sensor-readings --bootstrap-server localhost:9092
```

### Consume Messages (last 10)
```bash
docker exec -it sensordrift_kafka \
  kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic sensor-readings --from-beginning --max-messages 10
```

---

## Streaming Commands

### Run Kafka Producer Manually
```bash
docker exec -it sensordrift_flask \
  python /app/streaming/kafka_producer.py --batch 100
```

### Run Kafka Consumer Manually
```bash
docker exec -it sensordrift_flask \
  python /app/streaming/kafka_consumer.py
```

---

## Monitoring Commands

### Check Service Health
```bash
# PostgreSQL
docker exec sensordrift_postgres pg_isready -U sensordrift

# Redis
docker exec sensordrift_redis redis-cli PING

# Kafka
docker exec sensordrift_kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### View Resource Usage
```bash
docker stats
```

### View Disk Usage
```bash
docker system df
```

---

## Backup & Restore

### Backup Database
```bash
docker exec sensordrift_postgres \
  pg_dump -U sensordrift gas_monitoring > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore Database
```bash
docker exec -i sensordrift_postgres \
  psql -U sensordrift gas_monitoring < backup.sql
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs [service-name]

# Restart service
docker-compose restart [service-name]

# Rebuild and restart
docker-compose up -d --build [service-name]
```

### Database Connection Issues
```bash
# Check PostgreSQL status
docker exec sensordrift_postgres pg_isready

# Check connections
docker exec sensordrift_postgres \
  psql -U sensordrift -d gas_monitoring -c "SELECT count(*) FROM pg_stat_activity;"
```

### Airflow DAGs Not Showing
```bash
# Check scheduler logs
docker logs sensordrift_airflow_scheduler

# List DAG files
docker exec sensordrift_airflow_webserver ls -la /opt/airflow/dags/

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Out of Memory
```bash
# Check memory usage
docker stats --no-stream

# Increase Docker memory in Docker Desktop settings (recommend 8GB)
# Restart Docker Desktop
```

---

## Quick Tests

### Test End-to-End Pipeline
```bash
# 1. Start pipeline
./start-pipeline.sh

# 2. Wait for services (2 minutes)
sleep 120

# 3. Check services are running
docker-compose ps

# 4. Trigger Airflow DAG
# Visit http://localhost:8080, login, enable and trigger DAG

# 5. Check data in database
docker exec -it sensordrift_postgres \
  psql -U sensordrift -d gas_monitoring -c \
  "SELECT COUNT(*) FROM raw_data.sensor_readings;"

# 6. View dashboards
# Visit http://localhost:5001
```

---

## Performance Tips

### Speed Up Queries
```sql
-- Use continuous aggregates
SELECT * FROM analytics.readings_hourly
WHERE hour > NOW() - INTERVAL '24 hours';

-- Use indexes
SELECT * FROM processed_data.cleaned_sensor_readings
WHERE gas_type = 'Ammonia'
AND timestamp > NOW() - INTERVAL '1 hour';
```

### Reduce Memory Usage
```bash
# Stop non-essential services
docker-compose stop jupyter metabase

# Restart with limited services
docker-compose up -d postgres redis kafka flask-app airflow-webserver airflow-scheduler grafana
```

---

## Getting Help

### View Documentation
```bash
cat PROJECT_GUIDE.md
cat README.md
```

### Check Logs for Errors
```bash
# All services
docker-compose logs | grep -i error

# Specific service
docker-compose logs flask-app | grep -i error
```

---

## Common Workflows

### Morning Startup
```bash
./start-pipeline.sh
# Wait 2 minutes
# Open http://localhost:5001
```

### Daily Monitoring
```bash
# Check Airflow DAG runs
# Visit http://localhost:8080

# Check Grafana dashboards
# Visit http://localhost:3000

# Review data quality metrics
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT * FROM monitoring.data_quality_metrics WHERE timestamp > NOW() - INTERVAL '24 hours';"
```

### End of Day Shutdown
```bash
docker-compose down
```

---

**Happy Data Engineering! 🚀**
