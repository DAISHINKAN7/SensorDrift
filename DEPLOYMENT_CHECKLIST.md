# ✅ SensorDrift - Deployment Checklist

## Pre-Deployment

### System Requirements
- [ ] Docker Desktop installed (v20.10+)
- [ ] Docker Compose installed (v2.0+)
- [ ] At least 8GB RAM available
- [ ] At least 10GB free disk space
- [ ] Internet connection for image downloads

### Data Files
- [ ] UCI Gas Sensor Dataset downloaded
- [ ] `data/combined_sensor_data_clean.csv` exists
- [ ] `data/realistic_synthetic_gas_300k.csv` exists (optional)

### Configuration
- [ ] Review `docker-compose.yml`
- [ ] Update passwords in production (`.env` file)
- [ ] Check port availability (5432, 6379, 9092, 5001, 8080, 3000, etc.)

---

## Deployment Steps

### Step 1: Initial Setup
```bash
# Navigate to project directory
cd /path/to/SensorDrift

# Make startup script executable
chmod +x start-pipeline.sh

# Review configuration
cat docker-compose.yml
```
- [ ] Completed

### Step 2: Start Services
```bash
# Start everything
./start-pipeline.sh
```
- [ ] All services started successfully
- [ ] No error messages in output

### Step 3: Verify Services (Wait 2 minutes)
```bash
# Check running containers
docker-compose ps
```

**Expected Services (10 containers):**
- [ ] sensordrift_postgres (healthy)
- [ ] sensordrift_redis (healthy)
- [ ] sensordrift_zookeeper (running)
- [ ] sensordrift_kafka (healthy)
- [ ] sensordrift_minio (healthy)
- [ ] sensordrift_airflow_postgres (healthy)
- [ ] sensordrift_airflow_webserver (healthy)
- [ ] sensordrift_airflow_scheduler (running)
- [ ] sensordrift_grafana (healthy)
- [ ] sensordrift_metabase (running)
- [ ] sensordrift_flask (healthy)
- [ ] sensordrift_jupyter (running)

### Step 4: Verify Database
```bash
# Connect to PostgreSQL
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring

# Check tables exist
\dt raw_data.*;
\dt processed_data.*;
\dt analytics.*;
\dt monitoring.*;

# Exit
\q
```
- [ ] Database initialized
- [ ] All schemas created
- [ ] Reference data populated

### Step 5: Test Kafka
```bash
# List topics
docker exec -it sensordrift_kafka kafka-topics --list --bootstrap-server localhost:9092
```
- [ ] Kafka topics created
- [ ] Can list topics successfully

### Step 6: Verify Airflow
- [ ] Access http://localhost:8080
- [ ] Login successful (admin/admin)
- [ ] DAGs visible in UI
- [ ] 3+ DAGs listed

**Expected DAGs:**
- [ ] 01_data_ingestion_pipeline
- [ ] 02_data_transformation_pipeline
- [ ] 03_data_quality_monitoring

### Step 7: Test Data Ingestion
```bash
# Trigger ingestion DAG manually
# Via Airflow UI: Click "Trigger DAG" on 01_data_ingestion_pipeline
```
- [ ] DAG triggered successfully
- [ ] Tasks completing without errors
- [ ] Data appearing in database

### Step 8: Verify Dashboards

#### Flask Dashboard
- [ ] Access http://localhost:5001
- [ ] Page loads without errors
- [ ] Detection tab works
- [ ] Forecasting tab works
- [ ] Analytics tab works

#### Grafana
- [ ] Access http://localhost:3000
- [ ] Login successful (admin/admin)
- [ ] PostgreSQL datasource configured
- [ ] Can create new dashboard

#### Metabase
- [ ] Access http://localhost:3001
- [ ] Complete initial setup
- [ ] Connect to PostgreSQL
- [ ] Can run sample query

#### Jupyter Lab
- [ ] Access http://localhost:8888
- [ ] Can create new notebook
- [ ] Can connect to database

---

## Post-Deployment Testing

### Test 1: End-to-End Data Flow
```bash
# 1. Produce test data
docker exec -it sensordrift_flask python /app/streaming/kafka_producer.py --batch 10

# 2. Check database
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT COUNT(*) FROM raw_data.sensor_readings;"

# 3. Trigger transformation
# Via Airflow UI: Trigger 02_data_transformation_pipeline

# 4. Check processed data
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT COUNT(*) FROM processed_data.cleaned_sensor_readings;"
```
- [ ] Data flows from Kafka to PostgreSQL
- [ ] Transformation pipeline works
- [ ] Processed data available

### Test 2: Data Quality Monitoring
```bash
# Trigger quality DAG
# Via Airflow UI: Trigger 03_data_quality_monitoring

# Check quality metrics
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "SELECT * FROM monitoring.data_quality_metrics ORDER BY timestamp DESC LIMIT 5;"
```
- [ ] Quality checks run successfully
- [ ] Metrics logged in database

### Test 3: ML Predictions
- [ ] Access Flask dashboard (http://localhost:5001)
- [ ] Click "Test Detection" on Detection tab
- [ ] Verify prediction displays
- [ ] Check all 3 models show results

### Test 4: Visualizations
- [ ] Time-series chart renders in Flask
- [ ] Forecasting chart updates
- [ ] Analytics distribution chart visible
- [ ] Grafana dashboard shows data

---

## Performance Verification

### Database Performance
```bash
# Check query performance
docker exec -it sensordrift_postgres psql -U sensordrift -d gas_monitoring -c \
  "EXPLAIN ANALYZE SELECT * FROM analytics.readings_hourly WHERE hour > NOW() - INTERVAL '24 hours';"
```
- [ ] Query executes in <100ms
- [ ] Indexes being used

### System Resources
```bash
# Check resource usage
docker stats --no-stream
```
- [ ] Memory usage < 6GB
- [ ] CPU usage reasonable
- [ ] No OOM errors

---

## Production Readiness

### Security
- [ ] Change default passwords
- [ ] Enable SSL/TLS for production
- [ ] Review firewall rules
- [ ] Enable authentication on all services
- [ ] Review user permissions

### Monitoring
- [ ] Set up email alerts in Airflow
- [ ] Configure Grafana alerts
- [ ] Set up log aggregation
- [ ] Configure backup schedule

### Backup & Recovery
- [ ] Test database backup
```bash
docker exec sensordrift_postgres pg_dump -U sensordrift gas_monitoring > backup_test.sql
```
- [ ] Test database restore
```bash
docker exec -i sensordrift_postgres psql -U sensordrift gas_monitoring < backup_test.sql
```
- [ ] Document recovery procedures

### Documentation
- [ ] Team trained on system
- [ ] Runbooks created
- [ ] Contact information documented
- [ ] Escalation procedures defined

---

## Go-Live Checklist

### Final Verification
- [ ] All services running and healthy
- [ ] All tests passing
- [ ] Dashboards accessible
- [ ] Airflow DAGs scheduled
- [ ] Monitoring configured
- [ ] Backups working
- [ ] Team ready

### Communication
- [ ] Stakeholders notified
- [ ] Documentation shared
- [ ] Support team briefed
- [ ] Access credentials distributed

### Launch
- [ ] Enable Airflow DAG schedules
- [ ] Start monitoring dashboards
- [ ] Begin data collection
- [ ] Monitor for first 24 hours

---

## Rollback Plan

If issues occur:

```bash
# 1. Stop all services
docker-compose down

# 2. Check logs for errors
docker-compose logs > error_logs.txt

# 3. Review error logs

# 4. Fix issues and restart
./start-pipeline.sh
```

---

## Maintenance Schedule

### Daily
- [ ] Check Airflow DAG runs
- [ ] Review error logs
- [ ] Monitor data quality metrics
- [ ] Verify data freshness

### Weekly
- [ ] Review performance metrics
- [ ] Check disk usage
- [ ] Update documentation
- [ ] Team sync meeting

### Monthly
- [ ] Test backup/restore
- [ ] Review security settings
- [ ] Update dependencies
- [ ] Performance tuning

---

## Success Criteria

The deployment is successful when:

- ✅ All 12 services running
- ✅ Data flowing through pipeline
- ✅ Airflow DAGs executing on schedule
- ✅ Dashboards displaying data
- ✅ ML predictions working
- ✅ Data quality checks passing
- ✅ No critical errors in logs
- ✅ Team can access all services
- ✅ Backup/restore tested
- ✅ Monitoring alerts configured

---

## Support Contacts

- **Technical Lead**: [Your Name]
- **Database Admin**: [DBA Name]
- **DevOps**: [DevOps Contact]
- **Data Science**: [DS Contact]

---

## Additional Resources

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) - Complete documentation
- [README.md](README.md) - Quick start guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference

---

**Deployment Date**: _______________

**Deployed By**: _______________

**Sign-off**: _______________

---

✅ **Deployment Complete!**
