-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- ============================================================================
-- RAW DATA TABLES (Bronze Layer - Data Lake)
-- ============================================================================

-- Raw sensor readings table
CREATE TABLE IF NOT EXISTS raw_data.sensor_readings (
    reading_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    batch_id VARCHAR(50),
    gas_type VARCHAR(50),
    gas_label INTEGER,
    -- 128 sensor features
    f1 DOUBLE PRECISION, f2 DOUBLE PRECISION, f3 DOUBLE PRECISION, f4 DOUBLE PRECISION,
    f5 DOUBLE PRECISION, f6 DOUBLE PRECISION, f7 DOUBLE PRECISION, f8 DOUBLE PRECISION,
    f9 DOUBLE PRECISION, f10 DOUBLE PRECISION, f11 DOUBLE PRECISION, f12 DOUBLE PRECISION,
    f13 DOUBLE PRECISION, f14 DOUBLE PRECISION, f15 DOUBLE PRECISION, f16 DOUBLE PRECISION,
    f17 DOUBLE PRECISION, f18 DOUBLE PRECISION, f19 DOUBLE PRECISION, f20 DOUBLE PRECISION,
    f21 DOUBLE PRECISION, f22 DOUBLE PRECISION, f23 DOUBLE PRECISION, f24 DOUBLE PRECISION,
    f25 DOUBLE PRECISION, f26 DOUBLE PRECISION, f27 DOUBLE PRECISION, f28 DOUBLE PRECISION,
    f29 DOUBLE PRECISION, f30 DOUBLE PRECISION, f31 DOUBLE PRECISION, f32 DOUBLE PRECISION,
    f33 DOUBLE PRECISION, f34 DOUBLE PRECISION, f35 DOUBLE PRECISION, f36 DOUBLE PRECISION,
    f37 DOUBLE PRECISION, f38 DOUBLE PRECISION, f39 DOUBLE PRECISION, f40 DOUBLE PRECISION,
    f41 DOUBLE PRECISION, f42 DOUBLE PRECISION, f43 DOUBLE PRECISION, f44 DOUBLE PRECISION,
    f45 DOUBLE PRECISION, f46 DOUBLE PRECISION, f47 DOUBLE PRECISION, f48 DOUBLE PRECISION,
    f49 DOUBLE PRECISION, f50 DOUBLE PRECISION, f51 DOUBLE PRECISION, f52 DOUBLE PRECISION,
    f53 DOUBLE PRECISION, f54 DOUBLE PRECISION, f55 DOUBLE PRECISION, f56 DOUBLE PRECISION,
    f57 DOUBLE PRECISION, f58 DOUBLE PRECISION, f59 DOUBLE PRECISION, f60 DOUBLE PRECISION,
    f61 DOUBLE PRECISION, f62 DOUBLE PRECISION, f63 DOUBLE PRECISION, f64 DOUBLE PRECISION,
    f65 DOUBLE PRECISION, f66 DOUBLE PRECISION, f67 DOUBLE PRECISION, f68 DOUBLE PRECISION,
    f69 DOUBLE PRECISION, f70 DOUBLE PRECISION, f71 DOUBLE PRECISION, f72 DOUBLE PRECISION,
    f73 DOUBLE PRECISION, f74 DOUBLE PRECISION, f75 DOUBLE PRECISION, f76 DOUBLE PRECISION,
    f77 DOUBLE PRECISION, f78 DOUBLE PRECISION, f79 DOUBLE PRECISION, f80 DOUBLE PRECISION,
    f81 DOUBLE PRECISION, f82 DOUBLE PRECISION, f83 DOUBLE PRECISION, f84 DOUBLE PRECISION,
    f85 DOUBLE PRECISION, f86 DOUBLE PRECISION, f87 DOUBLE PRECISION, f88 DOUBLE PRECISION,
    f89 DOUBLE PRECISION, f90 DOUBLE PRECISION, f91 DOUBLE PRECISION, f92 DOUBLE PRECISION,
    f93 DOUBLE PRECISION, f94 DOUBLE PRECISION, f95 DOUBLE PRECISION, f96 DOUBLE PRECISION,
    f97 DOUBLE PRECISION, f98 DOUBLE PRECISION, f99 DOUBLE PRECISION, f100 DOUBLE PRECISION,
    f101 DOUBLE PRECISION, f102 DOUBLE PRECISION, f103 DOUBLE PRECISION, f104 DOUBLE PRECISION,
    f105 DOUBLE PRECISION, f106 DOUBLE PRECISION, f107 DOUBLE PRECISION, f108 DOUBLE PRECISION,
    f109 DOUBLE PRECISION, f110 DOUBLE PRECISION, f111 DOUBLE PRECISION, f112 DOUBLE PRECISION,
    f113 DOUBLE PRECISION, f114 DOUBLE PRECISION, f115 DOUBLE PRECISION, f116 DOUBLE PRECISION,
    f117 DOUBLE PRECISION, f118 DOUBLE PRECISION, f119 DOUBLE PRECISION, f120 DOUBLE PRECISION,
    f121 DOUBLE PRECISION, f122 DOUBLE PRECISION, f123 DOUBLE PRECISION, f124 DOUBLE PRECISION,
    f125 DOUBLE PRECISION, f126 DOUBLE PRECISION, f127 DOUBLE PRECISION, f128 DOUBLE PRECISION,
    -- Metadata
    data_source VARCHAR(50),
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
    data_quality_score DOUBLE PRECISION
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('raw_data.sensor_readings', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON raw_data.sensor_readings (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_gas_type ON raw_data.sensor_readings (gas_type);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_gas_label ON raw_data.sensor_readings (gas_label);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_batch_id ON raw_data.sensor_readings (batch_id);

-- ============================================================================
-- PROCESSED DATA TABLES (Silver Layer)
-- ============================================================================

-- Cleaned and validated sensor readings
CREATE TABLE IF NOT EXISTS processed_data.cleaned_sensor_readings (
    reading_id BIGSERIAL PRIMARY KEY,
    original_reading_id BIGINT REFERENCES raw_data.sensor_readings(reading_id),
    timestamp TIMESTAMPTZ NOT NULL,
    gas_type VARCHAR(50) NOT NULL,
    gas_label INTEGER NOT NULL,
    -- 8 primary features (after dimensionality reduction)
    feature_1 DOUBLE PRECISION,
    feature_2 DOUBLE PRECISION,
    feature_3 DOUBLE PRECISION,
    feature_4 DOUBLE PRECISION,
    feature_5 DOUBLE PRECISION,
    feature_6 DOUBLE PRECISION,
    feature_7 DOUBLE PRECISION,
    feature_8 DOUBLE PRECISION,
    -- Derived features
    concentration_ppm DOUBLE PRECISION,
    risk_score DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    threshold_exceeded BOOLEAN,
    -- Data quality flags
    is_outlier BOOLEAN DEFAULT FALSE,
    is_duplicate BOOLEAN DEFAULT FALSE,
    missing_values_count INTEGER DEFAULT 0,
    quality_check_passed BOOLEAN DEFAULT TRUE,
    processing_timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('processed_data.cleaned_sensor_readings', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_cleaned_readings_timestamp ON processed_data.cleaned_sensor_readings (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cleaned_readings_gas_type ON processed_data.cleaned_sensor_readings (gas_type);
CREATE INDEX IF NOT EXISTS idx_cleaned_readings_threshold ON processed_data.cleaned_sensor_readings (threshold_exceeded);

-- ============================================================================
-- ANALYTICS TABLES (Gold Layer)
-- ============================================================================

-- Predictions from ML models
CREATE TABLE IF NOT EXISTS analytics.predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    reading_id BIGINT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    predicted_gas_type VARCHAR(50),
    predicted_gas_label INTEGER,
    actual_gas_type VARCHAR(50),
    actual_gas_label INTEGER,
    confidence_score DOUBLE PRECISION,
    is_correct BOOLEAN,
    -- Model-specific outputs
    probabilities JSONB,
    feature_importance JSONB,
    prediction_time_ms DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('analytics.predictions', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON analytics.predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON analytics.predictions (model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_accuracy ON analytics.predictions (is_correct);

-- Aggregated statistics
CREATE TABLE IF NOT EXISTS analytics.hourly_statistics (
    stat_id BIGSERIAL PRIMARY KEY,
    hour_timestamp TIMESTAMPTZ NOT NULL,
    gas_type VARCHAR(50),
    total_readings INTEGER,
    avg_concentration DOUBLE PRECISION,
    max_concentration DOUBLE PRECISION,
    min_concentration DOUBLE PRECISION,
    std_concentration DOUBLE PRECISION,
    threshold_exceedances INTEGER,
    anomaly_count INTEGER,
    avg_risk_score DOUBLE PRECISION
);

-- Convert to hypertable
SELECT create_hypertable('analytics.hourly_statistics', 'hour_timestamp', if_not_exists => TRUE);

-- ============================================================================
-- MONITORING TABLES
-- ============================================================================

-- Pipeline execution logs
CREATE TABLE IF NOT EXISTS monitoring.pipeline_executions (
    execution_id BIGSERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    dag_id VARCHAR(100),
    task_id VARCHAR(100),
    execution_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50), -- 'running', 'success', 'failed'
    records_processed INTEGER,
    duration_seconds DOUBLE PRECISION,
    error_message TEXT,
    execution_metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_pipeline_executions_timestamp ON monitoring.pipeline_executions (execution_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_status ON monitoring.pipeline_executions (status);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS monitoring.data_quality_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    table_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value DOUBLE PRECISION,
    threshold_min DOUBLE PRECISION,
    threshold_max DOUBLE PRECISION,
    is_within_threshold BOOLEAN,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_dq_metrics_timestamp ON monitoring.data_quality_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_dq_metrics_table ON monitoring.data_quality_metrics (table_name);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    inference_time_avg_ms DOUBLE PRECISION,
    samples_evaluated INTEGER,
    confusion_matrix JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON monitoring.model_performance (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON monitoring.model_performance (model_name);

-- ============================================================================
-- REFERENCE DATA
-- ============================================================================

-- Gas type reference table
CREATE TABLE IF NOT EXISTS processed_data.gas_types (
    gas_id INTEGER PRIMARY KEY,
    gas_name VARCHAR(50) UNIQUE NOT NULL,
    chemical_formula VARCHAR(50),
    danger_type VARCHAR(50),
    threshold_ppm DOUBLE PRECISION,
    color_code VARCHAR(20),
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert gas types
INSERT INTO processed_data.gas_types (gas_id, gas_name, chemical_formula, danger_type, threshold_ppm, color_code, description)
VALUES
    (0, 'Ammonia', 'NH3', 'Toxic', 25, '#3b82f6', 'Toxic gas, pungent smell'),
    (1, 'Acetaldehyde', 'C2H4O', 'Irritant', 50, '#8b5cf6', 'Irritant, fruity smell'),
    (2, 'Acetone', 'C3H6O', 'Flammable', 750, '#ec4899', 'Flammable solvent'),
    (3, 'Ethanol', 'C2H5OH', 'Flammable', 1000, '#10b981', 'Flammable alcohol'),
    (4, 'Ethylene', 'C2H4', 'Asphyxiant', 100, '#f59e0b', 'Simple asphyxiant'),
    (5, 'Toluene', 'C7H8', 'Toxic', 50, '#ef4444', 'Toxic aromatic hydrocarbon')
ON CONFLICT (gas_id) DO NOTHING;

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- Recent readings view
CREATE OR REPLACE VIEW analytics.recent_readings AS
SELECT
    cr.reading_id,
    cr.timestamp,
    gt.gas_name,
    cr.gas_label,
    cr.concentration_ppm,
    cr.risk_score,
    cr.threshold_exceeded,
    gt.threshold_ppm,
    cr.anomaly_score,
    cr.quality_check_passed
FROM processed_data.cleaned_sensor_readings cr
JOIN processed_data.gas_types gt ON cr.gas_label = gt.gas_id
WHERE cr.timestamp > NOW() - INTERVAL '24 hours'
ORDER BY cr.timestamp DESC;

-- Model accuracy comparison view
CREATE OR REPLACE VIEW analytics.model_accuracy_comparison AS
SELECT
    model_name,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(AVG(CASE WHEN is_correct THEN 100.0 ELSE 0.0 END), 2) as accuracy_percentage,
    ROUND(AVG(confidence_score), 4) as avg_confidence,
    ROUND(AVG(prediction_time_ms), 2) as avg_prediction_time_ms
FROM analytics.predictions
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY model_name
ORDER BY accuracy_percentage DESC;

-- Gas distribution statistics
CREATE OR REPLACE VIEW analytics.gas_distribution_stats AS
SELECT
    gt.gas_name,
    gt.gas_id,
    COUNT(*) as reading_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(AVG(cr.concentration_ppm), 2) as avg_concentration,
    ROUND(MAX(cr.concentration_ppm), 2) as max_concentration,
    gt.threshold_ppm,
    SUM(CASE WHEN cr.threshold_exceeded THEN 1 ELSE 0 END) as threshold_exceedances
FROM processed_data.cleaned_sensor_readings cr
JOIN processed_data.gas_types gt ON cr.gas_label = gt.gas_id
WHERE cr.timestamp > NOW() - INTERVAL '24 hours'
GROUP BY gt.gas_name, gt.gas_id, gt.threshold_ppm
ORDER BY reading_count DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to calculate data quality score
CREATE OR REPLACE FUNCTION calculate_data_quality_score(
    p_reading_id BIGINT
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_score DOUBLE PRECISION := 100.0;
    v_null_count INTEGER;
BEGIN
    -- Check for null values in features
    SELECT
        (CASE WHEN f1 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f2 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f3 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f4 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f5 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f6 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f7 IS NULL THEN 1 ELSE 0 END +
         CASE WHEN f8 IS NULL THEN 1 ELSE 0 END)
    INTO v_null_count
    FROM raw_data.sensor_readings
    WHERE reading_id = p_reading_id;

    -- Reduce score based on missing values
    v_score := v_score - (v_null_count * 10.0);

    -- Ensure score is between 0 and 100
    v_score := GREATEST(0.0, LEAST(100.0, v_score));

    RETURN v_score;
END;
$$ LANGUAGE plpgsql;

-- Function to detect anomalies
CREATE OR REPLACE FUNCTION detect_anomaly(
    p_value DOUBLE PRECISION,
    p_mean DOUBLE PRECISION,
    p_std DOUBLE PRECISION,
    p_threshold DOUBLE PRECISION DEFAULT 3.0
) RETURNS BOOLEAN AS $$
BEGIN
    IF p_std = 0 THEN
        RETURN FALSE;
    END IF;

    RETURN ABS(p_value - p_mean) > (p_threshold * p_std);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CONTINUOUS AGGREGATES (for faster queries)
-- ============================================================================

-- Create continuous aggregate for hourly stats
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.readings_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    gas_label,
    COUNT(*) as reading_count,
    AVG(concentration_ppm) as avg_concentration,
    MAX(concentration_ppm) as max_concentration,
    MIN(concentration_ppm) as min_concentration,
    STDDEV(concentration_ppm) as std_concentration,
    SUM(CASE WHEN threshold_exceeded THEN 1 ELSE 0 END) as exceedances
FROM processed_data.cleaned_sensor_readings
GROUP BY hour, gas_label;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('analytics.readings_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- ============================================================================
-- RETENTION POLICIES (for data lifecycle management)
-- ============================================================================

-- Keep raw data for 90 days
SELECT add_retention_policy('raw_data.sensor_readings', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep processed data for 1 year
SELECT add_retention_policy('processed_data.cleaned_sensor_readings', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep predictions for 6 months
SELECT add_retention_policy('analytics.predictions', INTERVAL '6 months', if_not_exists => TRUE);

-- ============================================================================
-- GRANTS (for security)
-- ============================================================================

-- Grant permissions to application user
GRANT USAGE ON SCHEMA raw_data, processed_data, analytics, monitoring TO sensordrift;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA raw_data, processed_data, analytics, monitoring TO sensordrift;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA raw_data, processed_data, analytics, monitoring TO sensordrift;

-- Create Metabase database
CREATE DATABASE metabase;

-- Create read-only user for analytics tools
CREATE USER analytics_readonly WITH PASSWORD 'analytics_readonly';
GRANT CONNECT ON DATABASE gas_monitoring TO analytics_readonly;
GRANT USAGE ON SCHEMA analytics, processed_data TO analytics_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics, processed_data TO analytics_readonly;

-- Finish
SELECT 'Database initialization completed successfully!' as status;
