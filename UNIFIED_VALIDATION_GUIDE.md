# Unified Validation Guide

## Overview

The **Unified Validation System** consolidates all anomaly detection techniques (traditional, ML, and advanced AI) into a single, production-ready script. This eliminates the need for multiple validation workflows and provides a single source of truth for anomaly detection.

## Single Source of Truth

**Master Script:** [`scripts/unified_validation.py`](scripts/unified_validation.py)

This script includes **11 detection methods** organized into 3 categories:

### Detection Methods (11 Total)

#### 1. Traditional Methods (2)
- **Rule-Based**: Custom business logic validation
- **IQR (Interquartile Range)**: Statistical baseline detection

#### 2. Machine Learning (3)
- **Isolation Forest**: Tree-based anomaly detection (sklearn)
- **K-Means Clustering**: Unsupervised clustering anomalies
- **Autoencoder**: Deep neural network reconstruction error

#### 3. Advanced AI (6)
- **Fuzzy Logic**: Membership-based pattern recognition
- **Expert System**: Rule-based inference engine
- **Time Series Forecasting**: ARIMA-based anomaly detection
- **Genetic Algorithm**: Evolutionary feature optimization
- **Ensemble AI**: Multi-method consensus voting
- **Neural-Symbolic**: Hybrid neural networks with symbolic reasoning

---

## Quick Start

### CSV File Validation

```bash
python scripts/unified_validation.py --csv data/cleaned_data.csv --output report.json
```

### Database Validation

```bash
python scripts/unified_validation.py --db data/transactions.db --table transactions --output report.json
```

---

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | str | - | Path to CSV input file |
| `--db` | str | - | Path to SQLite database file |
| `--table` | str | `transactions` | Table name in database (only with `--db`) |
| `--output` | str | - | JSON report output path |
| `--compare` | flag | - | Show detailed method comparison table |

**Note:** Use either `--csv` OR `--db`, not both.

---

## Usage Examples

### 1. Basic CSV Validation
```bash
python scripts/unified_validation.py --csv data/cleaned_data.csv
```
Runs all 11 methods on CSV file, displays results in console.

### 2. CSV with Report Output
```bash
python scripts/unified_validation.py --csv data/cleaned_data.csv --output validation_report.json
```
Saves detailed JSON report with all method results and timing.

### 3. Database Validation
```bash
python scripts/unified_validation.py --db data/transactions.db --table transactions
```
Connects to SQLite database and validates the specified table.

### 4. Database with Detailed Comparison
```bash
python scripts/unified_validation.py --db data/transactions.db --table transactions --compare
```
Includes detailed comparison table showing anomaly counts, percentages, and execution times.

### 5. Custom Report Path
```bash
python scripts/unified_validation.py --csv data/cleaned_data.csv --output /reports/2024-12-26-validation.json
```
Saves report to specified directory.

---

## Output Format

### Console Output

The script displays a formatted report with 6 sections:

**[1] DATA LOADING**
- Source information (CSV path or DB connection)
- Record count
- Column count
- Numeric columns identified

**[2] TRADITIONAL VALIDATION**
- Rule-Based detection results
- IQR detection results

**[3] MACHINE LEARNING VALIDATION**
- Isolation Forest results
- K-Means clustering results
- Autoencoder results

**[4] ADVANCED AI VALIDATION**
- Fuzzy Logic results
- Expert System results
- Time Series Forecasting results
- Genetic Algorithm results
- Ensemble AI results
- Neural-Symbolic results

**[5] STATISTICS**
- Average anomalies across all methods
- Min/Max anomaly counts
- Standard deviation
- Total execution time
- Per-method timing

**[6] SAVING REPORT**
- JSON file path (if `--output` specified)
- Confirmation message

### JSON Report Format

```json
{
  "timestamp": "2025-12-26T20:39:26.193732",
  "data_source": "CSV: data/cleaned_data.csv",
  "records": 47600,
  "columns": 8,
  "numeric_columns": ["id", "transaction_amount", "account_balance", "risk_score"],
  "results": {
    "RULE_BASED": {
      "anomalies": 0,
      "time": 0.003958
    },
    "IQR": {
      "anomalies": 100,
      "time": 0.011988
    },
    ...
  }
}
```

---

## Data Source Configuration

### CSV Files

**Requirements:**
- Must have a header row with column names
- Numeric columns will be automatically detected
- Supports standard CSV format (comma-separated)

**Example Structure:**
```csv
id,timestamp,transaction_amount,account_balance,status,risk_score,category,is_flagged
1,2024-01-01,150.50,5000.00,completed,0.2,online,false
2,2024-01-01,3500.00,4800.00,completed,0.8,wire,true
```

**Supported CSV Dialects:**
- Standard (comma-separated)
- Tab-separated (auto-detected)
- Semicolon-separated (auto-detected)

### SQLite Database

**Requirements:**
- Valid SQLite3 database file (.db)
- Table must have at least one numeric column
- Column names must be unique

**Current Implementation:**
- Reads from any table with `SELECT * FROM table_name`
- Auto-detects numeric columns
- Optimized for transaction/event data

**Example Table:**
```sql
CREATE TABLE transactions (
  id INTEGER PRIMARY KEY,
  timestamp TEXT,
  transaction_amount REAL,
  account_balance REAL,
  status TEXT,
  risk_score REAL,
  category TEXT,
  is_flagged INTEGER
);
```

---

## Future: Git Secrets Integration

The script includes a **prepared code section** for secure database credential management using Git Secrets (or other secret managers).

### Enabling Git Secrets Integration

**Location:** Lines 44-59 in `scripts/unified_validation.py`

**Current State:** Code is commented out and includes:
1. Git Secrets environment variable loading
2. Database connection using secrets
3. Error handling for missing credentials

### Setup Instructions (When Ready)

1. **Install GitHub CLI:**
   ```bash
   sudo apt-get install gh
   ```

2. **Configure Secrets in GitHub (for CI/CD):**
   ```bash
   gh secret set DB_HOST --body "your-host"
   gh secret set DB_USER --body "your-user"
   gh secret set DB_PASSWORD --body "your-password"
   gh secret set DB_NAME --body "your-database"
   ```

3. **Uncomment the following lines in `scripts/unified_validation.py`:**
   ```python
   # Lines 44-59: Git Secrets integration code
   ```

4. **Update the database connection call:**
   ```python
   # Change from SQLite to PostgreSQL/MySQL/etc.
   self.connection = psycopg2.connect(...)
   ```

### Supported Secret Managers

The prepared code structure supports:
- **GitHub Secrets** (built-in for Actions)
- **Azure Key Vault** (enterprise)
- **HashiCorp Vault** (open-source)
- **AWS Secrets Manager** (cloud)

---

## Performance Characteristics

### Execution Time by Method

**Fastest:**
- Rule-Based: ~0.003s (instant)
- Genetic Algorithm: ~0.1s

**Medium Speed:**
- IQR: ~0.01s
- Fuzzy Logic: ~0.7s
- Expert System: ~0.9s
- Time Series: ~0.4s
- Ensemble AI: ~1.3s
- Isolation Forest: ~0.5s
- K-Means: ~1.0s

**Slowest:**
- Autoencoder: ~20s (deep learning)
- Neural-Symbolic: ~10s (complex inference)

**Total Runtime (typical):** 35-40 seconds for 50,000 records

### Memory Usage

- **CSV Loading**: ~50MB for 50,000 records
- **Model Training**: ~200MB peak (Autoencoder)
- **Report Generation**: <10MB

---

## Integration with CI/CD

### GitHub Actions

The unified validation is automatically triggered by:
- **Push to main/develop** branches
- **Pull Request** creation
- **Manual trigger** (workflow_dispatch)
- **Daily schedule** (2 AM UTC)

**File:** [`.github/workflows/unified-validation.yml`](.github/workflows/unified-validation.yml)

### Running Validation in Workflows

**CSV Validation:**
```yaml
- name: Run CSV Validation
  run: python scripts/unified_validation.py --csv data/cleaned_data.csv --output report.json
```

**Database Validation (with secrets):**
```yaml
- name: Run Database Validation
  env:
    DB_PATH: ${{ secrets.DB_PATH }}
  run: python scripts/unified_validation.py --db $DB_PATH --table transactions --output report.json
```

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'pandas'"**
```bash
pip install -r requirements.txt
```

**Issue: "No numeric columns found"**
- CSV/table must contain at least one numeric column (int or float)
- Check column data types: `df.dtypes`
- Verify no string-based numbers: convert to numeric first

**Issue: "Memory error with large datasets"**
- Process files in chunks
- Use database instead of CSV for datasets >1GB
- Increase system RAM or use cloud processing

**Issue: "Database connection failed"**
- Verify file path: `ls -la data/transactions.db`
- Check table exists: `sqlite3 data/transactions.db ".tables"`
- Ensure database is not locked by another process

**Issue: "Permission denied" on output file**
- Check directory permissions: `chmod 755 output_dir/`
- Ensure write access to output path
- Try specifying full path: `--output /tmp/report.json`

---

## Method Descriptions

### Traditional Methods

#### Rule-Based Validation
- Custom business logic checks
- Examples: transaction limits, status rules, flag conditions
- **Use Case**: Enforcing known constraints

#### IQR (Interquartile Range)
- Statistical baseline using Q1, Q3, and IQR bounds
- Flags values beyond 1.5 × IQR from quartiles
- **Use Case**: Quick outlier detection

### Machine Learning Methods

#### Isolation Forest
- Tree-based algorithm isolating anomalies
- Efficient for high-dimensional data
- **Use Case**: Fast anomaly detection with good accuracy

#### K-Means Clustering
- Unsupervised clustering with anomaly scoring
- Points far from cluster centers are flagged
- **Use Case**: Behavioral clustering and outliers

#### Autoencoder
- Deep neural network reconstruction error method
- Trains on normal data, flags reconstruction errors
- **Use Case**: Complex pattern learning from examples

### Advanced AI Methods

#### Fuzzy Logic
- Membership functions for pattern matching
- Soft classification (0-1 confidence scores)
- **Use Case**: Human-interpretable rules with uncertainty

#### Expert System
- Rule-based inference engine
- Knowledge base of domain expertise
- **Use Case**: Encoding expert knowledge systematically

#### Time Series Forecasting
- ARIMA models for temporal data
- Flags deviations from forecast
- **Use Case**: Sequential/temporal anomalies

#### Genetic Algorithm
- Evolutionary optimization of detection features
- Population-based search for patterns
- **Use Case**: Feature engineering and optimization

#### Ensemble AI
- Consensus voting from multiple AI methods
- Flags only high-confidence anomalies
- **Use Case**: Robust detection with high precision

#### Neural-Symbolic
- Hybrid neural networks with symbolic reasoning
- Combines deep learning with logic rules
- **Use Case**: Complex patterns with interpretable rules

---

## Best Practices

1. **Choose Data Source Wisely**
   - CSV: For exploration, ad-hoc validation, small datasets
   - Database: For production, large datasets, continuous monitoring

2. **Validation Frequency**
   - Development: Run ad-hoc as needed
   - Staging: Daily validation via scheduled workflow
   - Production: Real-time via integrated monitoring

3. **Review Results**
   - Compare method outputs (use `--compare` flag)
   - Investigate high-variance methods
   - Validate against labeled anomalies if available

4. **Method Selection**
   - Use ensemble for production (high precision)
   - Use Isolation Forest for speed
   - Use Autoencoder for complex patterns
   - Use multiple methods and compare

5. **Report Management**
   - Save timestamped reports: `validation_$(date +%Y%m%d).json`
   - Archive reports for compliance/audit
   - Set up monitoring alerts for anomaly spikes

---

## Dependencies

See [`requirements.txt`](requirements.txt) for full list:

**Core:**
- pandas
- numpy
- scikit-learn
- scipy

**Optional (AI methods):**
- tensorflow (for Autoencoder and Neural-Symbolic)
- statsmodels (for Time Series)

**Database:**
- sqlite3 (built-in with Python)

---

## File Structure

```
ETL_AnomalyDetection_AI/
├── scripts/
│   └── unified_validation.py          ← Master validation script
├── src/
│   ├── validation/
│   │   ├── anomaly_detector.py         ← AnomalyDetector class
│   │   ├── ml_anomaly.py               ← ML methods
│   │   ├── rule_validator.py           ← Traditional methods
│   │   └── ai_techniques.py            ← AI techniques
│   └── ...
├── data/
│   ├── cleaned_data.csv                ← CSV validation data
│   └── transactions.db                 ← SQLite validation database
└── .github/
    └── workflows/
        └── unified-validation.yml      ← GitHub Actions workflow
```

---

## Support & Documentation

- **Requirements:** [`requirements.txt`](requirements.txt)
- **AI Techniques Details:** See method docstrings in `src/validation/anomaly_detector.py`
- **GitHub Workflow:** [`.github/workflows/unified-validation.yml`](.github/workflows/unified-validation.yml)
- **Main README:** [`README.md`](README.md)

---

## Version History

- **v1.0** (2025-12-26): Initial unified validation system
  - Consolidated 11 detection methods
  - CSV and SQLite support
  - JSON report generation
  - Git Secrets integration template
  - GitHub Actions automation

---

## License

Part of ETL Anomaly Detection AI Framework
