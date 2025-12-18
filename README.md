AIETLTest

An AI-powered ETL testing framework for data processing with built-in anomaly detection and regulatory compliance validation.

## Features

- **Data Source Support**: Processes CSV files and database tables
- **Comprehensive Anomaly Detection**: Rule-based and statistical outlier detection
- **Regulatory Compliance**: Identifies AML, sanctions, and financial crime patterns
- **Detailed Reporting**: Generates HTML reports with categorized findings and severity scoring
- **Testing Framework**: Preserves all anomalies for continuous testing and validation
- **Automated CI/CD**: GitHub Actions workflows for automated testing and reporting

## Project Structure

```
AIETLTest/
├── .github/
│   └── workflows/
│       ├── etl-workflow.yml         # GitHub Actions CI/CD for CSV processing
│       └── db-testing-workflow.yml  # Database anomaly testing workflow
├── data/
│   ├── synthetic_data.csv           # Input synthetic dataset
│   ├── test_data_with_anomalies.csv # Output dataset with anomalies preserved (CSV)
│   └── transactions.db              # SQLite database for DB operations
├── logs/
│   ├── csv_anomaly_report.html      # HTML report for CSV processing
│   └── db_anomaly_report.html       # HTML report for database scanning
├── src/
│   ├── orchestrator.py         # Main ETL orchestration script (CSV)
│   ├── db_scanner.py           # Database anomaly scanning script
│   ├── setup_db.py             # Database setup from CSV
│   ├── add_anomalies.py        # Add regulatory anomalies to database
│   └── validation/
│       ├── anomaly_detector.py # Statistical anomaly detection
│       └── rule_validator.py   # Rule-based data validation
├── tests/
│   ├── test_anomaly.py         # Unit tests for anomaly detection
│   └── test_validation.py      # Unit tests for validation rules
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
   
   git clone https://github.com/hkrishnan62/AIETLTest.git
   cd AIETLTest
   

2. Create a virtual environment:
   
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  

3. Install dependencies:
   
   pip install -r requirements.txt
   

## Usage

### CSV Testing Framework

Execute the main orchestrator script for CSV anomaly detection:

```bash
cd src
python orchestrator.py
```

This will:
- Extract data from `../data/synthetic_data.csv`
- Apply comprehensive anomaly detection (rules + statistics)
- Classify findings by regulatory categories (Money Laundering, Structuring, etc.)
- Generate detailed HTML report with severity scoring
- **Preserve all data** including anomalies for testing purposes
- Save complete dataset to `../data/test_data_with_anomalies.csv`

### Database Anomaly Scanning

For database input, first set up the database from CSV:

```bash
cd src
python setup_db.py
```

Then run the database scanner:

```bash
python db_scanner.py
```

This will:
- Connect to the SQLite database at `../data/transactions.db`
- Scan the `transactions` table for anomalies
- Generate detailed HTML report with severity scoring and regulatory classifications
- **Preserve all data** including anomalies for testing purposes
- Save HTML report to `../logs/db_anomaly_report.html`

Validation Rules

The pipeline validates:
- **Required Columns**: `id`, `report_date`, `transaction_amount`, `account_type`, `account_balance`, `region`
- **Numeric Ranges**:
  - `transaction_amount`: 0-15,000
  - `account_balance`: 0-70,000
- **Categories**: `account_type` must be one of `['Retail', 'Corporate', 'Investment']`

### Anomaly Detection

Uses Interquartile Range (IQR) method with a factor of 1.5 to detect statistical outliers in `transaction_amount` and `account_balance` columns.

## Testing

Activate the virtual environment and run the test suite:


source /workspaces/.venv/bin/activate  # Or use the full path if not activated
/workspaces/.venv/bin/python -m pytest tests/


Or run individual test files:

/workspaces/.venv/bin/python -m pytest tests/test_validation.py
/workspaces/.venv/bin/python -m pytest tests/test_anomaly.py



## CI/CD

The project includes a GitHub Actions workflow that:
- Runs on pushes and pull requests to `main`
- Sets up Python environment
- Installs dependencies
- Executes the ETL pipeline
- Generates HTML reports
- Uploads logs as artifacts

### Manual Workflow Trigger

You can manually trigger the workflow from the GitHub Actions tab.

### Database Testing Workflow

The project includes a separate workflow for database anomaly testing:

- **Trigger**: Manual (`workflow_dispatch`)
- **Purpose**: Comprehensive database scanning with detailed regulatory anomaly reports
- **Features**: 
  - Detailed anomaly classification (Money Laundering, Structuring, etc.)
  - Severity scoring (Critical, High, Medium, Low)
  - HTML report with categorized findings
  - No data cleaning - preserves anomalies for testing

To run database testing:
1. Go to Actions tab
2. Select "Database Anomaly Testing"
3. Click "Run workflow"
4. Download the `db-anomaly-report` artifact

### Viewing Reports

After workflow execution:
1. Go to the Actions tab
2. Select the latest workflow run
3. Download artifacts from the "Artifacts" section
4. Open `etl_report.html` or `db_anomaly_report.html` in a web browser for formatted reports

## Data Description

The synthetic dataset includes:
- **id**: Unique transaction identifier
- **report_date**: Transaction timestamp
- **transaction_amount**: Transaction value
- **account_balance**: Account balance after transaction
- **risk_score**: Risk assessment score
- **account_age**: Account age in months
- **account_type**: Type of account (`Retail`, `Corporate`, `Investment`)
- **region**: Geographic region (`APAC`, `EU`, `US`)

## Database Support

The project supports SQLite databases for data processing. Use `setup_db.py` to create a database from the CSV file, then use `db_scanner.py` for anomaly detection on database tables. Use `add_anomalies.py` to add various regulatory anomalies (money laundering patterns, structuring, high-risk transactions, etc.) for testing the detection capabilities. The database scanner generates detailed HTML reports with anomaly classifications and severity scoring, making it ideal for compliance testing and regulatory monitoring.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

