# 📈 Market Dashboard

A comprehensive Python-based market data analysis and visualization platform with automated data updates and AI-powered insights.

## 🌟 Features

### 📊 **Interactive Dashboards**
- **Single Stock Analysis** - Detailed analysis of individual stocks with KPIs, charts, and AI insights
- **Multi-Stock Comparison** - Compare multiple stocks with correlation analysis and performance metrics
- **Price Forecasting** - AI-powered price predictions using multiple forecasting models

### 🤖 **AI-Powered Insights**
- **Market Analysis** - Automated insights using Ollama LLM integration
- **Technical Indicators** - RSI, MACD, Bollinger Bands, and more
- **Sentiment Analysis** - AI-generated market commentary and recommendations

### 🔄 **Automated Data Management**
- **Daily Updates** - Automated fetching and loading of market data
- **Email Notifications** - Get notified when updates succeed or fail
- **Data Validation** - Ensures data quality and consistency
- **Backup System** - Automatic database backups

### 📈 **Data Sources**
- **Yahoo Finance** - Real-time and historical stock data
- **Multiple Assets** - Stocks, ETFs, and market indices
- **Technical Indicators** - Comprehensive technical analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Internet connection for data fetching

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd market-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   
   # Windows
   myenv\Scripts\activate
   
   # Linux/macOS
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initial data setup**
   ```bash
   # Fetch initial data
   python scripts/fetch_data.py
   
   # Load data to database
   python scripts/load_to_db.py
   ```

5. **Run the dashboard**
   ```bash
   python main.py
   ```

6. **Open your browser**
   ```
   http://localhost:8050
   ```

## 📁 Project Structure

```
market-dashboard/
├── src/
│   └── market_dashboard/
│       ├── app.py                 # Main Dash application
│       ├── data/                  # Data management modules
│       │   ├── fetcher.py         # Data fetching from APIs
│       │   ├── loader.py          # Database loading
│       │   └── database.py        # Database operations
│       ├── components/            # UI components
│       │   ├── layouts.py         # Dashboard layouts
│       │   ├── callbacks.py       # Interactive callbacks
│       │   └── charts.py          # Chart generation
│       ├── ai/                    # AI and analysis
│       │   └── insights.py        # AI-powered insights
│       └── utils/                 # Utility functions
│           ├── data_processing.py # Data manipulation
│           └── formatting.py      # Display formatting
├── scripts/                       # Automation scripts
│   ├── fetch_data.py             # Manual data fetching
│   ├── load_to_db.py             # Manual data loading
│   ├── auto_daily_update.py      # Automated daily updates
│   ├── test_email.py             # Email testing
│   └── run_daily_update.bat      # Windows scheduler script
├── config/                        # Configuration files
│   ├── settings.py               # Application settings
│   └── email_config.json         # Email notification settings
├── data/                         # Data storage
│   ├── market.db                 # SQLite database
│   └── saved_data/               # Raw and processed data files
├── logs/                         # Application logs
├── tests/                        # Unit tests
├── main.py                       # Application entry point
└── requirements.txt              # Python dependencies
```

## 🎯 Core Components

### 1. **Data Management**
- **Fetcher** (`src/market_dashboard/data/fetcher.py`) - Downloads data from Yahoo Finance
- **Loader** (`src/market_dashboard/data/loader.py`) - Processes and stores data in SQLite
- **Database** (`src/market_dashboard/data/database.py`) - Database operations and queries

### 2. **Web Interface**
- **App** (`src/market_dashboard/app.py`) - Main Dash application setup
- **Layouts** (`src/market_dashboard/components/layouts.py`) - UI layout definitions
- **Callbacks** (`src/market_dashboard/components/callbacks.py`) - Interactive functionality
- **Charts** (`src/market_dashboard/components/charts.py`) - Visualization components

### 3. **AI Integration**
- **Insights** (`src/market_dashboard/ai/insights.py`) - AI-powered market analysis
- **Ollama Integration** - Local LLM for generating insights and recommendations

### 4. **Automation**
- **Auto Update** (`scripts/auto_daily_update.py`) - Automated daily data updates
- **Email Notifications** - Success/failure alerts via email
- **Scheduling** - Windows Task Scheduler and cron support

## 📊 Dashboard Features

### Single Stock Analysis
- **Key Performance Indicators** - Price, volume, volatility metrics
- **Interactive Charts** - Price/volume charts with technical indicators
- **AI Insights** - Automated analysis and recommendations
- **Export Functionality** - Download data as CSV

### Multi-Stock Comparison
- **Performance Comparison** - Normalized price comparisons
- **Correlation Analysis** - Heatmaps showing stock correlations
- **Statistical Analysis** - Returns distribution and volatility analysis
- **Comparative Insights** - AI-powered comparative analysis

### Price Forecasting
- **Multiple Models** - Linear regression, ARIMA, and more
- **Configurable Periods** - 7, 14, 30, or 60-day forecasts
- **Confidence Intervals** - Statistical confidence in predictions
- **AI Commentary** - Interpretation of forecast results

## 🔧 Configuration

### Application Settings (`config/settings.py`)
```python
# Database configuration
DB_PATH = "market.db"

# Data sources
STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"]
LOOKBACK_DAYS_STOCKS = 730

# AI configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
```

### Email Notifications (`config/email_config.json`)
```json
{
  "enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "sender_email": "your-email@gmail.com",
  "sender_password": "your-app-password",
  "recipient_emails": ["recipient@example.com"],
  "send_on_success": true,
  "send_on_failure": true
}
```

## 🤖 Automation Setup

### Daily Data Updates

1. **Configure Email Notifications** (Optional)
   ```bash
   python scripts/auto_daily_update.py --create-email-config
   # Edit config/email_config.json with your settings
   ```

2. **Test the Update Process**
   ```bash
   python scripts/auto_daily_update.py
   ```

3. **Schedule Daily Updates**

   **Windows (Task Scheduler):**
   - Create task to run `scripts/run_daily_update.bat` daily
   - Set working directory to project root

   **Linux/macOS (Cron):**
   ```bash
   # Add to crontab for daily 6 PM execution
   0 18 * * * cd "/path/to/market-dashboard" && python scripts/auto_daily_update.py
   ```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Test specific component
python -m pytest tests/test_imports.py

# Test email functionality
python scripts/test_email.py
```

### Manual Testing
```bash
# Test data fetching
python scripts/fetch_data.py

# Test data loading
python scripts/load_to_db.py

# Test dashboard
python main.py
```

## 📦 Dependencies

### Core Dependencies
- **Dash** - Web framework for Python
- **Plotly** - Interactive visualization library
- **Pandas** - Data manipulation and analysis
- **yfinance** - Yahoo Finance data fetching
- **SQLite** - Lightweight database

### AI Dependencies
- **requests** - HTTP library for Ollama integration
- **scikit-learn** - Machine learning algorithms
- **statsmodels** - Statistical analysis

### Full dependency list in `requirements.txt`

## 🔒 Security Considerations

- **Email Passwords** - Use app-specific passwords, not regular passwords
- **Configuration Files** - Added to `.gitignore` to prevent credential exposure
- **Database** - Local SQLite database for data security
- **API Keys** - No API keys required for Yahoo Finance

## 🛠️ Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check virtual environment is activated
- Verify all dependencies are installed
- Ensure port 8050 is available

**Data fetching fails:**
- Check internet connection
- Verify Yahoo Finance is accessible
- Check stock ticker symbols are valid

**Email notifications not working:**
- Use app passwords for Gmail
- Check SMTP settings
- Verify firewall allows SMTP connections

**AI insights not generating:**
- Ensure Ollama is running on localhost:11434
- Check Ollama model is installed
- Verify network connectivity to Ollama

## 📈 Usage Examples

### Basic Usage
```bash
# Start the dashboard
python main.py

# Navigate to http://localhost:8050
# Select stocks and generate analysis
```

### Automated Updates
```bash
# Set up automation
python scripts/auto_daily_update.py --create-email-config

# Test automation
python scripts/auto_daily_update.py

# Schedule with Task Scheduler or cron
```

### Custom Analysis
```python
from market_dashboard.data import DatabaseManager
from market_dashboard.ai import InsightsGenerator

# Load data
db = DatabaseManager("market.db")
data = db.load_market_data()

# Generate insights
insights = InsightsGenerator()
analysis = insights.generate_market_insights(data, "AAPL", "D")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚀 Advanced Features

### Technical Indicators
- **RSI (Relative Strength Index)** - Momentum oscillator
- **MACD (Moving Average Convergence Divergence)** - Trend following indicator
- **Bollinger Bands** - Volatility indicator
- **Volume Analysis** - Trading volume patterns
- **Price Patterns** - Support and resistance levels

### AI Capabilities
- **Market Sentiment Analysis** - AI interpretation of market conditions
- **Trend Identification** - Automated trend detection and analysis
- **Risk Assessment** - Volatility and risk metrics
- **Performance Predictions** - Short-term price forecasting
- **Comparative Analysis** - Multi-stock performance evaluation

### Data Quality Features
- **Data Validation** - Ensures data integrity and consistency
- **Missing Data Handling** - Intelligent gap filling and interpolation
- **Outlier Detection** - Identifies and handles anomalous data points
- **Historical Data Management** - Efficient storage and retrieval of time series data

## 🔧 Customization

### Adding New Stocks
Edit `config/settings.py`:
```python
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "SPY", "QQQ",
    "YOUR_NEW_STOCK"  # Add your stocks here
]
```

### Custom Technical Indicators
Extend `src/market_dashboard/utils/data_processing.py`:
```python
def custom_indicator(df):
    # Your custom indicator logic
    return df
```

### Custom AI Models
Extend `src/market_dashboard/ai/insights.py`:
```python
def custom_analysis(self, data, symbol):
    # Your custom AI analysis
    return insights
```

## 📊 Performance

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for data and logs
- **CPU**: Any modern processor
- **Network**: Stable internet for data fetching

### Performance Metrics
- **Data Fetching**: ~2-5 seconds for 9 stocks
- **Dashboard Loading**: ~1-3 seconds
- **AI Insights**: ~3-10 seconds depending on model
- **Database Operations**: <1 second for typical queries

### Optimization Tips
- **Regular Cleanup**: Remove old log files and backups
- **Database Maintenance**: Periodic VACUUM operations
- **Memory Management**: Restart dashboard periodically for long-running instances
- **Network Optimization**: Use wired connection for data fetching

## 🔍 Monitoring

### Log Files
- **Application Logs**: `logs/` directory
- **Update Logs**: `logs/auto_update_YYYYMMDD.log`
- **Error Tracking**: Automatic error logging and reporting

### Health Checks
```bash
# Check database status
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Records: {db.get_total_record_count()}')"

# Check latest data
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Latest: {db.get_latest_date()}')"

# Test AI integration
python -c "from market_dashboard.ai import InsightsGenerator; ig = InsightsGenerator(); print('AI integration working')"
```

## 🔄 Backup and Recovery

### Database Backups
- **Automatic Backups**: Created before each update
- **Retention Policy**: 7 days by default
- **Location**: `backups/` directory
- **Format**: SQLite database files

### Manual Backup
```bash
# Create manual backup
cp market.db backups/manual_backup_$(date +%Y%m%d).db

# Restore from backup
cp backups/backup_file.db market.db
```

### Data Recovery
```bash
# Rebuild database from processed files
python scripts/load_to_db.py

# Re-fetch recent data
python scripts/fetch_data.py
```

## 🌐 Deployment

### Local Development
```bash
# Development mode with debug
python main.py
```

### Production Deployment
```bash
# Production mode
export DASH_DEBUG=False
python main.py
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["python", "main.py"]
```

## 🔐 Security Best Practices

### Email Security
- Use app-specific passwords
- Enable 2-factor authentication
- Regularly rotate passwords
- Monitor email access logs

### Database Security
- Regular backups
- File permission restrictions
- Access logging
- Data encryption (if needed)

### Network Security
- Firewall configuration
- VPN for remote access
- HTTPS for production (if exposed)
- Regular security updates

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **Plotly/Dash** for the excellent visualization framework
- **Ollama** for local LLM capabilities
- **Python community** for the amazing ecosystem of libraries
- **Open source contributors** who make projects like this possible

## 📞 Support

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact maintainers for urgent issues

### Reporting Bugs
1. Check existing issues first
2. Provide detailed reproduction steps
3. Include system information and logs
4. Attach relevant configuration files (without sensitive data)

---

**Built with ❤️ for market analysis and data visualization**

*Last updated: 2025-08-27*
