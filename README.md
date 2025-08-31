# 📈 AI Market Pulse Dashboard

A comprehensive, modular AI-powered market analysis platform with automated data management, interactive visualizations, and intelligent insights. Built for professional market analysis with enterprise-grade automation and AI integration.

**Note**: While this version focuses on financial markets, the modular design allows easy adaptation to other data sources (e.g., Google Ads, IoT sensors, weather). This demonstrates extensibility beyond finance. The choice of financial data was made due to its availability and ease of access.

## 🚀 Core Features

### 🔧 **Modular Architecture**
- **Separation of Concerns**: Data fetching, storage, AI models, and visualization are completely decoupled
- **Domain Agnostic**: Originally built for stock market data, but designed to work with any time series data
- **Pluggable Components**: Swap out data sources (Yahoo Finance → Google Ads → IoT sensors) without breaking other modules
- **Extensible Design**: Add new forecasting models, AI providers, or dashboard types easily

### 📈 **Advanced Market Analysis**
- **Single Stock Analysis**: Detailed KPIs, price/volume charts, technical indicators, AI insights
- **Multi-Stock Comparison**: Normalized performance, correlation analysis, comparative insights
- **Price Forecasting**: Multiple ML models with confidence intervals and accuracy metrics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis

### 🤖 **AI & Machine Learning**
- **Ollama LLM Integration**: Local AI for market analysis and recommendations
- **Multiple Forecasting Models**: Linear Regression, ARIMA, Ensemble methods
- **Intelligent Insights**: AI-generated market commentary, trend analysis, and investment recommendations
- **Model Validation**: Cross-validation, accuracy scoring, confidence intervals

### 🔄 **Automation & Data Management**
- **Automated Data Pipeline**: Daily data fetching, processing, and database updates
- **SQLite Database**: Efficient storage with indexing and query optimization
- **Email Notifications**: Configurable alerts for success/failure with detailed reports
- **Scheduling Support**: Windows Task Scheduler and Linux/macOS cron integration

### 📊 **Interactive Web Interface**
- **Responsive Design**: Bootstrap-based UI that works on all devices
- **Real-time Updates**: Dynamic charts and data with loading indicators
- **Session Persistence**: Maintains state across tab switches and refreshes
- **Export Functions**: CSV downloads with customizable date ranges

## 🔧 Example Use Cases Beyond Finance

### **Digital Marketing**
- Compare Google Ads vs Facebook Ads performance
- Forecast ad spend ROI and campaign effectiveness
- AI analysis of marketing channel performance

### **Sales Analytics**
- Predict sales trends across regions or products
- Compare sales team performance
- Forecast revenue and identify growth opportunities

### **IoT Monitoring**
- Analyze sensor data streams for anomaly detection
- Forecast equipment failures and maintenance needs
- Monitor environmental conditions and trends

### **Social Media Analytics**
- Compare engagement metrics across platforms
- Forecast reach and viral potential
- AI commentary on content performance

## 🎯 Dashboard Components

### 1. **Single Stock Analysis Dashboard**
- **Key Performance Indicators**: Price, volume, volatility, returns
- **Interactive Charts**: Candlestick charts with volume overlay
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
- **AI Insights**: Automated analysis with trend identification and recommendations
- **Data Export**: Download historical data as CSV

### 2. **Multi-Stock Comparison Dashboard**
- **Performance Comparison**: Normalized price charts for multiple assets
- **Correlation Analysis**: Interactive heatmaps showing asset relationships
- **Statistical Analysis**: Return distributions, volatility comparisons
- **Comparative Insights**: AI-powered analysis of relative performance

### 3. **Price Forecasting Dashboard**
- **Multiple Models**: Choose from Linear Regression, ARIMA, or Ensemble methods
- **Configurable Periods**: 1, 3, 7 or 10-day forecasts
- **Confidence Intervals**: Statistical confidence bounds for predictions
- **Model Validation**: Accuracy metrics and performance scores
- **AI Commentary**: Interpretation of forecast results and investment implications

## 🔧 Technical Architecture

### **Data Layer**
- **Fetcher** (`src/market_dashboard/data/fetcher.py`): Yahoo Finance integration with error handling
- **Database** (`src/market_dashboard/data/database.py`): SQLite operations with connection pooling
- **Loader** (`src/market_dashboard/data/loader.py`): Data processing and storage pipeline

### **AI & Analytics Layer**
- **Insights Generator** (`src/market_dashboard/ai/insights.py`): AI-powered market analysis
- **Ollama Client** (`src/market_dashboard/ai/ollama_client.py`): Local LLM integration
- **Forecasting** (`src/market_dashboard/utils/forecasting.py`): ML models and predictions

### **Presentation Layer**
- **App** (`src/market_dashboard/app.py`): Main Dash application with configuration
- **Layouts** (`src/market_dashboard/components/layouts.py`): UI component definitions
- **Callbacks** (`src/market_dashboard/components/callbacks.py`): Interactive functionality
- **Charts** (`src/market_dashboard/components/charts.py`): Plotly visualization components

### **Automation Layer**
- **Auto Update** (`scripts/auto_daily_update.py`): Automated data pipeline
- **Email Notifications**: SMTP integration with configurable templates
- **Scheduling Scripts**: Windows batch files and Linux shell scripts

## 📁 Project Structure

```
market-dashboard/
├── src/
│   └── market_dashboard/
│       ├── app.py                 # Main Dash application
│       ├── data/                  # Data management modules
│       ├── components/            # UI components
│       ├── ai/                    # AI and analysis
│       └── utils/                 # Utility functions
├── scripts/                       # Automation scripts
├── config/                        # Configuration files
├── saved_data/                    # Data storage
├── tests/                        # Unit tests
├── main.py                       # Application entry point
├── README.md                     # Readme file
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Internet connection for data fetching
- Ollama (optional, for AI insights)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd market-dashboard
   python -m venv myenv
   
   # Windows
   myenv\Scripts\activate
   
   # Linux/macOS
   source myenv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. Set Up Ollama
 -Install [Ollama](https://ollama.com/download) (supports Linux, macOS, Windows).
 
 -Start the Ollama server:
   ```bash
   ollama serve
   ```
 -Pull and run the Mistral model (or any Ollama-supported model):
   ```bash
   ollama pull mistral
   ollama run mistral
   ```

3. **Initial Data Setup**
   ```bash
   # Fetch initial market data
   python scripts/fetch_data.py
   
   # Load data to database
   python scripts/load_to_db.py
   ```

4. **Run the Dashboard**
   ```bash
   python main.py
   ```

5. **Open Browser**
   ```
   http://localhost:8050
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
   - Recommended time: 6:00 PM (after market close)

   **Linux/macOS (Cron):**
   ```bash
   # Add to crontab for daily 6 PM execution
   0 18 * * * cd "/path/to/market-dashboard" && python scripts/auto_daily_update.py
   ```

## 🧪 Testing

### Run Tests
```bash
# Test imports and basic functionality
python -m pytest tests/

# Test email functionality
python scripts/test_email.py

# Test data pipeline
python scripts/fetch_data.py
python scripts/load_to_db.py
```

## 🔒 Security Considerations

- **Email Passwords**: Use app-specific passwords, not regular passwords
- **Configuration Files**: Email config added to `.gitignore` to prevent credential exposure
- **Database**: Local SQLite database for data security
- **API Access**: No API keys required for Yahoo Finance (free tier)
- **AI Processing**: Local Ollama ensures data privacy

## 🛠️ Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure port 8050 is available

**Data fetching fails:**
- Check internet connection
- Verify Yahoo Finance is accessible
- Check stock ticker symbols are valid

**Email notifications not working:**
- Use app passwords for Gmail (not regular password)
- Check SMTP settings in `config/email_config.json`
- Verify firewall allows SMTP connections

**AI insights not generating:**
- Ensure Ollama is running: `ollama serve`
- Check Ollama model is installed: `ollama pull mistral:latest`
- Verify network connectivity to localhost:11434

**Database errors:**
- Check write permissions in project directory
- Ensure sufficient disk space
- Try deleting `market.db` and re-running data scripts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request


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
Extend `src/market_dashboard/utils/calculations.py`:
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
- **CPU**: Any modern processor (2+ cores recommended)
- **Network**: Stable internet for data fetching

### Performance Metrics
- **Data Fetching**: ~2-5 seconds for 9 stocks (730 days)
- **Dashboard Loading**: ~1-3 seconds initial load
- **AI Insights**: ~3-10 seconds (depends on Ollama model)
- **Database Operations**: <1 second for typical queries
- **Chart Rendering**: ~0.5-2 seconds for complex visualizations

### Health Checks
```bash
# Check database status
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Records: {db.get_total_record_count()}')"

# Check latest data
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Latest: {db.get_latest_date()}')"

# Test AI integration
python -c "from market_dashboard.ai import InsightsGenerator; print('AI integration available')"
```

## 📞 Support

### Getting Help
- **Documentation**: This README and inline code comments
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact maintainers for urgent issues

### Reporting Bugs
1. Check existing issues first
2. Provide detailed reproduction steps
3. Include system information and logs
4. Attach relevant configuration files (without sensitive data)

### Feature Requests
1. Describe the use case and benefit
2. Provide examples or mockups if applicable
3. Consider contributing the feature yourself

## �🙏 Acknowledgments

- **Yahoo Finance**
- **Plotly/Dash** 
- **Ollama** 

---

