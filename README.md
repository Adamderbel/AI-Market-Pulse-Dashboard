# 📈 AI Market Pulse Dashboard

A comprehensive, modular, AI-powered market analysis platform featuring automated data management, interactive visualizations, and intelligent insights. It also includes an integrated chatbot assistant powered by my [AI-Powered RAG Chatbot](https://github.com/Adamderbel/AI-Powered-RAG-Chatbot) project. Designed for professional market analysis, it delivers enterprise-grade automation and advanced AI integration.

**Note**: While this version focuses on financial markets for the availability of data and ease of access, the modular design allows easy adaptation to other data sources (e.g., Google Ads, IoT sensors, weather).
> 🔗 *Demo:* [AI Market Pulse Dashboard](https://adamderbel.github.io/Adam-Portfolio/ai-market-pulse.html)

## 🚀 Core Features

### 🔧 **Modular Architecture**
- **Separation of Concerns**: Data fetching, storage, AI models, and visualization are completely decoupled
- **Domain Agnostic**: Originally built for stock market data, but designed to work with any time series data
- **Pluggable Components**: Swap out data sources (Yahoo Finance → Google Ads → IoT sensors) without breaking other modules
- **Extensible Design**: Add new forecasting models, AI providers, or dashboard types easily

### 📈 **Advanced Market Analysis**
- **Single Stock Analysis**: Detailed KPIs, price/volume charts, technical indicators, AI insights
- **Multi-Stock Comparison**: Normalized performance, correlation analysis, comparative insights
- **Price Forecasting**: Multiple ML models with confidence intervals and accuracy metrics, AI insights
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis

### 🤖 **AI & Machine Learning**
- **Ollama LLM Integration**: Local AI for market analysis and recommendations
- **Chatbot Helper:** An integrated AI assistant that guides users throughout the platform, answering questions and providing contextual support in real time.
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


## 📁 Project Structure

```
market-dashboard/
├── src/
│   └── market_dashboard/
│       ├── app.py                 # Main Dash application
│       ├── data/                  # Data management modules
│       ├── components/            # UI components
│       ├── ai/                    # AI and analysis
│       ├── assests/               # css styles 
│       └── utils/                 # Utility functions
├── scripts/                       # Automation scripts
├── config/                        # Configuration files
├── saved_data/                    # Data storage
├── tests/                        # Unit tests
├── main.py                       # Application entry point
├── README.md                     # Readme file
└── requirements.txt              # Python dependencies
```

## 🖥️ Setup Instructions

1. **Clone the Repository and Set Up Virtual Environment**
   ```bash
   git clone https://github.com/Adamderbel/AI-Market-Pulse-Dashboard.git
   cd market-dashboard
   python -m venv myenv
   ```
   Activate the virtual environment:
   - **Windows**: `myenv\Scripts\activate`
   - **Linux/macOS**: `source myenv/bin/activate`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up OpenRouter (for AI Features)**

* Go to [OpenRouter.ai](https://openrouter.ai/) and sign up or log in.
* Navigate to your **API Keys** section: [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
* Generate a new **API key**.
* Create a `.env` file in the project root (if it doesn’t exist) and add your key:

  ```bash
  OPENROUTER_API_KEY=your_api_key_here
  ```
* The application will automatically use this key to access OpenRouter’s AI models for generating insights and chatbot responses.

4. **Fetch and Load Initial Data**
   ```bash
   python scripts/fetch_data.py  # Fetch market data from Yahoo Finance
   python scripts/load_to_db.py  # Load data into SQLite database
   ```

5. **Run the Dashboard**
   ```bash
   python main.py
   ```

6. **Access the Dashboard**
   Open your browser and navigate to [http://localhost:8050](http://localhost:8050).


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


### Health Checks
```bash
# Check database status
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Records: {db.get_total_record_count()}')"

# Check latest data
python -c "from market_dashboard.data import DatabaseManager; db = DatabaseManager('market.db'); print(f'Latest: {db.get_latest_date()}')"

# Test AI integration
python -c "from market_dashboard.ai import InsightsGenerator; print('AI integration available')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request


---

