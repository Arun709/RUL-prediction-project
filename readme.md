# ✈️ Turbofan Engine Predictive Maintenance - Enhanced UI

A stunning, AI-powered predictive maintenance system for aircraft turbofan engines with a premium user interface featuring advanced visualizations, animations, and modern design.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)
![License](https://img.shields.io/badge/license-MIT-yellow)


## 🌐 Live Streamlit App
**[RUL Prediction Web App](https://rulpredictionproject.streamlit.app/)**

## 🌟 Features

### Premium UI/UX Design
- **Modern Gradient Theme** - Beautiful purple gradient color scheme
- **Glassmorphism Effects** - Semi-transparent cards with backdrop blur
- **Smooth Animations** - Fade-in effects and hover transitions
- **Responsive Layout** - Works perfectly on all screen sizes
- **Custom CSS Styling** - Professional, polished appearance

### Advanced Visualizations
- **Interactive Gauges** - Real-time RUL prediction displays
- **3D Plots** - Multi-dimensional data analysis
- **Heatmaps** - Sensor correlation matrices
- **Time Series Charts** - Historical trend analysis
- **Donut & Pie Charts** - Fleet status distribution
- **Polar Charts** - Performance metrics visualization
- **Multi-axis Plots** - Combined metric analysis

### Core Functionality
- **Fleet Dashboard** - Complete overview of all engines
- **Single Prediction** - Individual engine RUL analysis
- **Batch Analysis** - Upload and analyze multiple engines
- **Advanced Analytics** - Deep insights and cost analysis
- **Real-time Metrics** - Live status updates
- **Export Capabilities** - Download reports in CSV format

### Smart Features
- **Progress Animations** - Visual feedback during processing
- **Status Badges** - Color-coded health indicators
- **Confidence Scores** - AI prediction reliability metrics
- **Maintenance Recommendations** - Actionable insights
- **Cost Savings Tracking** - ROI visualization
- **Sensor Grid Layout** - Organized input interface

## 📋 Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

## 🚀 Installation

### 1. Clone or Download the Repository

```bash
# If you have the file, navigate to its directory
cd /path/to/turbofan-app
```

### 2. Install Required Packages

```bash
pip install streamlit pandas numpy plotly requests
```

Or create a requirements.txt file:

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
requests>=2.31.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 💻 Running the Application

### Basic Launch

```bash
streamlit run turbofan_app_enhanced.py
```

### With Custom Configuration

```bash
# Run on specific port
streamlit run turbofan_app_enhanced.py --server.port 8501

# Run on different address
streamlit run turbofan_app_enhanced.py --server.address 0.0.0.0

# Run with custom theme
streamlit run turbofan_app_enhanced.py --theme.base dark
```

### Full Production Settings

```bash
streamlit run turbofan_app_enhanced.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.maxUploadSize 10 \
  --browser.gatherUsageStats false
```

## 📱 Usage Guide

### 1. Fleet Dashboard Tab

**Monitor your entire engine fleet:**
- View total engines, critical status, and average RUL
- Analyze RUL distribution across the fleet
- Check fleet health status with interactive donut chart
- Review RUL vs confidence scatter plot
- Track trends over time
- Export detailed fleet reports

### 2. Single Prediction Tab

**Predict RUL for individual engines:**

1. **Enter Engine Configuration:**
   - Engine Unit ID
   - Current cycle number
   - Operating settings (Altitude, Mach, Throttle)

2. **Input Sensor Readings:**
   - Temperature sensors (Tab 1)
   - Pressure sensors (Tab 2)
   - Other sensors (Tab 3)

3. **Click "PREDICT RUL":**
   - View animated prediction process
   - See predicted RUL with gauge visualization
   - Check maintenance requirements
   - Review confidence scores
   - Get actionable recommendations

### 3. Batch Analysis Tab

**Process multiple engines at once:**

1. **Upload CSV File:**
   - Required columns: unit_id, time_cycle, settings, sensors
   - Maximum file size: 10MB

2. **Preview Data:**
   - View data samples
   - Check statistics
   - Validate data quality

3. **Analyze Batch:**
   - Process all engines
   - View distribution charts
   - Review detailed results table
   - Download predictions report

### 4. Analytics Tab

**Deep dive into performance metrics:**
- Multi-metric trend analysis
- Cost savings breakdown
- Prediction accuracy tracking
- Sensor correlation heatmaps
- Export analytics reports

## 🎨 Customization

### Changing Colors

Edit the CSS in the `st.markdown()` section:

```python
# Main gradient (lines ~28-29)
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Metric cards (line ~71)
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Status colors (lines ~103-117)
.status-healthy { background: #48bb78; }
.status-warning { background: #ed8936; }
.status-critical { background: #f56565; }
```

### Modifying Layouts

Adjust column ratios:

```python
# Equal columns
col1, col2 = st.columns(2)

# Custom ratio
col1, col2, col3 = st.columns([2, 1, 1])
```

### Adding New Visualizations

Use Plotly for interactive charts:

```python
import plotly.graph_objects as go

fig = go.Figure(data=[...])
fig.update_layout(...)
st.plotly_chart(fig, use_container_width=True)
```

## 🔌 API Integration

### Setting Up the Backend

The app expects a REST API endpoint for predictions. Example:

```python
# Your API endpoint should accept:
POST /predict
Content-Type: application/json

{
  "unit_id": 1,
  "time_cycle": 100,
  "setting_1": 0.5,
  "setting_2": 0.5,
  "setting_3": 0.5,
  "sensor_1": 642.5,
  ...
  "sensor_21": 555.0
}

# And return:
{
  "predicted_rul": 125.5,
  "maintenance_required": false,
  "confidence_level": "high",
  "confidence_score": 0.95
}
```

### Connecting to Your API

Update the API URL in the sidebar:

```python
api_url = st.sidebar.text_input("API URL", value="http://your-api-url:8000")
```

Or uncomment the API call section (around line 711):

```python
response = requests.post(f"{api_url}/predict", json=data)
result = response.json()
```

## 📊 Data Format

### CSV Upload Format

For batch analysis, your CSV should include:

```csv
unit_id,time_cycle,setting_1,setting_2,setting_3,sensor_1,sensor_2,...,sensor_21
1,100,0.5,0.5,0.5,642.5,1589.7,...,555.0
2,150,0.6,0.4,0.7,643.0,1590.2,...,556.5
```

### Required Columns

- `unit_id`: Engine identifier (integer)
- `time_cycle`: Current operational cycle (integer)
- `setting_1`, `setting_2`, `setting_3`: Operating conditions (float, 0-1)
- `sensor_1` through `sensor_21`: Sensor readings (float)

## 🎯 Best Practices

### Performance Optimization

1. **Use caching for data:**
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

2. **Limit data size:**
   - Keep datasets under 10,000 rows for smooth performance
   - Use pagination for larger datasets

3. **Optimize visualizations:**
   - Limit data points in scatter plots
   - Use sampling for large datasets

### User Experience

1. **Provide feedback:**
   - Use progress bars for long operations
   - Display status messages
   - Show success/error notifications

2. **Clear instructions:**
   - Add help text to inputs
   - Include tooltips
   - Provide examples

3. **Responsive design:**
   - Test on different screen sizes
   - Use `use_container_width=True`
   - Implement proper column layouts

## 🐛 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Use a different port
streamlit run turbofan_app_enhanced.py --server.port 8502
```

**2. Module Not Found**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**3. Slow Performance**
- Clear browser cache
- Reduce data size
- Enable caching with `@st.cache_data`
- Limit visualization complexity

**4. API Connection Failed**
- Verify API is running
- Check firewall settings
- Confirm correct API URL
- Test API with curl/Postman first

### Debug Mode

Enable debug information:

```bash
streamlit run turbofan_app_enhanced.py --logger.level=debug
```

## 🔒 Security Considerations

- **API Keys:** Store in environment variables, not in code
- **File Uploads:** Validate file types and sizes
- **Input Validation:** Sanitize all user inputs
- **HTTPS:** Use HTTPS in production
- **Authentication:** Implement user authentication for production

## 📈 Future Enhancements

- [ ] User authentication system
- [ ] Real-time data streaming
- [ ] PDF report generation
- [ ] Excel export with formatting
- [ ] Email notifications
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Dark/Light theme toggle
- [ ] Advanced filtering options
- [ ] Historical data comparison

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - feel free to use this for your projects!

## 👨‍💻 Support

For issues and questions:
- Check the troubleshooting section
- Review Streamlit documentation: https://docs.streamlit.io
- Plotly documentation: https://plotly.com/python/

## 🙏 Acknowledgments

- **Streamlit** - Amazing framework for ML/Data apps
- **Plotly** - Interactive visualization library
- **NASA** - C-MAPSS dataset
- **Design Inspiration** - Modern UI/UX best practices

---

**Built with ❤️ for the Aviation Industry**

*Preventing failures, one prediction at a time.*

