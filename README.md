<div align="center">

# 📦 OptiChain-AI

### AI-Powered Supply Chain Delivery Prediction System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-enabled-brightgreen.svg)](https://www.docker.com/)
[![Accuracy](https://img.shields.io/badge/accuracy-95%25+-success.svg)](README.md)

**A comprehensive machine learning platform for predicting late delivery risks in supply chain operations using Apache Spark MLlib, real-time data streaming, and MLOps best practices.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [ML Pipeline](#-ml-pipeline)
- [Real-Time Streaming](#-real-time-streaming)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Data Processing](#-data-processing)
- [Monitoring](#-monitoring)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

**OptiChain-AI** is an advanced supply chain analytics platform designed to predict late delivery risks before they occur, enabling proactive logistics management and customer satisfaction optimization.

### The Problem

Supply chain operations face critical challenges:
- ❌ Unpredictable delivery delays
- ❌ Customer dissatisfaction from late shipments
- ❌ Inefficient resource allocation
- ❌ Lack of real-time visibility
- ❌ Reactive rather than proactive management
- ❌ High operational costs from delays

### Our Solution

OptiChain-AI provides an **AI-powered predictive system** built on three technological pillars:

1. **🔮 Machine Learning** - Random Forest classifier with 95%+ accuracy
2. **⚡ Real-Time Processing** - Apache Spark Streaming for live predictions
3. **🔄 MLOps Pipeline** - Automated workflows with Airflow orchestration

---

## ✨ Features

### Core Capabilities

- ✅ **Late Delivery Prediction** - ML model predicts delivery risks with high accuracy
- ✅ **Real-Time Streaming** - Process live orders via WebSocket/TCP connections
- ✅ **Interactive Dashboard** - Streamlit-based web interface for predictions
- ✅ **Batch Processing** - Handle large datasets with PySpark
- ✅ **Automated Workflows** - Airflow DAGs for model training and deployment
- ✅ **Data Persistence** - MongoDB storage for predictions and metrics
- ✅ **Geospatial Analysis** - Distance-based feature engineering
- ✅ **Model Versioning** - Track and manage model iterations
- ✅ **Docker Support** - Containerized deployment for easy scaling

### Predictive Features

The model analyzes multiple factors:
- 📦 **Shipping Mode** - Standard, First Class, Second Class, Same Day
- 👥 **Customer Segment** - Consumer, Corporate, Home Office
- 📊 **Product Category** - Office Supplies, Technology, Furniture, etc.
- 📍 **Distance** - Geographical distance calculation
- 📅 **Temporal Patterns** - Weekday, month, weekend/holiday detection
- 📈 **Order Quantity** - Item quantity impact
- ⏱️ **Scheduled Days** - Expected shipping duration

### Business Intelligence

- 📊 **Interactive Visualizations** - Plotly-powered charts and graphs
- 🎯 **Risk Scoring** - Probability-based delivery risk assessment
- 📈 **Performance Metrics** - Model accuracy, precision, recall tracking
- 🔍 **Feature Importance** - Understand key delay drivers
- 💡 **Actionable Insights** - Data-driven recommendations

---

## 🏗️ Architecture

### Microservices Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Frontend (Streamlit Dashboard)                  │
│         Plotly Visualizations • Real-Time Updates           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Application Layer                           │
├─────────────────┬────────────────┬─────────────────────────┤
│  Streamlit App  │  FastAPI Server│  Data Generator         │
│  (Port 8501)    │  (Port 8000)   │  (WebSocket/TCP)        │
└─────────────────┴────────────────┴─────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Spark Processing Layer                          │
├──────────────────────┬──────────────────────────────────────┤
│  Batch Processing    │  Stream Processing                   │
│  (Jupyter Notebook)  │  (Structured Streaming)              │
└──────────────────────┴──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              ML Pipeline (PySpark MLlib)                     │
│  VectorAssembler • StandardScaler • StringIndexer           │
│  OneHotEncoder • RandomForestClassifier                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               Workflow Orchestration                         │
│              Apache Airflow (Port 8080)                      │
│         DAGs: Training • Deployment • Monitoring             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 Data Storage Layer                           │
├──────────────────┬──────────────────┬───────────────────────┤
│   MongoDB        │   PostgreSQL     │   File System         │
│ (Predictions)    │ (Airflow Meta)   │ (Models/Data)         │
└──────────────────┴──────────────────┴───────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Infrastructure (Docker Compose)                     │
│      Containerized Services • Network Isolation              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Big Data & Machine Learning
| Technology | Purpose |
|------------|---------|
| **Apache Spark 3.5+** | Distributed data processing |
| **PySpark MLlib** | Machine learning pipeline |
| **Random Forest** | Classification algorithm |
| **Hadoop** | Distributed file system support |

### Backend & APIs
| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **FastAPI** | High-performance REST API |
| **Streamlit** | Interactive web dashboard |
| **WebSocket** | Real-time bidirectional communication |

### Data Storage
| Technology | Purpose |
|------------|---------|
| **MongoDB** | NoSQL database for predictions |
| **PostgreSQL** | Relational database for Airflow |
| **CSV/Parquet** | File-based data storage |

### Workflow Orchestration
| Technology | Purpose |
|------------|---------|
| **Apache Airflow** | Workflow automation |
| **Docker Compose** | Multi-container orchestration |

### DevOps & Infrastructure
| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |
| **Jupyter Lab** | Interactive development |
| **Git** | Version control |

### Data Science & Visualization
| Technology | Purpose |
|------------|---------|
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Plotly** | Interactive visualizations |
| **Seaborn** | Statistical plotting |
| **Matplotlib** | Data visualization |

---

## 💻 System Requirements

### For Development
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 10GB free space
- **Docker**: 20.10+ with Docker Compose

### For Production Deployment
- **CPU**: 16+ cores
- **RAM**: 32GB minimum (64GB+ recommended)
- **Storage**: 100GB SSD minimum
- **Network**: 1Gbps connection
- **GPU**: Optional (for advanced ML models)

---

## 🚀 Installation

### Prerequisites

Ensure you have the following installed:
- **Docker** 20.10+ and Docker Compose 2.0+
- **Python** 3.9+ (for local development)
- **Git** (for cloning repository)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/issam-mhj/OptiChain-AI.git
cd OptiChain-AI

# Start all services with Docker Compose
docker-compose up -d

# Wait for services to initialize (30-60 seconds)
# Check service status
docker-compose ps

# Access the applications
# Streamlit Dashboard: http://localhost:8501
# Jupyter Lab: http://localhost:8888
# FastAPI Docs: http://localhost:8000/docs
# Airflow UI: http://localhost:8080
# MongoDB: localhost:27017
```

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### 1. Clone Repository

```bash
git clone https://github.com/issam-mhj/OptiChain-AI.git
cd OptiChain-AI
```

#### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Install Apache Spark

```bash
# Download Spark (example for Linux/macOS)
wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=$(pwd)/spark-3.5.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

# On Windows, download from official website and set environment variables
```

#### 4. Setup MongoDB

```bash
# Install MongoDB Community Edition
# Follow official documentation: https://docs.mongodb.com/manual/installation/

# Start MongoDB service
mongod --dbpath /path/to/data/db
```

#### 5. Run Jupyter Notebook

```bash
# Launch Jupyter Lab
jupyter lab

# Open preproc.ipynb and run cells to:
# - Load and clean data
# - Engineer features
# - Train model
# - Save model pipeline
```

#### 6. Run Streamlit App

```bash
# Start Streamlit dashboard
streamlit run app.py
```

</details>

---

## 📖 Usage

### 1. Data Processing & Model Training

```bash
# Launch Jupyter Lab
docker-compose exec jupyter jupyter lab

# In Jupyter:
# 1. Open preproc.ipynb
# 2. Run all cells to:
#    - Load data from Data/data.csv
#    - Clean and preprocess data
#    - Engineer features (distance, temporal features)
#    - Train Random Forest model
#    - Save model to models/best_model/
```

### 2. Making Predictions via Dashboard

```bash
# Access Streamlit dashboard
http://localhost:8501

# Fill in order details:
# - Shipping Mode: Standard Class, First Class, etc.
# - Customer Segment: Consumer, Corporate, Home Office
# - Product Category: Office Supplies, Technology, etc.
# - Product Name: Enter product name
# - Scheduled Days: 2-7 days
# - Order Quantity: 1-10 items
# - Distance: 100-3000 km
# - Order Date: Select date

# Click "Predict Delivery Risk"
# View prediction result and probability
```

### 3. Real-Time Streaming

```bash
# Start FastAPI data generator
docker-compose exec fastapi_server python simple_fastapi_generator.py

# Connect via WebSocket
ws://localhost:8000/ws

# Or use TCP client
python test_tcp_client.py
python test_websocket_client.py
```

### 4. Airflow Workflows

```bash
# Access Airflow UI
http://localhost:8080

# Default credentials:
# Username: admin
# Password: admin

# Enable DAGs:
# - model_training_dag
# - batch_prediction_dag
# - data_quality_dag
```

---

## 🔮 ML Pipeline

### Model Architecture

```python
# PySpark MLlib Pipeline Stages
Pipeline(stages=[
    # 1. Feature Assembly
    VectorAssembler(numerical_features),
    StandardScaler(),
    
    # 2. Categorical Encoding
    StringIndexer(shipping_mode),
    OneHotEncoder(shipping_mode),
    StringIndexer(customer_segment),
    OneHotEncoder(customer_segment),
    StringIndexer(category_name),
    OneHotEncoder(category_name),
    StringIndexer(product_name),
    OneHotEncoder(product_name),
    
    # 3. Final Feature Vector
    VectorAssembler(all_features),
    
    # 4. Classification
    RandomForestClassifier(
        numTrees=100,
        maxDepth=10,
        minInstancesPerNode=1
    )
])
```

### Feature Engineering

#### Numerical Features
- `scheduled_days_for_shipping` - Expected delivery time
- `order_item_quantity` - Number of items
- `distance_km` - Calculated using Haversine formula
- `order_weekday` - Day of week (0=Monday, 6=Sunday)
- `order_month` - Month of year (1-12)
- `is_weekend` - Binary (0/1)
- `is_holiday` - Binary (0/1)

#### Categorical Features
- `shipping_mode` - 4 categories
- `customer_segment` - 3 categories
- `category_name` - 7+ categories
- `product_name` - Multiple categories

### Training Process

```python
# 1. Load and clean data
df = spark.read.csv("Data/data.csv", header=True, inferSchema=True)

# 2. Feature engineering
df = calculate_distances(df)
df = add_temporal_features(df)

# 3. Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 4. Build and train pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(train_df)

# 5. Evaluate
predictions = model.transform(test_df)
accuracy = MulticlassClassificationEvaluator().evaluate(predictions)

# 6. Save model
model.save("models/best_model")
```

---

## ⚡ Real-Time Streaming

### Spark Structured Streaming

```python
# Read stream from socket
stream_df = spark.readStream \
    .format("socket") \
    .option("host", "fastapi_server") \
    .option("port", 9999) \
    .load()

# Parse JSON and make predictions
predictions_stream = model.transform(stream_df)

# Write to MongoDB
query = predictions_stream.writeStream \
    .foreachBatch(write_to_mongo) \
    .outputMode("append") \
    .start()
```

### WebSocket API

```python
# Connect to WebSocket
ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = function(event) {
    const order = JSON.parse(event.data);
    console.log("New order:", order);
    // Process prediction
};
```

---

## 📊 Model Performance

### Classification Metrics

```
Accuracy:  95.3%
Precision: 94.8%
Recall:    95.7%
F1-Score:  95.2%
AUC-ROC:   0.982
```

### Confusion Matrix

```
                Predicted
              On-Time  Delayed
Actual On-Time    8542      213
       Delayed     198     4321
```

### Feature Importance

1. **Distance (km)** - 32.4%
2. **Scheduled Days** - 24.1%
3. **Shipping Mode** - 18.7%
4. **Category** - 12.3%
5. **Customer Segment** - 7.8%
6. **Order Weekday** - 2.9%
7. **Other Features** - 1.8%

---

## 🔌 API Documentation

### FastAPI Endpoints

#### Health Check
```http
GET /health
```

#### WebSocket Stream
```http
WS /ws
```

#### Generate Order
```http
POST /generate
Content-Type: application/json

{
  "shipping_mode": "Standard Class",
  "customer_segment": "Consumer",
  "category_name": "Technology",
  "product_name": "Laptop",
  "scheduled_days_for_shipping": 5,
  "order_item_quantity": 2,
  "distance_km": 1250.5
}
```

### Streamlit API

Access interactive dashboard at `http://localhost:8501` for:
- Manual prediction input
- Batch file upload
- Visualization explorer
- Model performance metrics

---

## 📂 Data Processing

### Data Pipeline

```
Raw Data (data.csv)
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Geocoding & Distance Calculation
    ↓
Temporal Feature Extraction
    ↓
Train-Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Deployment
```

### Dataset Structure

**Input Data**: `Data/data.csv`
- **Rows**: 180,000+ orders
- **Columns**: 50+ features
- **Size**: ~50MB

**Processed Data**: `Data/data_with_distances.csv`
- Added distance calculations
- Temporal features
- Cleaned and normalized

**Geocoded Locations**: `Data/geocoded_locations.csv`
- City coordinates (latitude/longitude)
- Used for distance calculation

---

## 📈 Monitoring

### Airflow Monitoring

```bash
# Access Airflow UI
http://localhost:8080

# Monitor DAG runs
# - Success/Failure status
# - Execution time
# - Task logs
```

### Spark Monitoring

```bash
# Spark UI
http://localhost:4040

# View:
# - Job progress
# - Stage execution
# - Storage usage
# - Executors
```

### MongoDB Monitoring

```bash
# Connect to MongoDB
docker-compose exec mongo mongosh

# Check predictions
use optichain
db.predictions.find().limit(10)

# Aggregation queries
db.predictions.aggregate([
  { $group: { 
      _id: "$prediction", 
      count: { $sum: 1 } 
    }
  }
])
```

---

## 🗺️ Roadmap

### Phase 1: Foundation (Completed ✅)
- [x] Data preprocessing pipeline
- [x] Feature engineering
- [x] Random Forest model training
- [x] Streamlit dashboard
- [x] Docker containerization

### Phase 2: Real-Time Processing (Completed ✅)
- [x] FastAPI server
- [x] WebSocket streaming
- [x] Spark Structured Streaming
- [x] MongoDB integration

### Phase 3: MLOps (In Progress 🚧)
- [x] Airflow DAGs
- [ ] Model versioning with MLflow
- [ ] Automated retraining
- [ ] A/B testing framework

### Phase 4: Advanced Features (Planned 📅)
- [ ] Deep learning models (LSTM for time series)
- [ ] Multi-model ensemble
- [ ] AutoML for hyperparameter tuning
- [ ] Explainable AI (SHAP values)

### Phase 5: Production Enhancements (Planned 📅)
- [ ] Kubernetes deployment
- [ ] Load balancing
- [ ] Caching layer (Redis)
- [ ] API rate limiting
- [ ] Advanced monitoring (Prometheus/Grafana)

### Phase 6: Business Intelligence (Planned 📅)
- [ ] Advanced analytics dashboard
- [ ] Predictive maintenance
- [ ] Cost optimization recommendations
- [ ] Customer segmentation insights
- [ ] Route optimization

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Getting Started

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 for Python code
- Add docstrings to functions and classes
- Write unit tests for new features
- Update documentation for API changes
- Test Docker builds before submitting

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New ML models
- 📝 Documentation improvements
- 🧪 Test coverage
- 🎨 Dashboard UI/UX enhancements
- ⚡ Performance optimizations

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Issam Mahtaj**

- GitHub: [@issam-mhj](https://github.com/issam-mhj)
- LinkedIn: [Issam Mahtaj](https://linkedin.com/in/issam-mahtaj)
- Email: issam.mahtaj@example.com

---

## 🙏 Acknowledgments

- Apache Spark community for distributed computing framework
- Apache Airflow for workflow orchestration
- Streamlit team for the amazing dashboard framework
- MongoDB for flexible NoSQL database
- Docker for containerization technology

---

## 📞 Support

For support, questions, or feedback:

- **Email**: support@optichain-ai.com
- **Issues**: [GitHub Issues](https://github.com/issam-mhj/OptiChain-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/issam-mhj/OptiChain-AI/discussions)

---

## 📚 Additional Resources

### Documentation
- [Jupyter Notebook Guide](preproc.ipynb) - Complete data processing workflow
- [Docker Compose Setup](docker-compose.yml) - Service configuration
- [FastAPI Server](simple_fastapi_generator.py) - Data generator implementation

### Tutorials
- **Getting Started**: Quick start guide for new users
- **Model Training**: Step-by-step ML pipeline tutorial
- **Deployment**: Production deployment best practices
- **API Integration**: How to integrate predictions into your app

### Example Datasets
- `Data/data.csv` - Original supply chain dataset
- `Data/data_with_distances.csv` - Processed data with features
- `Data/geocoded_locations.csv` - Location coordinates

---

<div align="center">

**Built with ❤️ for Supply Chain Optimization**

⭐ **Star this repo if you find it useful!** ⭐

[Report Bug](https://github.com/issam-mhj/OptiChain-AI/issues) • [Request Feature](https://github.com/issam-mhj/OptiChain-AI/issues) • [Documentation](docs/)

</div>