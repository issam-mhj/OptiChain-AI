import streamlit as st
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="OptiChain AI - Late Delivery Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "OptiChain AI - Supply Chain Delivery Prediction System"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .on-time {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .delayed {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_spark_session():
    """Initialize Spark session with optimized configuration"""
    try:
        spark = SparkSession.builder \
            .appName("OptiChain-StreamlitApp") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Error initializing Spark: {e}")
        return None

# Load the trained model (cached)
@st.cache_resource
def load_model(_spark):
    """Load the pre-trained PySpark ML Pipeline model"""
    import pickle
    import os
    
    # Try Spark format first (preferred)
    try:
        model_path = "./models/best_model"
        if os.path.exists(model_path) and os.path.exists(f"{model_path}/metadata"):
            pipeline_model = PipelineModel.load(model_path)
            st.success("✅ Model loaded successfully (Spark format)!")
            return pipeline_model
    except Exception as e:
        st.warning(f"⚠️ Could not load Spark format: {e}")
    
    # Fallback to pickle format
    try:
        model_path_pkl = "./models/best_model.pkl"
        if os.path.exists(model_path_pkl):
            with open(model_path_pkl, 'rb') as f:
                pipeline_model = pickle.load(f)
            st.success("✅ Model loaded successfully (Pickle format)!")
            return pipeline_model
    except Exception as e:
        st.error(f"❌ Error loading pickle model: {e}")
    
    # If both fail
    st.error("❌ Could not load model in any format")
    st.info("💡 Save the model by running the notebook:\n"
            "- In Docker Jupyter: Use Spark format (recommended)\n"
            "- Locally on Windows: Use Pickle format")
    return None

# Feature definitions (must match training data)
CATEGORICAL_FEATURES = ['shipping_mode', 'customer_segment', 'category_name', 'product_name']
NUMERICAL_FEATURES = ['scheduled_days_for_shipping', 'order_weekday', 'order_month', 
                      'is_weekend', 'is_holiday']

def make_prediction(spark, model, input_data):
    """Make prediction using the loaded model"""
    try:
        # Create DataFrame with exact column order the model expects
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
        
        schema = StructType([
            StructField("shipping_mode", StringType(), True),
            StructField("customer_segment", StringType(), True),
            StructField("category_name", StringType(), True),
            StructField("product_name", StringType(), True),
            StructField("scheduled_days_for_shipping", IntegerType(), True),
            StructField("order_item_quantity", IntegerType(), True),
            StructField("distance_km", DoubleType(), True),
            StructField("order_weekday", IntegerType(), True),
            StructField("order_month", IntegerType(), True),
            StructField("is_weekend", IntegerType(), True),
            StructField("is_holiday", IntegerType(), True),
            StructField("late_delivery_risk", IntegerType(), True)
        ])
        
        # Convert input dict to Spark DataFrame with schema
        df = spark.createDataFrame([input_data], schema=schema)
        
        # Make prediction
        predictions = model.transform(df)
        
        # Extract prediction and probability
        result = predictions.select('prediction', 'probability').collect()[0]
        prediction = int(result['prediction'])
        probability = float(result['probability'][1])  # Probability of class 1 (delayed)
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">📦 OptiChain AI - Late Delivery Predictor</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts the risk of late delivery for supply chain orders using 
    a machine learning model trained on historical data.
    """)
    
    # Initialize Spark and load model
    spark = get_spark_session()
    if spark is None:
        st.stop()
    
    model = load_model(spark)
    if model is None:
        st.stop()
    
    # Sidebar - Input Form
    with st.sidebar:
        st.header("📋 Order Details")
        st.markdown("Fill in the order information below:")
        
        # Categorical inputs
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "First Class", "Second Class", "Same Day"],
            help="Select the shipping method"
        )
        
        customer_segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"],
            help="Customer category"
        )
        
        category_name = st.selectbox(
            "Product Category",
            ["Office Supplies", "Technology", "Furniture", "Clothing", 
             "Sporting Goods", "Health and Beauty", "Toys and Games"],
            help="Product category"
        )
        
        product_name = st.text_input(
            "Product Name",
            "Sample Product",
            help="Enter product name"
        )
        
        st.markdown("---")
        
        # Numerical inputs
        scheduled_days = st.slider(
            "Scheduled Days for Shipping",
            min_value=0,
            max_value=10,
            value=3,
            help="Expected number of days for shipping"
        )
        
        order_item_quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=1,
            help="Number of items in the order"
        )
        
        distance_km = st.slider(
            "Distance (km)",
            min_value=0.0,
            max_value=5000.0,
            value=500.0,
            step=10.0,
            help="Distance from warehouse to delivery location"
        )
        
        order_weekday = st.selectbox(
            "Order Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                   "Friday", "Saturday", "Sunday"][x],
            help="Day of the week when order was placed"
        )
        
        order_month = st.selectbox(
            "Order Month",
            options=list(range(1, 13)),
            format_func=lambda x: ["January", "February", "March", "April", "May", "June",
                                   "July", "August", "September", "October", "November", 
                                   "December"][x-1],
            help="Month when order was placed"
        )
        
        is_weekend = st.checkbox("Ordered on Weekend", value=False)
        is_holiday = st.checkbox("Ordered during Holiday Season", value=False)
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Delivery Risk", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Order Summary")
        
        # Display input summary
        summary_data = {
            "Feature": ["Shipping Mode", "Customer Segment", "Category", "Product",
                       "Scheduled Days", "Order Quantity", "Distance (km)", 
                       "Order Weekday", "Order Month", "Weekend", "Holiday"],
            "Value": [
                shipping_mode, customer_segment, category_name, product_name,
                scheduled_days, order_item_quantity, f"{distance_km:.1f}",
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][order_weekday],
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][order_month-1],
                "Yes" if is_weekend else "No",
                "Yes" if is_holiday else "No"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("ℹ️ Model Info")
        st.info("""
        **Model Type**: Random Forest Classifier
        
        **Features Used**:
        - Shipping details
        - Customer info
        - Product category
        - Temporal features
        
        **Target**: Late Delivery Risk (0/1)
        """)
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("🔄 Analyzing order..."):
            # Prepare input data
            input_data = {
                'shipping_mode': shipping_mode,
                'customer_segment': customer_segment,
                'category_name': category_name,
                'product_name': product_name,
                'scheduled_days_for_shipping': scheduled_days,
                'order_item_quantity': order_item_quantity,
                'distance_km': float(distance_km),
                'order_weekday': order_weekday,
                'order_month': order_month,
                'is_weekend': 1 if is_weekend else 0,
                'is_holiday': 1 if is_holiday else 0,
                'late_delivery_risk': 0  # Placeholder, will be predicted
            }
            
            # Make prediction
            prediction, probability = make_prediction(spark, model, input_data)
            
            if prediction is not None:
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Display prediction
                if prediction == 0:
                    st.markdown(
                        f'<div class="prediction-box on-time">✅ ON-TIME DELIVERY EXPECTED<br>'
                        f'<small>Confidence: {(1-probability)*100:.1f}%</small></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box delayed">⚠️ LATE DELIVERY RISK DETECTED<br>'
                        f'<small>Risk Score: {probability*100:.1f}%</small></div>',
                        unsafe_allow_html=True
                    )
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        "On-Time" if prediction == 0 else "Delayed",
                        delta="Good" if prediction == 0 else "Risk",
                        delta_color="normal" if prediction == 0 else "inverse"
                    )
                
                with col2:
                    st.metric(
                        "Delay Probability",
                        f"{probability*100:.1f}%",
                        delta=f"{(probability-0.5)*100:.1f}%" if probability > 0.5 else f"{(0.5-probability)*100:.1f}%",
                        delta_color="inverse" if probability > 0.5 else "normal"
                    )
                
                with col3:
                    st.metric(
                        "Confidence Level",
                        "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
                    )
                
                # Probability gauge chart
                st.subheader("📈 Risk Analysis")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Late Delivery Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if prediction == 1:
                    st.warning("""
                    **Action Required:**
                    - ⚡ Consider upgrading to faster shipping method
                    - 📞 Notify customer of potential delay
                    - 📦 Check inventory and warehouse status
                    - 🚚 Coordinate with logistics partner
                    """)
                else:
                    st.success("""
                    **All Good:**
                    - ✅ Order is on track for on-time delivery
                    - 📧 Send confirmation email to customer
                    - 📊 Continue monitoring order progress
                    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>OptiChain AI</strong> - Supply Chain Late Delivery Prediction System</p>
        <p>Powered by PySpark ML & Streamlit | Built with ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
