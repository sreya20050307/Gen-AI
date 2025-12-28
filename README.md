# Gen AI Email Classifier

A Flask-based web application for classifying emails as spam or ham using machine learning. The application uses a DeBERTa model for email classification and provides a user-friendly web interface for single email classification, batch processing, and model metrics visualization.

## 🚀 Features

- **Email Classification**: Classify individual emails as spam or ham
- **Batch Processing**: Process multiple emails at once
- **Model Metrics**: View detailed performance metrics and confusion matrix
- **User Authentication**: Secure login and registration system
- **Responsive UI**: Modern, mobile-friendly interface
- **Real-time Results**: Instant classification with confidence scores

## 📋 Prerequisites

Before running this application, make sure you have the following installed:

- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Virtual environment** (recommended)

## 🛠️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd gen-ai-email-classifier
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (if needed)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 5: Download Model Files
The DeBERTa model files are not included in the repository due to GitHub's 25MB file size limit. You need to download them separately:

```bash
# Create models directory if it doesn't exist
mkdir -p models/deberta_email_classifier

# Download the model files (you'll need to provide the actual download URLs)
# Example commands (replace with actual URLs):
# wget/curl commands to download model.safetensors and other model files
```

**Note**: The application includes a fallback keyword-based classification system that works without the DeBERTa model. If you don't download the model files, the app will still function using the keyword-based approach.

## 🏃‍♂️ Running the Application

### Method 1: Direct Run (Recommended for Development)
```bash
python app.py
```

### Method 2: Using Flask CLI
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

### Access the Application
Once running, open your browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### First Time Setup
1. **Register**: Create a new account or login if you already have one
2. **Welcome Page**: You'll be redirected to the welcome page after login

### Classifying Emails
1. **Navigate to Classify**: Click on "Classify Email" from the navigation
2. **Enter Email Text**: Paste the email content (subject + body) into the text area
3. **Click Classify**: Press the "Classify Email" button
4. **View Results**: See the prediction (Spam/Ham), confidence score, and probabilities

### Batch Classification
1. **Go to Batch**: Click on "Batch Classification" from the navigation
2. **Upload CSV**: Upload a CSV file with email data
3. **Process**: The system will classify all emails in the file

### View Metrics
1. **Access Metrics**: Click on "Metrics" from the navigation
2. **Analyze Performance**: View model accuracy, precision, recall, and confusion matrix
3. **Latest Prediction**: See the most recent email classification as an example

### Logout
- Click the user dropdown in the top-right corner
- Select "Logout" to end your session

## 🏗️ Project Structure

```
gen-ai-email-classifier/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/                     # Pre-trained models
│   └── deberta_email_classifier/  # DeBERTa model files
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Dashboard
│   ├── classify.html          # Single email classification
│   ├── result.html            # Classification results
│   ├── batch.html             # Batch processing
│   ├── metrics.html           # Model metrics
│   ├── login.html             # Login page
│   ├── register.html          # Registration page
│   └── welcome.html           # Welcome/landing page
├── static/                     # Static files (CSS, JS, images)
│   ├── css/
│   │   └── styles.css         # Custom styles
│   └── js/
│       └── main.js            # JavaScript functionality
├── uploads/                    # Uploaded files directory
└── __pycache__/               # Python cache files (auto-generated)
```

## ⚙️ Configuration

The application uses the following key configurations (defined in `config.py`):

- **Model**: Microsoft DeBERTa v3 Base
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 8
- **Learning Rate**: 2e-5

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors with Transformers/PyTorch**
- The app includes fallback keyword-based classification
- For full ML functionality, ensure compatible PyTorch version

**2. Model Loading Issues**
- Ensure model files exist in `models/deberta_email_classifier/`
- Check file permissions if loading fails

**3. Port Already in Use**
```bash
# Kill process using port 5000
# On Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On Linux/macOS:
lsof -ti:5000 | xargs kill -9
```

**4. Virtual Environment Issues**
```bash
# Deactivate and reactivate
deactivate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

## 📊 Model Performance

The DeBERTa model provides:
- **Accuracy**: ~94%
- **Precision**: ~92%
- **Recall**: ~89%
- **F1-Score**: ~90.5%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `app.py`
3. Create an issue in the repository

---

**Happy Email Classification! 🎉**