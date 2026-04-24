# Zero-Shot-Sentiment-Emotion-Detection-using-TARS

## 📌 Overview

This project implements a **Zero-Shot Text Classification system** for detecting **emotions and sentiment** using the **TARS (Task-Aware Representation of Sentences)** model.

Unlike traditional approaches, this system does not require labeled training data. It dynamically classifies text into predefined categories using semantic understanding.

The application is built with **Streamlit** for an interactive user experience.

---

## 🚀 Features

* 🔹 Zero-shot classification (no training required)
* 🔹 Detects multiple emotions from text
* 🔹 Performs sentiment analysis (positive, negative, neutral)
* 🔹 Interactive web interface using Streamlit
* 🔹 Displays top emotions with confidence scores
* 🔹 Works on real-time user input

---

## 🧠 Concept

Zero-shot learning allows a model to classify text into categories it has never seen during training by evaluating the relationship between the input text and label descriptions.

TARS uses a **text + label pair approach**, where it determines whether a given label fits the input text.

---

## ⚙️ How It Works

1. User inputs text through the Streamlit interface
2. The model pairs the text with multiple labels (emotions/sentiments)
3. Each pair is processed using a Transformer-based architecture
4. The model predicts how well each label matches the text
5. Top emotions and sentiment are displayed with confidence scores

---

## 🏗️ Project Structure

```
├── app.py              # Streamlit application
├── SA-CLIP.ipynb      # Model experimentation and testing
├── PPT.pdf            # Project explanation and theory
└── README.md          # Project documentation
```

---

## 💻 Tech Stack

* **Python**
* **Streamlit** (UI)
* **Flair NLP**
* **PyTorch**
* **Transformer Models (TARS)**

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/zero-shot-emotion-detection.git
cd zero-shot-emotion-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Example

**Input:**

```
I am really excited about this opportunity!
```

**Output:**

```
Sentiment: Positive  
Top Emotions: Joy, Excitement, Optimism
```

---

## 📌 Use Cases

* Social media analysis
* Customer feedback analysis
* Chatbots and conversational AI
* Review and opinion mining
* Real-time text analytics

---

## 👨‍💻 Author

**Nikhil Sharma**
