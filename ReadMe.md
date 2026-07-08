# 🎬 Large Scale Movie Recommendation System 🍿

## 📌 Table of Contents
- [🎯 Overview](#-overview)
- [📥 Data Ingestion and Preprocessing](#-data-ingestion-and-preprocessing)
- [🗄️ MongoDB Integration & Director Data](#-mongodb-integration--director-data)
- [🧠 Recommendation Algorithms](#-recommendation-algorithms)
- [🛠️ Flask API Endpoints](#-flask-api-endpoints)
- [💻 Frontend Interface](#-frontend-interface)
- [⚙️ Installation & Setup](#-installation--setup)
- [🚀 Running the Application](#-running-the-application)
- [🔄 Batch Retraining](#-batch-retraining)
- [📊 Hybrid Recommendation Notebook](#-hybrid-recommendation-notebook)
- [📺 Demo Videos](#-demo-videos)
- [🚀 Future Enhancements](#-future-enhancements)
- [📜 License](#-license)

---

## 🎯 Overview
The **Large-Scale Movie Recommendation System** is a Flask-based web application that provides intelligent movie recommendations using multiple machine learning models. It leverages the **IMDB and MovieLens datasets**, including movies, ratings, tags, links, and director information.

### Key Features:
✅ **Multi-Model Recommendations** – SVD, Collaborative Filtering, Hybrid, and Director-Based filtering.
✅ **IMDb Data Integration** – Find movies based on your favorite directors! 🎬
✅ **Real-time Feedback Mechanism** – Users can like or dislike recommendations to improve future suggestions. 🧠
✅ **MongoDB Storage** – Efficient caching to reduce data load times and enhance performance. 💾
✅ **Kubernetes-Based Scaling** – Designed for seamless handling of large datasets.
✅ **MLflow Integration** – For batch retraining and model switching.

---

## 📥 Data Ingestion and Preprocessing
- Fetches **MovieLens dataset** using the **Kaggle API** 📂
- Includes **movies.csv, ratings.csv, tags.csv, and links.csv**
- **Genre & Tag Merging**: Groups movie tags for enhanced content-based filtering.
- **Feature Engineering**: Uses **genre encoding and tag-based similarity metrics** to improve recommendations.
- **Extracts Release Year**: Extracts and normalizes release years from movie titles.
- **Prepares Data for Collaborative Filtering**: Formats ratings into a matrix for **SVD-based filtering**.

---

## 🗄️ MongoDB Integration & Director Data
- Stores **movies, ratings, and tags** in a **MongoDB instance** 🏪
- Uses **caching** to reduce repetitive queries & improve performance 🚀
- Downloads IMDb **crew & names datasets** to link movies with **directors** 👨‍🎬
- Runs as a **background process** to avoid UI lag ⏳

---

## 🧠 Recommendation Algorithms
### 🏆 **SVD-Based Recommendation (Matrix Factorization)**
- **Uses Singular Value Decomposition (SVD)** to predict user ratings 🎯
- Identifies latent relationships between users and movies for **highly personalized suggestions** 🔥

### 🔄 **Collaborative Filtering**
- Computes **Cosine Similarity** between user preferences 🤝
- Suggests movies based on similar viewing patterns 🏅

### 🎭 **Hybrid Recommendation Model**
- **Combines SVD & Collaborative Filtering** for improved accuracy 🎉
- Normalizes and merges scores for **balanced diversity & personalization** 🏆

### 🎬 **Director-Based Recommendation**
- Uses IMDb data for **content-based filtering** 🏅
- Fetches movies based on **user-inputted director names** 🔍

---

## 🛠️ Flask API Endpoints
| ⚡ Endpoint | 📝 Description |
|------------|--------------|
| `/` | Serves the **main UI** 🎨 |
| `/load_data` | Downloads & processes dataset, stores in MongoDB 🛠️ |
| `/show_movies` | Provides a **paginated movie list** 📜 |
| `/select_movies` | Saves **user-preferred movies** for recommendations 🎯 |
| `/recommend` | Generates **recommendations** based on chosen algorithm 🤖 |
| `/feedback` | Accepts **user feedback** to improve recommendations 👍👎 |
| `/refresh_recommendations` | Updates recommendations based on user changes 🔄 |
| `/director_progress` | Displays IMDb **director data loading progress** 🎬 |

---

## 💻 Frontend Interface
🎬 **Data Loading** – Users trigger `/load_data` for ingestion.
🎞️ **Movie Browsing** – Browse through movies with pagination.
🧠 **Recommendation Customization** – Choose between **SVD, Collaborative, Hybrid, or Director-based filtering**.
⭐ **Real-time Feedback** – Like/Dislike movies to enhance future suggestions.

---

## ⚙️ Installation & Setup
### 🔧 Prerequisites
- **MongoDB**
- **Python 3.8+**
- **Conda (Optional, but recommended)**
- **Docker & Kubernetes (for scalable deployment)**
- **MLflow for model tracking**
- **Flask, Pandas, NumPy, Scikit-Learn, Scipy**
- **Kaggle API**

### 🛠️ Installation Steps
1️⃣ Clone this repository:
   ```bash
   git clone https://github.com/your-repo/movie-recommendation.git
   cd movie-recommendation
   ```
2️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Start MongoDB:
   ```bash
   sudo systemctl start mongod
   ```
4️⃣ Configure Kaggle API for dataset download 🔑
5️⃣ Run the Flask app:
   ```bash
   python app.py
   ```
6️⃣ Access the web UI at **http://127.0.0.1:5000/** 🌍

---

## 🚀 Running the Application
### Standalone Mode
Run the app as a standalone service:
```bash
python app.py
```

### Scalable Mode with Kubernetes
Refer to the [K8 folder](./k8/README.md) for deployment instructions.

---

## 🔄 Batch Retraining
The system supports **batch retraining** using **MLflow**. The latest models are tracked and switched automatically to improve recommendation accuracy. Check the [Batch Retraining Folder](./batch_retraining/README.md).

---

## 📊 Hybrid Recommendation Notebook
The **hybrid-recommendation.ipynb** file evaluates the recommendation system’s accuracy using **MP@K (Mean Precision at K)** metrics.

---

## 📺 Demo Videos
If you were unable to run the application, view the demonstration videos in the **video folder**.

---

## 🚀 Future Enhancements
🔮 **Deep Learning-powered Recommendations** 🤖
🔒 **User Authentication & Profile-Based Filtering** 🔑
🎭 **Multi-Source Data (Rotten Tomatoes, IMDb Reviews, etc.)** 📊
📺 **Streaming Service Integration (Netflix, Hulu, etc.)** 🎬

---

## 📜 License
This project is licensed under the **MIT License** 📄. See `LICENSE` for details.

---

🚀 **Enhancing your movie-watching experience one recommendation at a time!** 🍿🎬

