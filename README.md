🧠 **Customer Segmentation using Machine Learning**

📌 **Project Overview**

This project focuses on customer segmentation using unsupervised machine learning techniques to identify distinct groups of customers based on their purchasing behavior, demographics, and engagement patterns.

The goal is to help businesses:

- Understand customer behavior
- Target marketing strategies effectively
- Improve customer retention and revenue

---

🎯 **Problem Statement**

Businesses often have large customer datasets but lack insights into customer groups and behavior patterns.

This project solves that problem by:

- Grouping customers into meaningful segments
- Identifying high-value and low-value customers
- Enabling data-driven decision-making

---

🛠️ **Technologies Used**

- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit (for deployment)

---

📊 **Dataset Description**

The dataset contains customer information such as:

- Income
- Recency
- Purchases (Web, Store, Catalog)
- Campaign responses
- Demographics (Age, Family Size, etc.)

---

⚙️ **Project Workflow**

1. Data Preprocessing

- Handling missing values
- Feature engineering:
  - Total Spend
  - Total Purchases
  - Family Size
  - Customer Tenure
- One-Hot Encoding for categorical variables

---

2. Feature Scaling

- Applied StandardScaler to normalize numerical features

---

3. Dimensionality Reduction

- Used Principal Component Analysis (PCA)
- Reduced dimensionality while preserving ~90% variance

---

4. Clustering Models

Implemented and compared multiple clustering techniques:

- K-Means Clustering
- Hierarchical Clustering (Ward Method)
- DBSCAN

---

📈 **Model Evaluation**

Evaluation metric used:

- Silhouette Score

Model Comparison:

Model| Clusters| Silhouette Score
K-Means| 3| ~0.17
Hierarchical (Ward)| 3| ~0.21
DBSCAN| 3| ~0.35

---

✅ **Final Model Selection**

Although DBSCAN showed a higher silhouette score, K-Means was selected for deployment because:

- It produces well-defined and interpretable clusters
- It is more stable and scalable
- It aligns better with business segmentation needs

---

🧩 **Customer Segments Identified**

Cluster| Segment Name| Description
0| Premium Customers| High spenders, frequent buyers
1| Low Value Customers| Low engagement and spending
2| Mid Value Customers| Moderate spending and activity

---

🚀 **Deployment (Streamlit App)**

Features of the app:

- Single customer prediction
- Bulk prediction via file upload (CSV/Excel)
- Automatic preprocessing
- Download prediction results

---

📸 Application Preview

(Add your screenshots here)

---

📂 **Project Structure**

customer-segmentation/
│
├── EDA.ipynb
├── Model_Building.ipynb
├── app.py
├── model_pipeline.pkl
├── dataset/
├── README.md

---

⚠️ **Challenges Faced**

- Ensuring consistent preprocessing between training and deployment
- Handling categorical encoding dynamically
- Maintaining feature alignment during inference

---

💡 **Key Learnings**

- Importance of data preprocessing in clustering
- Impact of scaling on distance-based algorithms
- Real-world challenges in model deployment
- Building end-to-end ML applications

---

📬 **Future Improvements**

- Improve clustering using advanced techniques
- Add visualization dashboard
- Deploy on cloud (Streamlit Cloud / AWS)

---

**👩‍💻 Author**

**Er. Pratiksha Mhaske**

**LinkedIn:** https://www.linkedin.com/in/pratiksha-mhaske-173643387

**GitHub:** https://github.com/PratikshaMhaske
