# Network Traffic Classification

This project performs network traffic classification using **Logistic Regression** and **Support Vector Machine (SVM)** models from the **scikit-learn** library. It involves data preprocessing, visualization, and hyperparameter tuning to achieve high classification accuracy.

## 📊 Dataset Overview

- **Total Records:** 104,345 rows, 23 columns
- **Key Columns:**
  - `dt`, `switch`, `pktcount`, `bytecount`, `dur`, `dur_nsec`, `tot_dur`
  - `flows`, `packetins`, `pktperflow`, `byteperflow`, `pktrate`
  - `Pairflow`, `port_no`, `tx_bytes`, `rx_bytes`, `tx_kbps`, `rx_kbps`, `tot_kbps`
  - Encoded Protocols: `Protocol_ICMP`, `Protocol_TCP`, `Protocol_UDP`
  - **Label:** Target variable for classification

## 🛠️ Project Workflow

### 1. Data Preprocessing

- **Handling Missing Values:**
  - Numerical columns filled using median values.
  - Columns with missing ratios between 10%-15% also filled with their median values.

- **Encoding Categorical Variables:**
  - `Protocol` column was encoded using **one-hot encoding**.

- **Duplicate and Outlier Removal:**
  - Removed duplicate rows.
  - Detected and removed outliers using the **Z-score** method.

### 2. Data Normalization

- **StandardScaler** from `scikit-learn` was used to normalize numerical columns.

### 3. Data Visualization

- **Univariate Analysis:**
  - Boxplots and histograms for distribution visualization.
- **Bivariate Analysis:**
  - Scatter plots to observe the relationship between numerical columns and the target variable.
- **Correlation Analysis:**
  - Heatmap for correlation between numerical features.

### 4. Model Training

We used the following models from **scikit-learn**:

- **Logistic Regression**
- **Support Vector Machine (SVM)**

### 5. Hyperparameter Tuning

- **GridSearchCV**: Performed exhaustive search over a specified parameter grid.
- **HalvingRandomSearchCV**: Efficient tuning by discarding underperforming models early.

Optimized parameters include:

- `C`: Regularization strength
- `gamma`: Kernel coefficient
- `kernel`: Choice of kernel (`rbf`, `linear`)

### 6. Model Performance

| Model               | Accuracy         |
|---------------------|------------------|
| Logistic Regression | 76.84%          |
| SVM (Baseline)      | 96.96%          |
| SVM (Optimized)     | **97.57%**      |

## 📂 Project Structure

```
├── data/
│   └── network_traffic.csv
├── models/
│   ├── logistic_regression.pkl
│   └── svm_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
└── README.md
```

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/network-traffic-classification.git
cd network-traffic-classification
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execute Pipeline

```bash
python src/train.py
python src/evaluate.py
```

## 📊 Results

- **Optimized SVM** achieved **97.57% accuracy** through fine-tuned hyperparameters.

## 📌 Dependencies

Ensure the following packages are installed:

```bash
scikit-learn
numpy
pandas
matplotlib
seaborn
```

## 📣 Contributions

Feel free to open issues and contribute to improving the classification model and extending the feature set.

## 📜 License

This project is licensed under the MIT License.

