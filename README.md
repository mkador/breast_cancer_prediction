# 🧠 Breast Cancer Prediction with SVM

This project demonstrates how to use **Support Vector Machines (SVM)** for binary classification to predict whether a tumor is **malignant** or **benign** based on features from a real-world breast cancer dataset.

---

## 📂 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Data Set**, which can be found on:
- [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

This dataset contains 569 instances and 30 numerical features derived from digitized images of fine needle aspirates (FNA) of breast masses.

---

## 📌 Objective

- Load and explore the dataset
- Apply **Support Vector Machine (SVM)** classification
- Perform **hyperparameter tuning** using `GridSearchCV`
- Evaluate and visualize the results

---

## 🔧 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📊 Workflow

1. **Data Loading & Exploration**
   - Dataset loaded from CSV
   - Basic statistics and visualizations to understand feature distributions

2. **Data Preprocessing**
   - Feature scaling
   - Train-test split

3. **Model Building**
   - Use of `SVC` (Support Vector Classification)
   - Hyperparameter tuning via `GridSearchCV` for:
     - `C`: Regularization parameter
     - `gamma`: Kernel coefficient
     - `kernel`: Linear, RBF, etc.

4. **Evaluation**
   - Confusion matrix
   - Accuracy score
   - Classification report

5. **Visualization**
   - Heatmaps for correlation
   - Visual comparisons of different SVM kernels

---

## ✅ Results

- The model achieves high accuracy in classifying tumors.
- Optimal parameters identified through `GridSearchCV` boost model performance.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mkador/breast-cancer-svm.git
   cd breast-cancer-svm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Breast_cancer_prediction_with_SVM.ipynb
   ```

---

## 📁 File Structure

```
breast-cancer-svm/
│
├── Breast_cancer_prediction_with_SVM.ipynb   # Main notebook
├── Breast_Cancer_Diagnostic.csv              # Dataset
├── README.md                                 # Project overview
└── requirements.txt                          # Python dependencies
```

---

## 📌 References

- UCI Machine Learning Repository
- Scikit-learn Documentation
- [Support Vector Machines — scikit-learn docs](https://scikit-learn.org/stable/modules/svm.html)

---

## 📬 Contact

For questions or suggestions, feel free to reach out via [email](mkador169@gmail.com)
