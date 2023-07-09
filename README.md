# 10-Year Cardiovascular Risk Prediction using Naive Bayes Classifier

## 📋 Project Overview
This project aims to predict the 10-year risk of coronary heart disease (CHD) based on various risk factors using a Naive Bayes (NB) classifier. The Kaggle dataset used in this project is sourced from an ongoing cardiovascular study conducted in Framingham, Massachusetts. It consists of over 4,000 records and 15 attributes, including demographic, behavioral, and medical risk factors.

## Dataset Description
The dataset is structured as follows:

### :busts_in_silhouette: Demographic:
- Sex: Indicates the gender of the patient (male or female)
- Age: Represents the age of the patient
- Education: Represents the level of education of the patient

### :walking: Behavioral:
- is_smoking: Indicates whether the patient is a current smoker ("YES" or "NO")
- Cigs Per Day: Represents the average number of cigarettes smoked per day

### :hospital: Medical (history):
- BP Meds: Indicates whether the patient was on blood pressure medication
- Prevalent Stroke: Indicates whether the patient had previously experienced a stroke
- Prevalent Hyp: Indicates whether the patient was hypertensive
- Diabetes: Indicates whether the patient had diabetes

### :pill: Medical (current):
- Tot Chol: Represents the total cholesterol level
- Sys BP: Reflects the systolic blood pressure
- Dia BP: Reflects the diastolic blood pressure
- BMI: Represents the Body Mass Index
- Heart Rate: Reflects the heart rate
- Glucose: Represents the glucose level

### :chart_with_upwards_trend: Predict variable (desired target):
- TenYearCHD: Binary variable indicating the 10-year risk of future coronary heart disease ("1" for "Yes" and "0" for "No")




## :sparkles: Project Features
- Data Loading: The dataset was imported using the Pandas library and preprocessed for analysis.
- Outlier Detection and Removal: Statistical methods were employed to detect and remove outliers for robust analysis.
- Descriptive Statistics: Calculated measures of central tendency and dispersion provided insights into the dataset.
- Standardization: Features were standardized using calculated descriptive statistics for fair comparison.
- Data Split: Randomly split the dataset into training and testing sets with an 80%-20% proportion.
- Exploratory Data Analysis: Conducted visual analysis of feature distributions and tested for normality.
- Naive Bayes Classifier: Implemented NB classifier from scratch and compared performance with standard Python packages.


## :rocket: Getting Started
To get started with this project:
1. Open the `10Year_Cardiovascular_Risk_Factor.ipynb` notebook using Jupyter Notebook or any compatible Python development environment.
2. Execute the cells in the notebook sequentially to perform data preprocessing, model training, evaluation, and result analysis.
3. Follow the instructions and comments provided in the notebook for a detailed understanding of each step.
4. Explore the results and analysis presented in the notebook to gain insights into the model's performance and the factors influencing CHD prediction.

## :computer: Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Seaborn
- Scipy

## 📊 Results
The project successfully built a predictive model for CHD risk assessment. By leveraging the NB classifier and rigorous data preprocessing, descriptive analysis, and statistical testing, we obtained accurate predictions and valuable insights into the dataset. The model's performance was evaluated using accuracy metrics. The accuracy of the model is calculated using the `accuracy_score` function from the scikit-learn library. The resulting accuracy is found to be `88.74%`.

## :checkered_flag: Conclusion
This project demonstrates the application of machine learning techniques for cardiovascular risk prediction. By utilizing the NB classifier and implementing various data preprocessing and analysis techniques, we obtained valuable insights into the dataset and developed a reliable model for CHD 10-year risk assessment.

For detailed information and code implementation, please refer to the project notebook.

## :books: References
- Kaggle Dataset: [Cardiovascular Risk Dataset](https://www.kaggle.com/datasets/mamta1999/cardiovascular-risk-data)

## :busts_in_silhouette: Contributors
- Ahmed Elzayat
- Nourhan Ahmed
- Khaled Badr
- Hazem Zakaria
- Omar Nabil
