## Airline Passenger Satisfaction Prediction and Key Driver Analysis

## 🎯 Project Objective
This project aims to develop a robust machine learning model to accurately predict airline passenger satisfaction and, crucially, to identify the specific service and operational factors that have the greatest influence (both positive and negative) on the overall passenger experience.
The ultimate goal is to provide the airline with actionable, data-driven insights to prioritize service improvements, enhance customer loyalty, and ultimately drive revenue growth in a competitive market.

## 📊 Dataset
The analysis is based on the Kaggle Airline Passenger Satisfaction Dataset, a comprehensive survey containing over 100,000 entries. 
The dataset includes:
1. Target Variable: Satisfaction (Binary: Satisfied vs. Neutral/Dissatisfied)
2. Ordinal Service Ratings (14 features): Scores from 0 to 5 for factors like Seat comfort, Inflight wifi service, Ease of Online booking, and Cleanliness.
3. Continuous Variables: Age, Flight Distance, Departure Delay in Minutes, and Arrival Delay in Minutes.
4. Nominal/Categorical Variables: Customer Type, Type of Travel, Gender, and Class.

## ⚙️ Methodology and Workflow
The project followed a standard machine learning pipeline:
**1. Exploratory Data Analysis (EDA)**
   Initial analysis focused on visualizing feature distributions and direct relationships with the target variable using Seaborn.
   **Segmentation Analysis**: Confirmed that Loyal Customers and passengers in Business Class drive the majority of the satisfied population.
   **Operational Friction**: Visualized via stacked histograms, confirming that Departure Delay in Minutes directly and negatively correlates with satisfaction, causing a proportional rise in the "Neutral or Dissatisfied" segment as delay time increases.
   **Service Thresholds**: Identified that service ratings must maintain a minimum score of 3/5 to avoid dissatisfaction, with ratings of 4 or 5 providing a disproportionate boost to satisfaction (non-linear effect).
  
 ##  Data Preprocessing & Feature Engineering
Preparing the data for high-performance classification models:
    
**Missing Value Imputation**: Handled missing values (primarily in Arrival Delay in Minutes) using Median Imputation, a robust method suitable for skewed delay distributions.
    
**Encoding**: Applied One-Hot Encoding to nominal categorical features (Gender, Type of Travel) to prevent the model from misinterpreting them as ordinal.
   
**Scaling**: Applied StandardScaler to continuous numerical features (Age, Flight Distance, Delays) to normalize their scale and ensure equal contribution during modeling.
    
**Data Integrity Check**: Crucially, steps were taken to prevent data leakage, ensuring the predictive model was trained without access to the target variable's numerical representation, leading to a valid and generalizable result.
    
## ** Model Training and Evaluation **
   Multiple classification algorithms were tested (including Decision Tree and XGBoost) before selecting the optimal ensemble method.
    
 **Model**: Tuned Random Forest Classifier.
 **Hyperparameter Tuning**: Employed techniques to optimize parameters like n_estimators and max_depth to prevent overfitting.
 **Performance**: The Random Forest model demonstrated the highest predictive capability on the unseen test set, confirming its reliability as an operational tool.
    
    **Model Accuracy**
    F1 Score    0.9425
    Tuned Random Forest   60.9473 
    
 ##  🌟 Key Findings & Actionable Insights
Feature importance analysis identified the factors most strongly correlated with positive and negative satisfaction:
    
Top 3 Drivers of Satisfaction (Highest ROI)
Seat Comfort: The foundational physical experience remains a top predictor of overall contentment
Inflight Entertainment: The quality of the digital experience on board is a major factor, particularly when ratings reach 4 or 5.
Ease of Online Booking: The pre-flight digital experience is nearly as critical as the in-flight service, underscoring the necessity of a seamless UI/UX.
    
Primary Dissatisfiers (Risk Factors)
Operational Delay: The duration of the Departure Delay is a direct and quantifiable source of negative sentiment, confirming that timely operation is key to mitigating dissatisfaction
Loyalty Risk: Disloyal Customers are highly likely to report dissatisfaction, confirming that targeted service recovery efforts should be focused on new or non-frequent travelers.
    
    
    
   ##  **💻 Technical Stack and Setup**
    
This project requires a Python environment with the following libraries:bash
   
**Core Data Analysis and Manipulation**
import pandas as pdimport numpy as np
    
 **Visualization**
   import seaborn as snsimport matplotlib.pyplot as plt
    
**Machine Learning and Preprocessing**
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report***

## Next Steps (Future Work)

Future iterations of this project could include:

1.  **Causal Inference:** Using libraries like `DoWhy` to move beyond correlation to establish definitive **causal links** between service improvements and overall satisfaction, providing a clear ROI for every investment.[7]
2.  **Model Deployment:** Implementing the high-performance Random Forest model as a 
