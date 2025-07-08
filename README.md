## Problem Statement
In today’s competitive job market, retaining skilled employees—especially in fast-growing fields like data science—is a significant challenge for companies. Hiring and training new employees is costly and time-consuming. Therefore, predicting which employees are likely to leave their jobs can help HR departments take proactive measures to improve retention.

This project aims to build a predictive model that determines the likelihood of a data scientist switching jobs. Using machine learning techniques, we analyze various features like education level, experience, training hours, company type, etc., to make an informed prediction.

## Why are we doing this?

**Business Need**
* Reduce attrition costs: Losing employees disrupts productivity and is expensive.
* Targeted retention efforts: HR can focus on employees most likely to leave.
* Better hiring decisions: Understand which candidate profiles are likely to stay long-term.

**Academic and Technical Goals**
* Apply a full ML pipeline: From EDA and preprocessing to model selection and deployment.
* Handle class imbalance using SMOTE.
* Use advanced algorithms like Logistic Regression, XGBoost, LightGBM.
* Perform feature engineering, model evaluation, and SHAP-based interpretability.

  ## Appendix

  **A. Dataset Description**
* Enrollee_id : Unique ID for enrollee
* City: City code
* Citydevelopmentindex: Developement index of the city (scaled)
* Gender: Gender of enrolee
* Relevent_experience: Relevent experience of enrolee
* Enrolled_university: Type of University course enrolled if any
*Education_level: Education level of enrolee
* Major_discipline :Education major discipline of enrolee
* Experience: Enrolee total experience in years
* Company_size: No of employees in current employer's company
* Company_type : Type of current employer
* Lastnewjob: Difference in years between previous job and current job
* Training_hours: training hours completed
* Target:
 >* 0 – Not looking for job change,
 >* 1 – Looking for a job change

**B. Tools & Technologies**
* Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn)
* Machine Learning: Logistic Regression, Random Forest, XGBoost, LightGBM
* Hyperparameter Tuning: GridSearchCV
* Model Explainability: SHAP
* Deployment: Flask (for optional web app)
* EDA & Visualization: Seaborn, Matplotlib

**C. Performance Metrics Used**
* ROC-AUC Score: To evaluate model's ability to distinguish classes
* Confusion Matrix: To analyze true/false positives and negatives
* Classification Report: Precision, Recall, F1-Score
* Feature Importance (Model & SHAP): To understand feature contribution

**D. Preprocessing Steps**
* Label Encoding for binary categories
* One-Hot Encoding for nominal categorical features
* Missing value handling using SimpleImputer
* Standardization using StandardScaler
* Imbalanced data handled using SMOTE

**E. Model Comparison Summary**
(Model	ROC-AUC Score)
* Logistic Regression	0.72
* Random Forest	0.77
* XGBoost	0.78
* LightGBM	**0.81**

## **Future Scope**

* **Model Deployment in Real-Time HR Systems-**
>Integrate the model into HR tools to provide instant predictions about which employees are at risk of leaving, enabling proactive retention strategies.

* **More Sophisticated Feature Engineering-**
> Generate interaction features (e.g., experience × company size).
> Use NLP techniques on job descriptions (if available) for richer inputs.
>Time-based features for modeling employee tenure more accurately.

* **Ensemble Stacking or Blending-**
>Improve prediction accuracy by combining multiple models (e.g., stacking Logistic Regression, XGBoost, and LightGBM).

* **Continuous Model Retraining-**
>Set up automated retraining pipelines as new data (e.g., from employee feedback or exit surveys) becomes available.

* **Geographic or Department-Wise Risk Segmentation**
>Extend the model to give department- or location-specific attrition risks and recommendations.

* **Integration with External Datasets**
>Include external job market trends, company reviews (e.g., Glassdoor), or macroeconomic factors that influence employee decisions.

## References

SMOTE: Synthetic Minority Over-sampling Technique

Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.

https://arxiv.org/abs/1106.1813

XGBoost

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

https://doi.org/10.1145/2939672.2939785

LightGBM

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T.Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.

https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

SHAP (SHapley Additive exPlanations)

Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems.

https://github.com/slundberg/shap

Scikit-learn Documentation

https://scikit-learn.org/stable/user_guide.html

Source for machine learning algorithms, pipelines, preprocessing, evaluation metrics
