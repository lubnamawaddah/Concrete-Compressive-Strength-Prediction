<div align="center">  
  <h1>Machine Learning Regression for Concrete Compressive Strength Prediction</h1>  
</div>

<div align="center">  
  <h2>Project Domain</h2>
</div>

Concrete is widely used in construction due to its desirable engineering properties. Its high strength, especially when reinforced, combined with its ability to be shaped and hardened at room temperature, makes it a top choice for building apartments and high-rise structures. Additionally, its resistance to heat and water makes reinforced concrete ideal for structures exposed to extreme conditions, such as tunnels, bridges, dams, and reservoirs [1].

Understanding the behavior of concrete structures under external loads and improving design methods requires a thorough study of its mechanical properties. Among these, compressive strength is the most critical, as it directly affects structural safety and is essential for assessing performance throughout a structure's lifecycle, from initial design to aging assessments. However, predicting compressive strength is challenging due to the complex composition of concrete, which includes coarse and fine aggregates, cement paste, and admixtures that are randomly distributed. Traditionally, compressive strength is measured through physical experiments. This involves preparing cube or cylinder samples with specific mix ratios, curing them for a set time, and testing them using compression machines. While effective, this approach is both time-consuming and costly, reducing efficiency in practical applications [2].

In recent years, machine learning (ML), a branch of artificial intelligence (AI), has emerged as a powerful tool for predicting concrete compressive strength. Techniques such as neural networks, linear regression, decision trees, and support vector machines are commonly employed. ML excels in regression tasks by learning patterns directly from input data, resulting in faster, more accurate, and cost-effective predictions compared to traditional regression methods. This makes ML a valuable approach in the field [3], [4].

In this project, concrete strength was predicted using Linear Regression, Random Forest, Gradient Boosting Regressor, and Support Vector Regressor to identify the most effective model. The performance of each model was evaluated using multiple metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Coefficient of Determination (R²), and Mean Absolute Percentage Error (MAPE). These metrics were used to comprehensively assess the accuracy and reliability of the models.

---

**Reference :**

[1] Hoang Nguyen et al., ["Efficient machine learning models for prediction of concrete strengths"](https://www.sciencedirect.com/science/article/pii/S095006182032955X) Construction and Building Materials, Vol. 266, Part B, 2021.

[2] De-Cheng Feng et al., ["Machine learning-based compressive strength prediction for concrete: An adaptive boosting approach"](https://www.sciencedirect.com/science/article/pii/S0950061819324420) Construction and Building Materials, Vol. 230, 2020.

[3] Kadir Güçlüer et al., ["A comparative investigation using machine learning methods for concrete compressive strength estimation"](https://www.sciencedirect.com/science/article/pii/S2352492821002701) Materials Today Communications, Vol. 27, 2021.

[4] Hadi Salehi et al., ["Emerging artificial intelligence methods in structural engineering"](https://www.sciencedirect.com/science/article/pii/S0141029617335526) Engineering Structures, Vol. 171, 2018.

<div align="center">  
  <h2>Business Understanding</h2>
</div>

### Problem Statements
1. How can the compressive strength of concrete be predicted based on its composition of materials?
2. How can machine learning models be evaluated to ensure that the prediction of concrete compressive strength is accurate and reliable?

---

### Goals
1. Building a machine learning model to predict concrete compressive strength based on input variables such as material composition, concrete age, and others.
2. Evaluating the model's performance and selecting the best algorithm based on evaluation metrics.
---

### Solution Statements
1. Using machine learning algorithms such as Linear Regression, Random Forest, Gradient Boosting Regressor, and Support Vector Regression to build the prediction model.
2. Measuring the model's performance using evaluation metrics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R² (R-squared), and MAPE (Mean Absolute Percentage Error).

<div align="center">  
  <h2>Data Understanding</h2>
</div>

### Dataset Description

The dataset used is the "Concrete Compressive Strength" available at : [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength). This dataset contains information about the composition of concrete materials and the resulting compressive strength.


### Attributes in the Dataset

**Feature :**
- **Cement (kg/m³) :** The amount of cement used.
- **Blast Furnace Slag (kg/m³) :** The amount of blast furnace slag used.
- **Fly Ash (kg/m³) :** The amount of fly ash used.
- **Water (kg/m³) :** The amount of water used.
- **Superplasticizer (kg/m³) :** The amount of superplasticizer used.
- **Coarse Aggregate (kg/m³) :** The amount of coarse aggregate used.
- **Fine Aggregate (kg/m³) :** The amount of fine aggregate used.
- **Age (day) :** The age of the concrete in days.

**Target :**
- **Concrete compressive strength (MPa) :** The compressive strength of the concrete.

### Summary Statistics
- Number of Data Points : 1030
- Number of Attributes : 9
- Attribute Details : 8 quantitative input variables, and 1 quantitative output variable
- Missing Attribute Values : None

---

### Rename Column

The column names are renamed to make them more concise.
- Cement (kg/m³) -> Cement
- Blast Furnace Slag (kg/m³) -> Blast Furnace Slag
- Fly Ash (kg/m³) -> Fly Ash
- Water (kg/m³) -> Water
- Superplasticizer (kg/m³) -> Superplasticizer
- Coarse Aggregate (kg/m³) -> Coarse Aggregate
- Fine Aggregate (kg/m³) -> Fine Aggregate
- Age (day) -> Age
- Concrete compressive strength (MPa) -> Concrete Compressive Strength

### Check Missing Value

It was found that there were no missing values, so no handling was necessary.

### Check Duplicate Data

The data contains 25 duplicate entries, these duplicate data will be removed during the data preparation stage.

### Check Outliers

![outliers_before](https://github.com/user-attachments/assets/ef321af9-194a-4968-866c-d5bba26f2af8)

It was found that the data contained outliers, these outliers need to be removed during the data preparation stage.

---

### Exploratory Data Analysis

- **Distribution of Each Feature**

  ![distribution_feature](https://github.com/user-attachments/assets/412b68d5-2739-4268-913c-dec246fbb583)

  Insight:
  **1. Cement :** The distribution shows a right-skewed pattern, indicating that most samples have lower cement content, with fewer samples having higher values.
  **2. Blast Furnace Slag :** The distribution is heavily right-skewed, with most samples showing very low usage, confirming that it is less common in the mix.
  **3. Fly Ash :** Similar to Blast Furnace Slag, the distribution is right-skewed, indicating low usage across samples, reinforcing its rarity in the concrete mix.
  **4. Water :** The distribution is relatively uniform, peaking around 180-200, suggesting a balanced and consistent application of water in the samples.
  **5. Superplasticizer :** The distribution shows that primarily low amounts are used, with a significant number of samples having very little to none, indicating sparing application.
  **6. Coarse Aggregate :** The distribution appears normal, with a peak around 950-1000, indicating a typical range of usage for coarse aggregates in the samples.
  **7. Fine Aggregate :** This distribution is also normal, peaking around 750-800, suggesting a consistent application of fine aggregates.
  **8. Age :** The distribution is right-skewed, indicating that most samples are relatively recent, falling within the 0-40 days range.
  **9. Concrete Compressive Strength :** The distribution shows that most samples fall within the 30-40 MPa range, indicating typical strength levels for the concrete.
---

- **Correlation Heatmap**

  ![heatmap](https://github.com/user-attachments/assets/d88de658-c4b9-4d2f-a230-ae68a0068402)

---

- **Relationship between Concrete Age and Compressive Strength**

  ![age_comprressive_strength](https://github.com/user-attachments/assets/6ac9bd61-0b87-49b7-b025-bbe3c0572c28)

  Insight:

  The scatter plot illustrates a positive relationship between concrete age and compressive strength, indicating that strength generally increases as the age of the concrete advances. However, there is significant clustering of data points at specific ages, particularly at 0, 28, and 56 days. This clustering suggests that these ages are common testing points. Additionally, there is notable variability in compressive strength at lower ages, indicating that while age is a contributing factor to strength, other variables may also influence the results.

---

- **Relationship between Cement Amount and Compressive Strength**

  ![cement_compressive_strength](https://github.com/user-attachments/assets/23c87475-7163-4c7e-baf1-c78ee0022a84)

  Insight:

  The scatter plot demonstrates a positive correlation between cement amount and concrete compressive strength, indicating that higher cement content generally contributes to increased strength. Most data points cluster around lower cement amounts (100-300 kg), but there is still considerable variability in compressive strength at higher cement levels (300-500 kg). This variability suggests that while cement content is a significant factor in determining strength, other variables may also play a role. Overall, this relationship highlights the critical importance of cement in influencing concrete quality.

---

<div align="center">  
  <h2>Data Preparation</h2>
</div>

In this stage, several data preparation techniques were applied to ensure that the data is in the correct format for model training and evaluation. The data preparation steps taken are as follows:

**1. Remove Duplicate Data**

During the data understanding stage, it was found that there were 25 duplicate entries, so these duplicate entries were removed.

**2. Remove Outliers**

During the data understanding stage, it was found that the data has outliers, then the outlier data is cleaned using the IQR (Interquartile Range) method.

The IQR method works by calculating the first quartile (Q1) and third quartile (Q3) of the data, and identifying outliers as values that fall outside the range defined by 1.5 times the IQR below Q1 or above Q3.

**3. Separate Features and Target Variable**

After the data is clean, next step is separate the dataset into features (X) and target variable (y). This is because machine learning models require a clear distinction between input variables (features) and output variables (target).

**4. Splitting the Data into Training and Testing Sets**

After separating the features and target, the dataset is split into training and testing sets. This step is important to ensure that the model is trained on one portion of the data and evaluated on another portion that it has not seen before. The training set is used to train the model, while the testing set helps assess the model's performance and generalization ability. In this case, the data is split with an 80:20 ratio, meaning 80% of the data is used for training and 20% is used for testing.

**5. Scaling Data**

Data scaling is applied using StandardScaler. This technique standardizes the features by removing the mean and scaling them to unit variance. Standardizing the data ensures that all features contribute equally to the model, preventing features with larger values from dominating the learning process.

--- 

<div align="center">  
  <h2>Modeling</h2>
</div>

In this stage, several machine learning models are used to predict the compressive strength of concrete based on various components in the concrete mix. Below is an explanation of the three models applied.

### **1. Linear Regression**

  ```
  lr = LinearRegression()
  lr.fit(X_train_scaled, y_train)
  y_pred_lr = lr.predict(X_test_scaled)
  ```

  The Linear Regression (LR) model works by fitting a linear relationship between the input features (X) and the target variable (y). It assumes that the target variable can be predicted as a weighted sum of the input features plus an intercept term. The goal is to find the best-fitting line by minimizing the sum of squared errors between the predicted values and the actual values in the training data.

  **Parameters Used :**

  The parameters used in this model are the default parameters of the LR in scikit-learn. The default parameters for LinearRegression() are as follows:
  - ``` fit_intercept=True ``` : This indicates that the model will calculate an intercept (bias) term. By default, it is set to True, meaning the intercept will be included in the model.
  - ``` normalize=False ``` : This controls whether the input features should be normalized (centered and scaled) before fitting. By default, it is set to False, meaning the data is not normalized. Normalization is not done here unless explicitly specified.
  - ``` copy_X=True ``` : This determines whether to copy the input data X before fitting. By default, it is True, meaning the input data will be copied.
  - ``` n_jobs=None ``` : This controls the number of CPU cores used during the fitting process. None means that a single core will be used by default.
  - ``` positive=False ``` : This parameter controls whether the coefficients should be forced to be positive. By default, it is False, meaning the model will not enforce positive coefficients.

  **Model Advantages :**
  - Linear Regression is a simple and easy-to-understand model.
  - It has high interpretability, making it easy to explain the relationship between features and the target.
  - It is fast in training, especially for datasets with fewer features.
  
  **Model Disadvantages :**
  - It cannot handle non-linear relationships between features and the target.
  - It is sensitive to outliers, which can significantly affect prediction results.
  - Its performance is poor on complex datasets or datasets with many interacting features.

---

### **2. Random Forest**

  ```
  rf = RandomForestRegressor(random_state=42)
  rf.fit(X_train_scaled, y_train)
  y_pred_rf = rf.predict(X_test_scaled)
  ```

  Random Forest is an ensemble learning algorithm that builds multiple decision trees and merges them together to get a more accurate and stable prediction. The key idea is that individual decision trees are likely to overfit, but by aggregating the predictions of multiple trees, the Random Forest model reduces the variance and improves generalization. Each tree is trained on a random subset of the data with random feature selection at each split, making the model robust against overfitting.
  
  **Parameters Used :**
  - ``` random_state=42 ``` : This parameter is used to ensure reproducibility of the results. It sets the random seed so that the model produces the same results each time it is run. This helps in comparing models consistently.

  **Model Advantages :**
  - Random Forest is an ensemble-based algorithm that can handle non-linear relationships and interactions between features.
  - It is resistant to overfitting, especially when the number of trees is sufficiently large.
  - It can handle data with many features and does not require extensive data preprocessing.

  **Model Disadvantages :**
  - It tends to be slower in training compared to simpler models like Linear Regression.
  - The model is less interpretable, especially with a large number of trees in the ensemble.
  - It requires more computational resources (memory and time) compared to simpler models.

---

### **3. Gradient Boosting Regressor**

  ```
  model = GradientBoostingRegressor(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)
  y_pred_gb = model.predict(X_test_scaled)
  ```

  Gradient Boosting is an ensemble learning algorithm that builds a series of decision trees sequentially. Unlike Random Forest, which builds trees independently, Gradient Boosting trains trees in a way that each new tree corrects the errors made by the previous trees. The model focuses on minimizing the loss function (error) by adjusting the weights of the observations that were mispredicted by the previous trees. This approach leads to high accuracy and is particularly effective in solving complex regression problems.

  **Parameters Used :**
  - ``` n_estimators=100 ``` : The number of decision trees to be built. In this case, 100 trees will be built during the training process. More trees generally improve the model's accuracy but can also increase computation time.
  - ``` random_state=42 ``` : This ensures reproducibility of the results by fixing the random seed. It allows for consistent results each time the model is run.

  **Model Advantages :**
  - Gradient Boosting is a powerful ensemble algorithm for handling non-linear problems.
  - It can handle complex data and provides excellent results in many regression tasks.
  - It tends to perform well with data that has many variables and interactions between features.

  **Model Disadvantages :**
  - It has longer training times, especially on large datasets.
  - It is more prone to overfitting if not properly tuned.
  - It is less interpretable compared to linear models.

---

### **3. Support Vector Regressor**

  ```
  model = SVR()
  model.fit(X_train_scaled, y_train)
  y_pred_svr = model.predict(X_test_scaled)
  ```

  Support Vector Regressor (SVR) is a machine learning algorithm based on the concept of Support Vector Machines (SVM), which is used for regression tasks. SVR tries to fit the best line (or hyperplane in higher dimensions) within a margin of tolerance defined by the epsilon parameter. The goal is to minimize the prediction error while keeping the margin as large as possible. Unlike linear regression, SVR uses kernel functions to transform the input data into a higher-dimensional space to handle non-linear relationships between the features and the target variable.

  **Parameters Used :**

  The parameters used in this model are the default parameters of the SVR class in scikit-learn. The default parameters for SVR() are as follows:
  - ``` C=1.0 ``` : Regularization parameter that controls the trade-off between fitting the model and keeping it simple. Default is 1.0.
  - ``` kernel='rbf' ``` : Specifies the Radial Basis Function kernel, used for non-linear relationships.
  - ``` degree=3 ``` : Degree of the polynomial kernel (not used here as the RBF kernel is selected).
  - ``` gamma='scale' ``` : The coefficient for the RBF kernel, calculated as 1 / (n_features * X.var()).
  - ``` epsilon=0.1 ``` : Tolerance for error where no penalty is applied.
  - ``` random_state=None ``` : Controls randomness, not set, so results may vary across runs.

  **Model Advantages :**
  - SVR can handle non-linear relationships by using kernel functions.
  - SVR, with proper tuning of parameters, is less prone to overfitting, especially in high-dimensional spaces.
  - SVR performs well even when there are more features than data points.

  **Model Disadvantages :**
  - SVR can be computationally expensive, especially on large datasets with many features.
  - Finding the right parameters (like C, epsilon, and the kernel) requires careful tuning, and the model is sensitive to these parameters.
  - Similar to other kernel-based methods, the SVR model is harder to interpret compared to simpler models like Linear Regression.

---

<div align="center">  
  <h2>Evaluation</h2>
</div>

During the evaluation phase, various metrics were used to measure the performance of the model in predicting the compressive strength of concrete. The metrics used include MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R² (R-squared), and MAPE (Mean Absolute Percentage Error). These metrics were chosen because they are appropriate for regression tasks, where the primary goal is to minimize prediction errors.

### **1. MAE (Mean Absolute Error)**

**Explanation :**  
MAE measures the average absolute difference between the predicted values and the actual values. The lower the MAE value, the better the model is at making accurate predictions.

**Formula :**
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- \( y_i \) is the actual value
- \( \hat{y}_i \) is the predicted value
- \( n \) is the number of data points


---

### 2. **MSE (Mean Squared Error)**

**Explanation :**  
MSE measures the average squared difference between the predicted values and the actual values. MSE penalizes larger errors more heavily, making it more sensitive to outliers.

**Formula :**
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
Where :
- $ y_i $ is the actual value
- $ \hat{y}_i $ is the predicted value
- $ n $ is the number of data points

---

### 3. **RMSE (Root Mean Squared Error)**

**Explanation :**  
RMSE is the square root of MSE and measures the prediction error in the same units as the original data. RMSE provides a clearer picture of both large and small model errors.

**Formula :**
$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$
Where :
- $ y_i $ is the actual value
- $ \hat{y}_i $ is the predicted value
- $ n $ is the number of data points

---

### 4. **R² (R-squared)**

**Explanation :**  
R² measures the proportion of variation in the data that can be explained by the model. The R² value ranges from 0 to 1, where 1 indicates that the model explains all the variation in the data, and 0 indicates that the model performs no better than the mean model.

**Formula :**
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
Where :
- $ y_i $ is the actual value
- $ \hat{y}_i $ is the predicted value
- $ \bar{y} $ is the mean of the actual data
- $ n $ is the number of data points

---

### 5. **MAPE (Mean Absolute Percentage Error)**

**Explanation :**  
MAPE measures the average percentage error between the predicted values and the actual values. MAPE is easier to interpret as it presents the error in percentage terms. The lower the MAPE value, the better the model is at predicting the actual values.

**Formula :**
$$
MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
$$
Where :
- $ y_i $ is the actual value
- $ \hat{y}_i $ is the predicted value
- $ n $ is the number of data points

---

### **Result Based on Evaluation Metrics**

Based on the evaluation conducted on the applied models, the results are as follows:

**1. Linear Regression**

| Metric     | Value  |
|------------|--------|
| **MAE**    | 5.28   |
| **MSE**    | 48.00  |
| **RMSE**   | 6.93   |
| **R²**     | 0.82   |
| **MAPE**   | 21.90% |

  Explanation :

  - MAE indicates that the average prediction error is 5.28 MPa, which is relatively significant compared to the target value range (concrete compressive strength).
  - MSE and RMSE show higher error values, suggesting that the model tends to have larger errors in some predictions.
  - R² of 0.82 demonstrates that the model can explain 82% of the variance in the data. This is a good value, though there is still room for improvement.
  - MAPE of 21.90% shows that the model has an average prediction error of approximately 21.9% from the actual values, which is acceptable but indicates potential for further error reduction.

---

**2. Random Forest**

| Metric     | Value  |
|------------|--------|
| **MAE**    | 3.17   |
| **MSE**    | 19.28  |
| **RMSE**   | 4.39   |
| **R²**     | 0.93   |
| **MAPE**   | 12.56% |

  Explanation :

  - MAE of 3.17 indicates that the model has a relatively low average prediction error, suggesting high accuracy in predicting concrete compressive strength.
  - MSE and RMSE are significantly lower, demonstrating that the model is more consistent in its predictions and has smaller errors compared to previous models.
  - R² of 0.93 shows that the model can explain 93% of the variance in the data, which is an excellent result and indicates a very good fit.
  - MAPE of 12.56% reveals that the model has a low percentage error in predictions, signifying high performance and reliable forecasting of concrete strength.

---

**3. Gradient Boosting Regressor**

| Metric     | Value  |
|------------|--------|
| **MAE**    | 3.56   |
| **MSE**    | 23.08  |
| **RMSE**   | 4.80   |
| **R²**     | 0.91   |
| **MAPE**   | 13.52% |

  Explanation :

  - MAE of 3.56 indicates a moderate prediction error, showing the model's ability to estimate concrete compressive strength with reasonable accuracy.
  - MSE and RMSE values suggest some variability in predictions, with the model experiencing moderate levels of error.
  - R² of 0.91 demonstrates that the model explains 91% of the variance in the data, which is a very good result and indicates strong predictive performance.
  - MAPE of 13.52% shows that the model's predictions deviate by approximately 13.5% from actual values, which is acceptable but indicates potential for further refinement.

---

**4. Support Vector Regressor**

| Metric     | Value  |
|------------|--------|
| **MAE**    | 6.62   |
| **MSE**    | 71.58  |
| **RMSE**   | 8.46   |
| **R²**     | 0.73   |
| **MAPE**   | 30.42% |

  Explanation :

  - MAE of 6.62 indicates a relatively high prediction error, suggesting that the model struggles to accurately estimate concrete compressive strength.
  - MSE and RMSE show significantly larger error values, demonstrating considerable variability and inconsistency in the model's predictions.
  - R² of 0.73 indicates that the model explains 73% of the variance in the data, which is moderate but leaves a substantial portion of the data unexplained.
  - MAPE of 30.42% reveals a high percentage error, suggesting that the model's predictions deviate considerably from actual values and may not be reliable for precise estimations.

---

### **Best Model Selection**

After comparing the four models, the Random Forest model was chosen as the best model based on the evaluation results. Here are the reasons why Random Forest was selected:

- Random Forest demonstrated better performance with a higher R² value and lower MAE.
- Random Forest is capable of handling non-linear relationships between features, which Linear Regression cannot address, and is more flexible than Gradient Boosting in certain cases.
- Random Forest tends to be more resistant to overfitting, making it a more stable choice for this dataset.

---

### **Conclusion**
The performance of the four regression models: **Linear Regression, Random Forest, Gradient Boosting Regressor, and Support Vector Regressor** was evaluated using several metrics.

- **Linear Regression** demonstrated decent performance with an R² score of 0.82, indicating it explains 82% of the variance in the target variable. However, it had relatively higher error metrics, with a Mean Absolute Error (MAE) of 5.28, Mean Squared Error (MSE) of 48.00, and Root Mean Squared Error (RMSE) of 6.93, suggesting a higher degree of error in predictions. The Mean Absolute Percentage Error (MAPE) of 21.90% indicates that the model's predictions were off by an average of 21.90%.

- **Random Forest** outperformed the other models, achieving the highest R² score of 0.93, which means it explains 93% of the variance in the target variable. The error metrics were significantly lower compared to Linear Regression, with an MAE of 3.17, MSE of 19.28, and RMSE of 4.39, indicating better accuracy. The MAPE of 12.56% suggests a relatively lower error rate in predictions.

- **Gradient Boosting Regressor** also performed well, with an R² score of 0.91, indicating it explains 91% of the variance. While its error metrics were slightly higher than those of Random Forest (with an MAE of 3.56, MSE of 23.08, and RMSE of 4.80), it still outperformed Linear Regression. The MAPE of 13.52% suggests slightly higher relative errors compared to Random Forest but remains better than Linear Regression.

- **Support Vector Regressor** had the lowest performance among the models, with an R² score of 0.73, explaining only 73% of the variance. Its error metrics were higher, with an MAE of 6.62, MSE of 71.58, and RMSE of 8.46, indicating a greater degree of error in predictions. The MAPE of 30.42% reflects a significant average error in its predictions.

In conclusion, **Random Forest** performed the best in terms of both accuracy and error metrics, followed closely by **Gradient Boosting Regressor**. **Linear Regression** showed moderate performance, while **Support Vector Regressor** was the least effective for this dataset, making it less suitable for accurate predictions.