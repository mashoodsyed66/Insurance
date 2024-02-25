# Medical Insurance Cost prediction

**Introduction:**

Welcome to the Medical Cost Personal Datasets project! Here, we're diving into healthcare expenses. We want to figure out what makes medical costs go up or down. By using data analysis and machine learning, we're hoping to find some answers that can help us understand healthcare spending better.

## 1. **About the Data:**

We're using a dataset called Medical Cost Personal Datasets. It's got a bunch of information about people's health insurance and how much they spend on medical stuff. Let's break down the main things we're looking at:

1. **Age**: This tells us how old the main person with the insurance is. Different ages might mean different healthcare needs and costs.

2. **Sex**: This just tells us if the person is a man or a woman. Sometimes, healthcare costs can be different for men and women.

3. **BMI (Body Mass Index)**: It's a number that shows if someone's weight is healthy for their height. If someone's BMI is high, they might have more health problems, which could mean higher healthcare costs.

4. **Children**: This tells us how many kids are covered by the health insurance. More kids might mean more visits to the doctor and more medical bills.

5. **Smoker**: It's a yes or no answer to whether the person smokes. Smoking can lead to health issues, so smokers might have higher medical costs.

6. **Region**: It tells us where the person lives in the United States. Different areas might have different healthcare costs and access to healthcare.

7. **Charges**: This is the most important thing we're looking at. It's how much money the insurance has to pay for medical bills. Understanding why these charges vary can help us plan better for healthcare expenses.


## 2. **Objective:**
We're using this data to try and understand why healthcare costs vary so much , and see if we can predict them. And what are the most important factors for insurance charges. Our goal is to make healthcare more understandable and help people plan for medical expenses better.
<br> </br>

## 3. **EDA and Preprocessing:**
In the boxplots below, you'll notice some dots that are far away from the main bunch, these are called outliers. For **BMI (Body Mass Index)** , a few people have really high  BMIs compared to most others. Also In the **Charges** column, there are a lot of dots that stand out. This means some people have medical bills that are way higher than usual.
![output_8_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/c73d30c3-da14-40b0-88fc-b99c2df09ca5)



### Investigating Outliers

To identify outliers in the `charges` column, we calculated the Interquartile Range (IQR) and used it to determine the upper and lower bounds. Any data points falling outside these bounds are considered outliers.

```python
column_name = 'charges' 
# Calculate the IQR (Interquartile Range) for the column
Q1 = data[column_name].quantile(0.25)
Q3 = data[column_name].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify and access the outlier values for the specified column
outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]

nosmoke = outliers[outliers['smoker']=='no']
print(nosmoke)
```

**Table**

Upon further investigation of these outliers, we found that out of the 139 cases, only 3 were non-smokers. The remaining outliers with significantly higher charges were all smokers.

|   age | sex    |   bmi |   children | smoker   | region    |   charges |
|------:|:-------|------:|-----------:|:---------|:----------|----------:|
|    55 | female |  26.8 |          1 | no       | southwest | 35160.1   |
|    61 | female |  33.33|          4 | no       | southeast | 36580.3   |
|    59 | female |  34.8 |          2 | no       | southwest | 36910.6   |

This table highlights that while outliers in medical charges exist across various demographics, the majority of outliers with substantially higher charges are associated with smokers.
<br> </br>

Also from below visual we can see that the majority **Charges** that are very high are smokers and also have **BMI** greater than 30 which means they are obese.

![download](https://github.com/mashoodsyed66/Insurance/assets/65015378/429ce23c-7745-41cc-891e-3a1b6ad39028)


### Encoding
Next, we performed encoding transformations to prepare our categorical data for machine learning algorithms. For the 'sex' and 'smoker' columns, we converted categorical values ('male'/'female', 'yes'/'no') into binary representations (1/0), facilitating algorithmic understanding. Additionally, we utilized one-hot encoding on the 'region' column, creating separate binary columns for each region category. This ensures our model can effectively interpret and utilize categorical data. Below is the code implementing these transformations:

```python
data_ohe = data.copy()
data_ohe['sex'] = data_ohe['sex'].apply(lambda x: 1 if x == 'male' else 0)  # 1 for 'male', 0 for 'female'
data_ohe['smoker'] = data_ohe['smoker'].apply(lambda x: 1 if x == 'yes' else 0)  # 1 for 'yes', 0 for 'no'
data_ohe = pd.get_dummies(data_ohe, columns=['region'], prefix=['region'])
```
![Capture](https://github.com/mashoodsyed66/Insurance/assets/65015378/e39e61b3-fabe-43f1-8d03-880f848e6371)





## 4. **Model Training**

We trained three different linear regression models to predict medical charges using various techniques. 

### Model 1: Simple Linear Regression
- **Mean Squared Error (MSE):** 5812.10
- **Mean Absolute Error (MAE):** 4145.45
- **R-squared (R2):** 0.77

### Model 2: Polynomial Regression with Ridge Regularization
- **R-squared (R2):** 0.85
- **MSE:** 4724.27

### Model 3: Polynomial Regression with Lasso Regularization and Box-Cox Transformation
- **R-squared (R2):** 0.84
- **MSE:** 17953.40

Among these models, Model 2 - Polynomial Regression with Ridge Regularization, performed the best, achieving the highest R-squared score of 0.85 and the lowest mean squared error of 4724.27. For further insights, refer to the scatterplot below, showcasing the predictions of Model 2 against the actual medical charges.

![output_29_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/5c9282a3-2bfc-40e6-b718-4f10f2e6fc42)


## 5. Interpretation:
 **Feature Importance Plot:**
In the feature importance plot below, we observe that 'smoker' has the highest feature importance among all the variables, indicating its significant influence on predicting medical charges.
![output_26_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/ec070a65-cde9-4b2a-9f4e-80822dfece65)


## 6. Insights and key findings:
1. Smoking is the most important factor in this dataset for the increase in health insurance charges. On average people who smoke have 3.8 times more insurance charges then people who don't.
2. We analyzed the outliers of 'charges' and found out that out of 139 only 3 people were non smokers, who had extreme high charges for insurance and that was because they had both high bmi and very old age(60 approximately).

## 7.Next Steps:
1. A better r2 score and mse could be achieved by using other regression models such as random forest or decision tree regression
2. A better prediction could be achieved if this dataset had other important features as well such as 'Disease'
 
