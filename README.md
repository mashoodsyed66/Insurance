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

We're using this data to try and understand why healthcare costs vary so much and see if we can predict them. Our goal is to make healthcare more understandable and help people plan for medical expenses better.
<br> </br>

In the boxplots below, you'll notice some dots that are far away from the main bunch, these are called outliers. For **BMI (Body Mass Index)** , a few people have really high  BMIs compared to most others. Also In the **Charges** column, there are a lot of dots that stand out. This means some people have medical bills that are way higher than usual.
![output_8_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/c73d30c3-da14-40b0-88fc-b99c2df09ca5)


Sure, here's how you can include the provided code and the explanation in your README:

```markdown
## Handling Outliers

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

### Outliers Table

Upon further investigation of these outliers, we found that out of the 139 cases, only 3 were non-smokers. The remaining outliers with significantly higher charges were all smokers.

|   age | sex    |   bmi |   children | smoker   | region    |   charges |
|------:|:-------|------:|-----------:|:---------|:----------|----------:|
|    55 | female |  26.8 |          1 | no       | southwest | 35160.1   |
|    61 | female |  33.33|          4 | no       | southeast | 36580.3   |
|    59 | female |  34.8 |          2 | no       | southwest | 36910.6   |
```

This table highlights that while outliers in medical charges exist across various demographics, the majority of outliers with substantially higher charges are associated with smokers.
```

![output_11_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/b396dd61-2a6d-4312-bd2d-088787d883f5)
![output_12_1](https://github.com/mashoodsyed66/Insurance/assets/65015378/18c63fe6-c1fd-4cae-a2d4-4dcc8386bb85)
![output_25_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/113d015a-eff9-434f-9d0b-77ea7ff3ac43)
![output_26_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/ec070a65-cde9-4b2a-9f4e-80822dfece65)
![output_29_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/5c9282a3-2bfc-40e6-b718-4f10f2e6fc42)
![output_32_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/ac054931-76da-4880-ab3e-2a525e6d8457)
![output_35_0](https://github.com/mashoodsyed66/Insurance/assets/65015378/9129fd83-b517-4c28-bce0-218244d1f3af)
