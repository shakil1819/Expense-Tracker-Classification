# Peakflo - Take Home Task (AI/ML, Data Analyst, FDE 2026)

[accounts-bills.json](attachment:0c36f913-f959-418f-9705-07ea3c6e1578:accounts-bills.json)

## Overview

Welcome to the Peakflo Data Analyst take-home assignment! This task is designed to assess your ability to build a classification algorithm, analyze data, and communicate your approach effectively.

Time Estimate: 3-5 hours  
Submission Deadline: 5 days from receipt

---

## Business Context

At Peakflo, we process thousands of expense transactions monthly across different departments and categories. Properly categorizing these expenses into the correct accounting accounts is crucial for financial reporting, budgeting, and compliance.

Currently, our finance team manually reviews and categorizes many expenses, which is time-consuming and prone to inconsistency. We need an automated classification system that can predict the correct account for each expense based on available information.

---

## The Challenge

Build an algorithm that achieves at least 85% overall accuracy in classifying expenses to their correct account names.

Bonus points: our internal algorithm can solve it at 92% without overfitting. Try to beat our algorithm if you are up for a challenge.

### Dataset

You are provided with a JSON file (`accounts-bills.json`) containing 4,894 expense records from one of our client Singapore operations.

Dataset Structure:

- Total Records: 4,894 expenses
- Unique Vendors: 337
- Unique Account Categories: 97 account IDs, 103 account names
- Amount Range: -$15,195 to $161,838,000 SGD

Fields:
```json
{
  "_id": {"$oid": "..."},              
  "vendorId": "...",                    
  "itemName": "...",                    
  "itemDescription": "...",             
  "accountId": "...",                   
  "accountName": "...",                 
  "itemTotalAmount": 0.00               
}
```

Sample Account Categories:

- 611202 Online Subscription/Tool (1,179 records)
- 132098 IC Clearing account (706 records)
- 619203 Supplies/Expenses (225 records)
- 614123 Employee On Record (175 records)
- And 99 more categories...

---

## Requirements

### 1. Data Analysis & Exploration

- Analyze the dataset to understand patterns, distributions, and characteristics
- Identify potential challenges (class imbalance, data quality issues, etc.)
- Document key insights that informed your modeling approach

### 2. Algorithm Development

- Build a classification model that predicts `accountName` from available features
- Minimum Performance: 85% overall accuracy
- Use appropriate validation techniques to ensure reliable performance estimates
- Consider strategies for handling class imbalance and rare categories

### 3. Model Evaluation

- Report accuracy and other relevant metrics (precision, recall, F1-score)
- Analyze performance across different account categories
- Identify which categories your model handles well and which need improvement

### 4. Deliverables

Submit the following:

#### A. Code (required)

- Clean, well-commented code in Python, R, or your preferred language
- Include data preprocessing, feature engineering, model training, and evaluation
- Ensure code is reproducible (set random seeds where applicable)
- Organize code logically (separate files/sections for different components)

Acceptable formats:

- Jupyter Notebook (.ipynb) - PREFERRED
- Python script(s) (.py)
- R script(s) (.R) with R Markdown (.Rmd)

#### B. Written Description (required)

A document (PDF, Markdown, or included in notebook) covering:

1. Executive Summary (1 paragraph)
    - Key approach and main results
2. Data Analysis (1-2 pages)
    - Exploratory findings
    - Data quality observations
    - Key insights from initial analysis
3. Methodology (1-2 pages)
    - Algorithm choice and rationale
    - Feature engineering approach
    - Handling of class imbalance
    - Validation strategy
4. Results (1 page)
    - Performance metrics with validation approach
    - Breakdown by category performance (top/bottom performers)
    - Confusion matrix or error analysis (optional but encouraged)
5. Discussion (1 page)
    - Strengths and limitations of your approach
    - Ideas for improvement with more time/resources
    - Potential business considerations for deployment

Total Length: 4-6 pages (excluding code and visualizations)

---

## Evaluation Criteria

Your submission will be evaluated on:

1. Performance (40%)
    - Does the model meet the 85% accuracy threshold?
    - Are evaluation methods appropriate and rigorous?
    - Quality of performance metrics and analysis
2. Approach & Methodology (30%)
    - Soundness of the technical approach
    - Feature engineering creativity and effectiveness
    - Handling of data challenges (imbalance, missing values, etc.)
3. Code Quality (15%)
    - Clarity and organization
    - Documentation and comments
    - Reproducibility
    - Best practices
4. Communication (15%)
    - Clarity of written description
    - Quality of insights from data analysis
    - Critical thinking about limitations and improvements
    - Appropriate use of visualizations

---

## Tips for Success

1. Start with exploration: Understand the data before building models
2. Consider class imbalance: Some account categories have very few examples
3. Feature engineering matters: The text fields contain valuable signals
4. Validate properly: Use appropriate train/test splits or cross-validation
5. Think like an analyst: What would you tell stakeholders about deploying this model?
6. Don't over-engineer: A simple, well-executed approach often beats complex solutions
7. Document assumptions: Explain your decision-making process

---

## Submission Instructions

Please submit:

1. Your code (Jupyter Notebook preferred, or .py/.R files)
2. Written description (PDF/Markdown or included in notebook)
3. Any additional files needed to run your code (requirements.txt, etc.)
4. A brief README with instructions to run your code

Submit via: [Insert submission method - email/portal/etc.]

Questions? If you have clarifying questions about the assignment, please reach out to [contact email] within the first 48 hours.

---

## Notes

- You may use any libraries, frameworks, or pre-trained models you wish
- You may conduct external research and use online resources
- The dataset represents real business data patterns (anonymized)
- Focus on demonstrating your analytical thinking and technical skills
- Quality over quantity - we value clear, thoughtful work over exhaustive experimentation

---

Good luck! We look forward to reviewing your submission.