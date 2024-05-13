# Machine Learning 382 - GroupF CYO Project
## **Overview:**
This project is a comprehensive study aimed at **building a predictive model to assess the creditworthiness of individuals or businesses**. The purpose of this project is to predict the risk of default on loans or credit lines, which is a critical aspect of financial risk assessment.

The project utilizes a dataset titled “Credit Risk Assessment” authored by Urvish Vekariya and sourced from Kaggle. The dataset will undergo a series of data analysis and preprocessing steps, including dataset analysis, univariate and bi-variate analysis, handling missing values, removing duplicates, and outlier value handling. Two models will be built and their predictions will be evaluated. The feature importance from each model will be analyzed and the models will be saved as pickle files for future use. The second model will undergo additional cross-validation to ensure its robustness.

Finally, the validated model will be deployed as a server in a web application using DASH, providing a practical interface for credit risk assessment. This project is a significant step towards leveraging machine learning for effective and efficient credit risk management. It aims to provide a reliable tool for financial institutions to make informed decisions regarding loan approvals and credit line extensions.


## **Hypothesis:**
_Hypothesis 1:_ Younger individuals are more likely to get their loans approved than older individuals
Justification: There are a larger amount of people who are in the age range of 22 to 24 years old, and have a loan approval rate of 66%. 

_Hypothesis 2:_ The attribute employment length does not significantly influence loan repayment behavior.
Justification: There is no significant difference in the loan status distribution among the employment length categories.

_Hypothesis 3:_ Individuals who own their homes or have a mortgage are more likely to have their loans approved 
Justification: Individuals who own their homes (OWN category) or have a mortgage (MORTGAGE category) have a significantly higher rate of loan approval compared to those in other categories.

_Hypothesis 4:_ Loans intended for debt consolidation, education, and home improvement have a higher approval rate compared to loans intended for medical and personal reasons
Justification: The likelihood of a loan being approved is higher than it being rejected, regardless of the loan intent category.

_Hypothesis 5:_ Having a credit history is beneficial for loan approval, the length of the credit history might not significantly impact the approval rate.
Justification: The likelihood of a loan being approved is higher than it being rejected, regardless of the length of the credit history. However, longer credit histories might lead to a slightly higher rejection rate due to the potential inclusion of more negative events.