# COMP9417_GROUP_PROJECT

Multitask Machine Learning on Clinical Data

## Data preprocessing

### Introduction
This report covers the various data preprocessing steps that are performed so that the dataset becomes better in quality and suitability with respect to the tasks of analysis and modeling that will follow. All these are very critical for model accuracy, robustness, and model reliability.

#### 1. **Data Cleaning**
**Removing Unary Features**: Features that have a single unique value across the entire dataset are removed, as they contribute nil to the differentiation of models, but they cause added complexity to the fitting process.
**Delete Features and Samples with High Missing Rates**: It got rid of those features and samples that exhibited high missing data. This would dampen the noise and potential for overfitting, preparing the data for analysis more efficiently.
**Impact Comparison of Different Missing Rate Thresholds:** This covers the we explore the impacts of different missing thresholds on model training to come up with a proper strategy for data retention.

#### 2. **Feature Classification**
- **Continuous Features**:
- **Integers and Floats**: We differentiated integer and floating-point continuous feature data because, according to the type of feature, there was a need to implement data processing methods.
- **Categorical Features**:
**Binary and Multinoml:** Binary and multinomial categorical features were identified and processed properly to enable appropriate encoding and handling.

#### 3. **Multi-strategy Imputation**
- **Simple Imputation**: The process of handling missing values was initially done through statistical methods (mean, median, mode) suitable for simple or small datasets.
- **KNN Imputation**: KNN imputation assumes that similar instances have a close neighboring data point and was thus implemented for imputing the missing values in this analysis.
- **Regression-Based Imputation**:
- **Linear Regression**: Linear relationships were utilized to predict missing values.
- **Reverse Regression**: Target variables were uniquely used to predict and impute missing features.
- **Random Forest Regression**: Leveraged for its robust predictive power to estimate missing values.
- **Support Vector Regression (SVR)**: Applied to address complex non-linear relationships in data filling.
**Comparison of Imputation Techniques Effect on Training and Performance**: We went into a critical assessment of the training and performance effects of the different imputation techniques.

#### 4. **Outlier Management**
**Identification and Mitigation**: Outliers were identified statistically (IQR, Box plots) and reduced in their adverse impact on the models through trimming or replacing.

#### 5. **Feature Selection and Multi-task Modeling**
**Correlation Analysis**: In this kind of analysis, using statistical measures such as mutual information, each label's correlation with the features is to be calculated, so as to keep only the most predictive for a given specific task. Data Reconstruction: Optimization of subsets of features was made for each label according to the correlation analysis in an effort to construct features that can provide model precision in tasks. Independent modeling: each of these labels performs independent modeling to customize the performance for specific tasks that will improve the preciseness of multi-task models. ### Conclusion In specific, the outlined steps of data preprocessing improved not only the quality and reliability but also helped in making the models for different tasks more flexible and accurate, remaining solid grounds for deeper analysis and complex model development, which ensures efficiency and effectiveness in decision-making driven by data.
