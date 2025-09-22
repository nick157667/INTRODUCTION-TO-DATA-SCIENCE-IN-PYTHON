# INTRODUCTION-TO-DATA-SCIENCE-IN-PYTHON
## TASK1 Data Processing
- TODO 1: Read the data and Fill your dataset
  - Used pandas.read_csv() to load the dataset from a TSV file
  - Used df.info() to display column information and non-null counts
  - Used df.head() to preview the first few rows of data
  - Used df.isnull().sum() to count missing values in each column
  - Used df.dropna() to remove rows containing null values
  - Verified missing values post-cleaning using df_cleaned.isnull().sum()
  - Used df_cleaned.info() to confirm the updated structure
- TODO 2: Split the data into a Training and Testing set
  - Used sklearn.model_selection.train_test_split to split the dataset into training (80%) and testing (20%) subsets
<img width="1296" height="485" alt="image" src="https://github.com/user-attachments/assets/c52d1888-ae14-4222-aac7-b705e8683fed" />

- TODO 3: Extracting Basic Statistics
  - calculate_average_rating(data):Computes the mean star rating
  - calculate_verified_fraction(data):Calculates the fraction of verified purchases
  - calculate_total_users(data):Counts the total unique users
  - calculate_total_items(data):Counts the total unique products
  - calculate_five_star_fraction(data):Computes the proportion of 5-star reviews




## TASK2 Classification
- TODO 1: Define the feature function
  - Used data['review_body'].fillna('').apply(len).values to compute the length of each review, ensuring missing values were replaced with an empty string
  - Extracted data['star_rating'].values as a feature
  - Combined review length and star ratings into a single feature matrix using np.vstack((review_length, star_rating)).T
  - Converted the verified_purchase column to integers using data['verified_purchase'].astype(int).values
- TODO 2: Fit your model
  - Used LogisticRegression() to initialize the Logistic Regression model
  - Used model.fit(X_train, y_train) to train the Logistic Regression model on the training data
  - Used model.predict(X_test) to make predictions on the test data using the trained model
- TODO 3: Compute Accuracy of Your Model
  - Used accuracy_score(y_test, y_pred) to compute the model's accuracy
<img width="698" height="275" alt="image" src="https://github.com/user-attachments/assets/4d868b60-9686-4010-ac5d-bfaf3f62df01" />

- TODO 4: Finding the Balanced Error Rate
  - Calculated True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) using confusion_matrix
  - Computed Sensitivity (Recall for the positive class) using sensitivity = tp / (tp + fn)
  - Computed Specificity (Recall for the negative class) using specificity = tn / (tn + fp)
  - Calculated the Balanced Error Rate using ber = compute_ber(y_test, y_pred)




## TASK3 Regression

- TODO 1: Unique Words in a Sample Set
  - Without Stemming
    - Removed punctuation and converted text to lowercase
    - Counted occurrences of each word using collections.defaultdict



## TASK4 Recommendation Systems





