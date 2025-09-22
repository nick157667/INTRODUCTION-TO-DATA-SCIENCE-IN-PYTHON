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
  - calculate_average_rating(data)
    - Computes the mean star rating
  - calculate_verified_fraction(data)
    - Calculates the fraction of verified purchases
  - calculate_total_users(data)
    - Counts the total unique users
  - calculate_total_items(data)
    - Counts the total unique products
  - calculate_five_star_fraction(data)
    - Computes the proportion of 5-star reviews



## TASK2 Classification

## TASK3 Regression

## TASK4 Recommendation Systems





