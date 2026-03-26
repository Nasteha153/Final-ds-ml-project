# Car Price Prediction – Reflection Paper

## 1. Introduction

The goal of this project was to build a machine learning system capable of predicting the selling price of used cars based on historical data. This problem is important in real life because both buyers and sellers often struggle to determine a fair price for a vehicle. An accurate prediction system can help reduce uncertainty and improve decision-making.

In this project, I developed a complete ML pipeline starting from data preprocessing to model training, evaluation, and deployment. The final system allows users to input car details and receive an estimated price through an API and a user-friendly interface.

---

## 2. Dataset

The dataset used in this project was obtained from Kaggle and contains over 1000 records of used cars.

### Features:

* Year (manufacturing year of the car)
* Kilometers Driven (total distance covered)
* Fuel Type (Petrol, Diesel, etc.)
* Seller Type (Individual or Dealer)
* Transmission (Manual or Automatic)
* Owner (number of previous owners)

### Target Variable:

* Selling Price

### Preprocessing Steps:

To prepare the dataset for machine learning, the following steps were applied:

* Converted the target variable (`selling_price`) into numeric format
* Handled missing values using median (for numerical) and mode (for categorical)
* Removed duplicate records
* Detected and capped outliers using the IQR method
* Created a new feature called `car_age` from the year column
* Applied one-hot encoding to categorical variables (fuel, seller type, transmission)
* Standardized numerical features using a scaler

These steps ensured that the dataset was clean, consistent, and suitable for model training.

---

## 3. Models

Two different machine learning algorithms were used in this project:

### 3.1 Linear Regression

Linear Regression is a simple algorithm that models the relationship between input features and the target variable using a straight-line equation. It assumes a linear relationship and is easy to interpret.

### 3.2 Random Forest Regressor

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to improve prediction accuracy. It can handle complex, non-linear relationships and is generally more robust than simple models.

### Why These Models?

* Linear Regression was chosen as a baseline model
* Random Forest was chosen for its higher accuracy and ability to capture complex patterns

---

## 4. Results

### Evaluation Metrics:

The models were evaluated using the following metrics:

* R² Score (explains variance)
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

### Performance:

* Linear Regression:

  * R² Score ≈ 0.84
* Random Forest:

  * R² Score ≈ 0.88

### Sanity Checks:

To validate the model, several manual checks were performed:

* Cars with higher year (newer cars) produced higher predicted prices ✔
* Cars with higher kilometers driven produced lower prices ✔
* Cars with multiple owners showed reduced value ✔

### Conclusion:

Random Forest performed better than Linear Regression because it can model non-linear relationships in the data.

---

## 5. Deployment

The trained models were deployed using a Flask API.

### How the API Works:

1. The user sends a JSON request containing car features
2. The API processes the input data
3. The model generates a prediction
4. The API returns the predicted price

### Endpoint:

```
POST /predict?model=rf
```

### Example Request:

```json
{
  "year": 2015,
  "km_driven": 50000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": 1
}
```

### Example Response:

```json
{
  "prediction": 455000
}
```

Additionally, a frontend interface was created using HTML, CSS, and JavaScript to allow users to interact with the system visually.

---

## 6. Lessons Learned

During this project, several important lessons were learned:

* Data preprocessing is critical for model performance
* Handling categorical data correctly is essential
* Simpler models are easier to interpret but may be less accurate
* Ensemble models like Random Forest provide better performance
* Integrating machine learning models with APIs is important for real-world applications
* Debugging errors (such as feature mismatch and deployment issues) is part of the development process

### Challenges Faced:

* Handling categorical encoding correctly for API input
* Fixing feature mismatch errors during prediction
* Deploying frontend and backend separately
* Understanding model evaluation metrics

### Key Takeaway:

Building a complete machine learning system requires not only training models but also preparing data, evaluating performance, deploying the model, and presenting results clearly.