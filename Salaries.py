import statsmodels.api as sm
import pandas as pd

def polynomial_regression(data, target_variable, degree, new_data):
    # Identify and handle non-numeric values
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values (NaN) after conversion
    data = data.dropna()

    # Separate independent variables and dependent variable
    X_train = data.drop(target_variable, axis=1)
    Y_train = data[target_variable]

    # Add polynomial terms and a constant term to the training data
    X_train_poly = sm.add_constant(X_train)
    for d in range(2, degree + 1):
        for col in X_train.columns:
            X_train_poly[col + f'_degree_{d}'] = X_train[col] ** d

    # Fit the model
    model = sm.OLS(Y_train, X_train_poly).fit()

    # Print the summary of the regression
    print(model.summary())

    # Now, let's use the trained model to make predictions on new data

    # Take input from the user for X1, X2, X3 
    x1 = float(input("Enter your Gender (for M ='1' & for F ='2') : "))
    x2 = float(input("Enter Your Year of Experience : "))
    x3 = float(input("Enter Your Country Code : "))

    # Create a DataFrame for user input
    user_input = pd.DataFrame({'X1': [x1], 'X2': [x2], 'X3': [x3]})

    # Add polynomial terms and a constant term to the new data
    new_data_poly = sm.add_constant(user_input)
    for d in range(2, degree + 1):
        for col in user_input.columns:
            new_data_poly[col + f'_degree_{d}'] = user_input[col] ** d

    # Ensure the order of columns is the same as in the training data
    new_data_poly = new_data_poly.reindex(columns=X_train_poly.columns, fill_value=0)

    # Predict Y for the new data
    Y_pred = model.predict(new_data_poly)

    # Print the predicted values
    print('This is Your Expected Salary(in $):', Y_pred.values)

# Example usage:
# Assume 'data' is your training dataset
data = pd.read_csv('./Salary.csv')

# Specify the target variable and the degree of the polynomial regression
target_variable = 'Y'
degree = 2

# Call the function with user input for new data
polynomial_regression(data, target_variable, degree, None)
