from argparse import ArgumentParser
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sanitization import sanitize_data
from metrics import generate_dispersal_graph
# from metrics import generate_graph_max_depth_behavior


argument_parser = ArgumentParser()
argument_parser.add_argument("-d", "--day", required = True, type = int, help = "The day you want to estimate the amount of rain")

arguments = vars(argument_parser.parse_args())
day = arguments['day']

if not (day >= 0 and day <= 31):
    print("Day must be between 0 and 31!")
    sys.exit()

sanitized_data_df = sanitize_data(desired_day=day)

X = sanitized_data_df.drop(["TP_EST", "DATA", "DAY"], axis = 1).values
y = sanitized_data_df["TP_EST"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, shuffle=True)

# generate_graph_max_depth_behavior(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

regressor = DecisionTreeRegressor(max_depth=2) # Some tests that i made with the function generate_graph_max_depth_behavior told me that, in general, the best depth is always lower
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(f"The mean square error of my model is: {mean_squared_error(y_test, y_pred)}")
print(f"The mean absolute error of my model is: {mean_absolute_error(y_test, y_pred)}")

generate_dispersal_graph(y_test=y_test, y_pred=y_pred)





