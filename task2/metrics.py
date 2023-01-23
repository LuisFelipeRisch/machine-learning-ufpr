import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import constants

def generate_graph_max_depth_behavior(X_train, X_test, y_train, y_test) -> None:
    """
    Explanation: Generate a graph to monitor de mean square error according to max depth. 
    The objective of this function is to visualize which max depth is better to the chosen model.  

    @X_train: The data to be given to the regressor to train
    @X_test: The data used to the regressor predict the target feature
    @y_train: The target feature data to be given to the regressor to train
    @y_test: The target feature data to be compared with the target feature data predicted by the regressor

    @return: The function returns nothing. It only generate a png in the root

    """
    max_depths = range(1, 20)

    testing_error = []
    for max_depth in max_depths:
        regressor = DecisionTreeRegressor(max_depth=max_depth)
        regressor.fit(X_train, y_train)
        testing_error.append(mean_squared_error(y_test, regressor.predict(X_test)))
    
    plt.plot(max_depths, testing_error, color='green', label='Testing error')
    plt.xlabel('Tree depth')
    plt.ylabel('Mean squared error')
    plt.legend()
    plt.savefig(constants.ERROR_GRAPH)

def generate_dispersal_graph(y_test, y_pred) -> None:
    """
    Explanation: Generate a dispersal graph where in x axis tells the real data and the y axis tells the predicted data by the regressor

    @y_test: The real data from the dataset
    @y_pred: The predicted data by the regressor

    @return: The function returns nothing. It only generate a png in the root

    """
    plt.scatter(y_test, y_pred)
    plt.xlabel('Amount of rain of the dataset')
    plt.ylabel('Amount of rain predicted by the regressor')
    plt.savefig(constants.DISPERSAL_GRAPH)
