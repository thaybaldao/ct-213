from math import inf


def hill_climbing(cost_function, neighbors, theta0, epsilon, max_iterations):
    """
    Executes the Hill Climbing (HC) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param neighbors: function which returns the neighbors of a given point.
    :type neighbors: list of numpy.array.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    theta = theta0
    history = [theta0]
    iteration = 1
    theta_J = cost_function(theta)

    while (theta_J >= epsilon) and (iteration <= max_iterations):
        best = None
        best_J = inf
        for neighbor in neighbors(theta):
            neighbor_J = cost_function(neighbor)
            if neighbor_J < best_J:
                best = neighbor
                best_J = neighbor_J

        if best_J > theta_J:
            return theta, history

        theta = best
        theta_J = best_J

        history.append(theta)
        iteration += 1

    return theta, history
