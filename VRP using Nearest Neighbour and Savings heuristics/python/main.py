import utility as utility
import loader as loader
import numpy as np


def main():

    # Paths to the data and solution files.
    vrp_file = "n32-k5.vrp"  # "data/n80-k10.vrp"
    sol_file = "n32-k5.sol"  # "data/n80-k10.sol"

    # Loading the VRP data file.
    px, py, demand, capacity, depot = loader.load_data(vrp_file)

    # Displaying to console the distance and visualizing the optimal VRP solution.
    vrp_best_sol = loader.load_solution(sol_file)
    best_distance = utility.calculate_total_distance(vrp_best_sol, px, py, depot)
    print("Best VRP Distance:", best_distance)
    utility.visualise_solution(vrp_best_sol, px, py, depot, "Optimal Solution")

    # Executing and visualizing the nearest neighbour VRP heuristic.
    # Uncomment it to do your assignment!

    nnh_solution = nearest_neighbour_heuristic(px, py, demand, capacity, depot)
    nnh_distance = utility.calculate_total_distance(nnh_solution, px, py, depot)
    print("Nearest Neighbour VRP Heuristic Distance:", nnh_distance)
    utility.visualise_solution(nnh_solution, px, py, depot, "Nearest Neighbour Heuristic")

    # Executing and visualizing the saving VRP heuristic.
    # Uncomment it to do your assignment!
    
    sh_solution = savings_heuristic(px, py, demand, capacity, depot)
    sh_distance = utility.calculate_total_distance(sh_solution, px, py, depot)
    print("Saving VRP Heuristic Distance:", sh_distance)
    utility.visualise_solution(sh_solution, px, py, depot, "Savings Heuristic")


def nearest_neighbour_heuristic(px, py, demand, capacity, depot):

    """
    Algorithm for the nearest neighbour heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    # TODO - Implement the Nearest Neighbour Heuristic to generate VRP solutions. - DONE

    # Keep track of visited nodes to make sure each node is only travelled to once
    num_nodes = len(px)
    visited = [False] * num_nodes
    visited[depot] = True

    # Initialize iwth empty set of routes, step 1
    routes = []

    while not all(visited):  # while not all nodes are visited, step 4
        current_route = []
        current_load = 0
        # Each Route starts from depot, step 1
        current_node = depot

        while True:
            next_node = None
            shortest_distance = float('inf')

            for i in range(num_nodes):
                if not visited[i] and current_load + demand[i] <= capacity:  # Nearest unvisited node that doesn't exceed capacity, one condition of step 2
                    #print(f'current node: {current_node} \n i: {i}')
                    distance = utility.calculate_euclidean_distance(px,py,current_node,i)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        next_node = i

            # If no feasible node is found current route is closed and new route started, step 3
            if next_node is None:
                break

            visited[next_node] = True
            current_route.append(next_node)
            current_load += demand[next_node]
            current_node = next_node

        routes.append(current_route)

    return routes



def savings_heuristic(px, py, demand, capacity, depot):

    """
    Algorithm for Implementing the savings heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    # TODO - Implement the Saving Heuristic to generate VRP solutions.

    num_nodes = len(px)
    savings = {}
    route_dict = {}
    route_loads = {}

    for i in range(num_nodes):
        if i != depot:
            route_dict[i] = [depot, i, depot]
            route_loads[i] = demand[i]


    # Compute and store savings for each possible merge
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != depot and j != depot:
                len1 = utility.calculate_euclidean_distance(px, py, i, depot)
                len2 = utility.calculate_euclidean_distance(px, py, depot, j)
                len3 = utility.calculate_euclidean_distance(px, py, i, j)
                savings[(i, j)] = len1 + len2 - len3

    sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)

    # Merge routes based on savings
    for (i, j), saving in sorted_savings:
        if i in route_dict and j in route_dict:
            route_i = route_dict[i]
            route_j = route_dict[j]

            if route_i and route_j and route_i != route_j:
                combined_demand = route_loads[route_i[1]] + route_loads[route_j[1]]

                if combined_demand <= capacity:
                    # Merge the routes
                    new_route = route_i[:-1] + route_j[1:]
                    new_load = combined_demand

                    # Update the route dictionary and load for each node in the new route
                    for node in new_route[1:-1]:
                        route_dict[node] = new_route
                        route_loads[node] = new_load

    # Get unique routes
    unique_routes = []
    seen_routes = set()

    for route in route_dict.values():
        route_tuple = tuple(route)
        if route_tuple not in seen_routes:
            unique_routes.append(route)
            seen_routes.add(route_tuple)

    return unique_routes


if __name__ == '__main__':
    main()
