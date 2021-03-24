from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()
        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.f = 0
        heapq.heappush(pq, (start.f, start))

        while len(pq) > 0:
            node = heapq.heappop(pq)[1]
            node.closed = True
            if node.i == goal_position[0] and node.j == goal_position[1]:
                return self.construct_path(node), node.f
            for successor in self.node_grid.get_successors(node.i, node.j):
                succ = self.node_grid.get_node(successor[0], successor[1])
                if not succ.closed:
                    edge_cost = self.cost_map.get_edge_cost(node.get_position(), succ.get_position())
                    if succ.f > node.f + edge_cost:
                        succ.f = node.f + edge_cost
                        succ.parent = node
                        heapq.heappush(pq, (succ.f, succ))
        return [], inf

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()
        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.f = start.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.f, start))

        while len(pq) > 0:
            node = heapq.heappop(pq)[1]
            node.closed = True
            for successor in self.node_grid.get_successors(node.i, node.j):
                succ = self.node_grid.get_node(successor[0], successor[1])
                if not succ.closed:
                    succ.parent = node
                    if succ.i == goal_position[0] and succ.j == goal_position[1]:
                        return self.construct_path(succ), succ.f
                    succ.f = succ.distance_to(goal_position[0], goal_position[1])
                    heapq.heappush(pq, (succ.f, succ))
        return [], inf

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()
        pq = []
        start = self.node_grid.get_node(start_position[0], start_position[1])
        start.g = 0
        start.f = start.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start.f, start))

        while len(pq) > 0:
            node = heapq.heappop(pq)[1]
            node.closed = True
            if node.i == goal_position[0] and node.j == goal_position[1]:
                return self.construct_path(node), node.f
            for successor in self.node_grid.get_successors(node.i, node.j):
                succ = self.node_grid.get_node(successor[0], successor[1])
                if not succ.closed:
                    edge_cost = self.cost_map.get_edge_cost(node.get_position(), succ.get_position())
                    h_cost = succ.distance_to(goal_position[0], goal_position[1])
                    if succ.f > node.g + edge_cost + h_cost:
                        succ.g = node.g + edge_cost
                        succ.f = succ.g + h_cost
                        succ.parent = node
                        heapq.heappush(pq, (succ.f, succ))
        return [], inf
