from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
from func_timeout import func_timeout, FunctionTimedOut
import time

SAFETY_MARGIN = 0.2


def get_robot_score(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    robot_pos = robot.position

    robot_livens = min(robot.battery, env.num_steps / 2)
    robot_score = robot.credit + robot_livens

    robot_package = robot.package
    if robot_package:
        package_destination = robot_package.destination
        package_position = robot_package.position
        distance_from_destination = manhattan_distance(robot_pos, package_destination)
        added_score = manhattan_distance(package_position, package_destination)
        robot_score = robot_score + added_score - distance_from_destination
    else:
        packages = env.packages
        min_distance_from_package = manhattan_distance(packages[0].position, robot_pos)
        if packages[1].on_board:
            min_distance_from_package = min(min_distance_from_package, manhattan_distance(packages[1].position, robot_pos))
        robot_score = robot_score - min_distance_from_package
    return robot_score


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot_score = get_robot_score(env, robot_id)
    other_robot_score = get_robot_score(env, 1 - robot_id)

    return robot_score - other_robot_score


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 4
    def minimaxL(self, env: WarehouseEnv, agent_id, agent_turn, depth):
        if env.done():
            credits = env.get_balances()
            return credits[agent_id] - credits[1 - agent_id], None
        elif depth == 0:
            return smart_heuristic(env, agent_id), None

        if agent_turn:
            operators, children = self.successors(env, agent_id)
            curMax = float('-inf')
            curOp = None
            for child, op in zip(children, operators):
                v, _ = self.minimaxL(child, agent_id, 1 - agent_turn, depth - 1)
                if curMax < v:
                    curMax = v
                    curOp = op
            return curMax, curOp
        else:  # other agent turn
            operators, children = self.successors(env, 1 - agent_id)
            curMin = float('inf')
            curOp = None
            for child, op in zip(children, operators):
                v, _ = self.minimaxL(child, agent_id, 1 - agent_turn, depth - 1)
                if curMin > v:
                    curMin = v
                    curOp = op
            return curMin, curOp

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        search_depth = 1
        result = None
        operators, children = self.successors(env, agent_id)
        default_op = operators[0]
        time_limit_with_SM = time_limit - SAFETY_MARGIN
        try:
            new_time_limit = time_limit_with_SM - (time.time() - start)
            while new_time_limit > 0:
                _, latest_result_op = func_timeout(new_time_limit, self.minimaxL, args=(env, agent_id, 1, search_depth))
                result = latest_result_op
                search_depth = search_depth + 1
                new_time_limit = time_limit_with_SM - (time.time() - start)
        except FunctionTimedOut:
            pass

        if result:
            return result
        return default_op


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def minimaxL_alpha_beta(self, env: WarehouseEnv, agent_id, agent_turn, depth, alpha, beta):
        if env.done():
            credits = env.get_balances()
            return credits[agent_id] - credits[1 - agent_id], None
        elif depth == 0:
            return smart_heuristic(env, agent_id), None

        elif agent_turn:
            operators, children = self.successors(env, agent_id)
            curMax = float('-inf')
            curOp = None
            for child, op in zip(children, operators):
                v, _ = self.minimaxL_alpha_beta(child, agent_id, 1 - agent_turn, depth - 1, alpha, beta)
                if curMax < v:
                    curMax = v
                    curOp = op
                alpha = max(curMax, alpha)
                if curMax >= beta:
                    return float('inf'), None
            return curMax, curOp
        else:  # other agent turn
            operators, children = self.successors(env, 1 - agent_id)
            curMin = float('inf')
            curOp = None
            for child, op in zip(children, operators):
                v, _ = self.minimaxL_alpha_beta(child, agent_id, 1 - agent_turn, depth - 1, alpha, beta)
                if curMin > v:
                    curMin = v
                    curOp = op
                beta = min(curMin, beta)
                if curMin <= alpha:
                    return float('-inf'), None
            return curMin, curOp

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        search_depth = 1
        result = None
        operators, children = self.successors(env, agent_id)
        default_op = operators[0]
        time_limit_with_SM = time_limit - SAFETY_MARGIN
        try:
            new_time_limit = time_limit_with_SM - (time.time() - start)
            while new_time_limit > 0:
                _, latest_result_op = func_timeout(new_time_limit, self.minimaxL_alpha_beta, args=(env, agent_id, 1, search_depth, float('-inf'), float('inf')))
                result = latest_result_op
                search_depth = search_depth + 1
                new_time_limit = time_limit_with_SM - (time.time() - start)
        except FunctionTimedOut:
            pass

        if result:
            return result
        return default_op


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def operators_probability(self, operators):
        denominator = len(operators)
        if 'move west' in operators:
            denominator += 2
        if 'pick up' in operators:
            denominator += 2

        probability = {}
        for i, operator in enumerate(operators):
            if operator == 'move west':
                probability[operator] = 3 / denominator
            elif operator == 'pick up':
                probability[operator] = 3 / denominator
            else:
                probability[operator] = 1 / denominator

        return probability

    def expectimaxL(self, env: WarehouseEnv, agent_id, agent_turn, depth):
        if env.done():
            credits = env.get_balances()
            return credits[agent_id] - credits[1 - agent_id], None
        elif depth == 0:
            return smart_heuristic(env, agent_id), None

        if agent_turn:
            operators, children = self.successors(env, agent_id)
            curMax = float('-inf')
            curOp = None
            for child, op in zip(children, operators):
                v, _ = self.expectimaxL(child, agent_id, 1 - agent_turn, depth - 1)
                if curMax < v:
                    curMax = v
                    curOp = op
            return curMax, curOp
        else:  # other agent turn
            operators, children = self.successors(env, 1 - agent_id)
            probability = self.operators_probability(operators)
            return_val = 0
            for child, op in zip(children, operators):
                v, _ = self.expectimaxL(child, agent_id, 1 - agent_turn, depth - 1)
                return_val += (v * probability[op])
            return return_val, None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        search_depth = 1
        result = None
        operators, children = self.successors(env, agent_id)
        default_op = operators[0]
        time_limit_with_SM = time_limit - SAFETY_MARGIN
        try:
            new_time_limit = time_limit_with_SM - (time.time() - start)
            while new_time_limit > 0:
                _, latest_result_op = func_timeout(new_time_limit, self.expectimaxL, args=(env, agent_id, 1, search_depth))
                result = latest_result_op
                search_depth = search_depth + 1
                new_time_limit = time_limit_with_SM - (time.time() - start)
        except FunctionTimedOut:
            pass

        if result:
            return result
        return default_op


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)