import numpy as np
import itertools

class Attacker:



    def __init__(self, environment, learning = False, learning_algorithm = "none", exploration= "probabilistic"):

        self.current_node = environment.start_node
        self.environment = environment
        self.learning = learning
        self.learning_algorithm = learning_algorithm
        self.exploration = exploration
        self.rng = np.random.default_rng()
        self.returns = {}

        return

    def reset(self):
        self.current_node = self.environment.start_node
        self.Q_table = np.zeros((self.environment.num_of_nodes-1, self.environment.num_of_nodes, self.environment.values_per_node))


    def select_random_action(self, environment):
        neighbours = environment.neighbours_matrix[self.current_node]
        actions = itertools.product(neighbours,range(environment.values_per_node))
        actions = [x for x in actions if x not in environment.attacker_nodes_with_maximal_value]
        node, attack_type = self.rng.choice(actions)
        return (node,attack_type)

    def select_probabilistic_action(self, environment):
        neighbours = environment.neighbours_matrix[self.current_node]
        available_nodes = [node for node in neighbours if node not in environment.attacker_nodes_with_maximal_value]
        defense_value = environment.defence_values[:, :environment.values_per_node]
        # Check if all attack_values for available_nodes are zeros
        if any(np.all(defense_value[node] == 0) for node in available_nodes):
            # Use uniform distribution if all attack_values are zeros
            selected_node = self.rng.choice(available_nodes)
        else:
            # Calculate probabilities for each available node based on the inverse of attack strength
            node_probabilities = [1.0 / max(1, sum(defense_value[node])) for node in available_nodes]
            total_inverse_strength = sum(node_probabilities)
            node_probabilities = [p / total_inverse_strength for p in node_probabilities]  # Normalize probabilities

            # Choose a node to compromise probabilistically
            selected_node = self.rng.choice(available_nodes, p=node_probabilities)

        # Calculate probabilities for each attack type based on the inverse of attack strength
        attack_strength = defense_value[selected_node]

        # Check if the attacker_strength vector is a zero vector
        if np.all(attack_strength == 0) or np.all(attack_strength == 10):
            # Use uniform distribution for attack type if attacker_strength is zero
            attack_type_probabilities = [1.0 / environment.values_per_node for _ in range(environment.values_per_node)]
        else:
            # Assign probabilities for each attack type based on higher values having lower probability
            attack_type_probabilities = [10-attack_strength[i]  for i in range(environment.values_per_node)]

            # Normalize probabilities to ensure they sum to 1
        total_probability = sum(attack_type_probabilities)
        attack_type_probabilities = [p / total_probability for p in attack_type_probabilities]

        # Choose an attack type to increase strength probabilistically
        selected_attack_type = self.rng.choice(range(environment.values_per_node), p=attack_type_probabilities)

        return (selected_node, selected_attack_type)




    def select_action(self, environment, episode_number):
        if self.learning:
            if self.exploration == "probabilistic":
                return self.select_probabilistic_action(environment)
        else:
            return self.select_random_action(environment)

    def update_position(self,node):
        self.current_node = node









