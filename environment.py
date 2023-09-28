import numpy as np


class Environment:
    def __init__(self,num_of_nodes,values_per_node,neighbours_matrix,detection_probability,data_node,start_node = 0,initial_security = 2):
        
        self.neighbours_matrix = neighbours_matrix
        self.detection_probability = detection_probability
        self.start_node = start_node
        self.data_node = data_node
        self.num_of_nodes = num_of_nodes
        self.values_per_node = values_per_node
        self.attacker_detected = False
        self.data_node_cracked = False
        self.attacker_attack_successful = False
        self.attacker_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value_dqn = []
        self.attacker_maxed_all_values = False
        self.defender_maxed_all_values = False

        self.initial_security = initial_security
        self.attack_values = np.zeros((self.num_of_nodes,self.values_per_node))
        self.defence_values = np.zeros((self.num_of_nodes,self.values_per_node + 1)) + initial_security # +1 because there is the detection probability

        # set security flaws of the network
        self.defence_values[0][0] = 0
        self.defence_values[1][1] = 0
        self.defence_values[2][1] = 0
        self.defence_values[3][0] = 0
        self.defence_values[4][1] = 0
        self.defence_values[5][2] = 0

        for i in range(1,num_of_nodes):
            self.defence_values[i][-1] = self.detection_probability # the last value is the detection probability

        self.defence_values[start_node]-=self.initial_security 
        self.defence_values[start_node][-1] = 0

    def reset(self):
        self.attacker_detected = False
        self.data_node_cracked = False
        self.attacker_attack_successful = False

        self.attacker_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value_dqn = []
        self.attacker_maxed_all_values = False
        self.defender_maxed_all_values = False

        self.attack_values = np.zeros((self.num_of_nodes,self.values_per_node))
        self.defence_values = np.zeros((self.num_of_nodes,self.values_per_node + 1))+self.initial_security

        self.defence_values[0][0] = 0
        self.defence_values[1][1] = 0
        self.defence_values[2][1] = 0
        self.defence_values[3][0] = 0
        self.defence_values[4][1] = 0
        self.defence_values[5][2] = 0

        for i in range(1, self.num_of_nodes):
            self.defence_values[i][-1] = self.detection_probability  # the last value is the detection probability


    def do_defender_action(self,node,defence_type):
        if self.defence_values[node][defence_type] == 10:
            self.defender_nodes_with_maximal_value.append((node, defence_type))
        else:
            self.defence_values[node][defence_type] += 1
        if len(self.defender_nodes_with_maximal_value) == self.num_of_nodes*(self.values_per_node+1):
            self.defender_maxed_all_values = True
        return

    def do_attacker_action(self,node,attack_type):
        self.attacker_attack_successful = False
        if self.attack_values[node][attack_type] == 10:
            self.attacker_nodes_with_maximal_value.append((node, attack_type))
        else:
            self.attack_values[node][attack_type] += 1

        if len(self.attacker_nodes_with_maximal_value) == self.num_of_nodes*self.values_per_node:
            self.attacker_maxed_all_values = True
        
        if self.attack_values[node][attack_type] > self.defence_values[node][attack_type]: #successful attack
            if node == self.data_node:
                self.data_node_cracked = True
                return (100,-100)
            else:
                self.attacker_attack_successful = True
                return (0,0)
        else:
            if np.random.default_rng().random()*10 < self.defence_values[node][-1]: #detected?
                self.attacker_detected = True
                return (-100,100)
            return (0, 0)
    

    def termination_condition(self):
        if self.data_node_cracked or self.attacker_detected:
            return True
        if self.attacker_maxed_all_values or self.defender_maxed_all_values:
            return True
        return False
    

    
