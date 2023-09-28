import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import itertools

class Defender:
    def __init__(self, environment, learning = True, learning_algorithm = "dqn", exploration = "epsilon_greedy", algorithms_parameters = None):

        if algorithms_parameters == None:
            self.algorithms_parameters = {
                                        ("dqn", "epsilon_greedy"): {
                                            "eta" : 0.001,
                                            "gamma" : 0.95,
                                            "epsilon" : 0.05,
                                            "batch_size" : 15
                                        }
                                    }
        else:
            self.algorithms_parameters = algorithms_parameters

        self.model = None
        self.replay_buffer = None
        self.optimizer = None
        self.loss_fn = None
        self.Q_table = None
        self.returns = None


        self.environment = environment
        self.learning = learning
        self.learning_algorithm = learning_algorithm
        self.exploration = exploration
        self.rng = np.random.default_rng()

        if self.learning_algorithm == "dqn":
            self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(self.environment.num_of_nodes, self.environment.values_per_node+1)),
                keras.layers.Dense(6, activation="sigmoid"),
                keras.layers.Dense((self.environment.values_per_node+1)*self.environment.num_of_nodes)
            ])

            # Target network
            self.target_model = keras.Sequential([
                keras.layers.Flatten(input_shape=(self.environment.num_of_nodes, self.environment.values_per_node+1)),
                keras.layers.Dense(6, activation="sigmoid"),
                keras.layers.Dense((self.environment.values_per_node + 1) * self.environment.num_of_nodes)
            ])



            self.replay_buffer = deque(maxlen=2000)
            self.optimizer = keras.optimizers.Adam(learning_rate=self.algorithms_parameters[("dqn","epsilon_greedy")]["eta"])
            self.loss_fn = keras.losses.mean_squared_error
        else:
            self.Q_table = np.zeros((environment.num_of_nodes, environment.values_per_node + 1))
            self.returns = {}

        return

    def reset(self):
        return


    def select_random_action(self, environment): 
        actions = itertools.product(range(environment.num_of_nodes),range(environment.values_per_node+1))
        actions = [x for x in actions if x not in environment.defender_nodes_with_maximal_value]
        node, defence_type = self.rng.choice(actions)
        return (node,defence_type)

    def dqn_epsilon_greedy(self, environment, epsilon):
        x = self.rng.random()
        if x < epsilon:
            return self.select_random_action(environment)
        else:
            Q_values = self.model.predict(environment.defence_values[np.newaxis],verbose = 0)
            actions = [x for x in np.argsort(Q_values, axis=None) if x not in environment.defender_nodes_with_maximal_value_dqn]
            node, defence_type = divmod(actions[0], environment.values_per_node + 1)
            return (node, defence_type)


    def select_action(self, environment, episode):
        if self.learning:
            if self.learning_algorithm == "dqn":
                return self.dqn_epsilon_greedy(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["epsilon"])
            else:
                if episode < 5:
                    return self.select_random_action(environment) 
                elif self.exploration == "epsilon_greedy":
                    return self.select_action_epsilon_greedy(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["epsilon"])
                elif self.exploration == "softmax":
                    return self.select_action_softmax(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["tau"])
        else:
            return self.select_random_action(environment)

    def to_flatten_index(self, action):
        return action[0]*(self.environment.values_per_node + 1) + action[1]

    def append_experience(self, experience):
        self.replay_buffer.append(experience)
        return

    def sample_experiences(self, batch_size):
        indices = self.rng.integers(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def dqn_training_step(self):
        MIN_REPLAY_SIZE = 20
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return
        experiences = self.sample_experiences(self.algorithms_parameters[("dqn","epsilon_greedy")]["batch_size"])
        states, actions, rewards, next_states, dones = experiences
        actions = [self.to_flatten_index(action) for action in actions]
        next_Q_values = self.target_model.predict(next_states,verbose = 0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1-dones)*self.algorithms_parameters[("dqn","epsilon_greedy")]["gamma"]*max_next_Q_values)
        mask = tf.one_hot(actions, (self.environment.values_per_node +1 ) * self.environment.num_of_nodes)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values,Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return