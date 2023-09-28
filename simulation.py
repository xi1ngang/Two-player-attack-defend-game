import numpy as np
from attacker import Attacker
from defender import Defender
from environment import Environment
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Initialize lists to store data
detection_probability_data = []
mean_defender_win_percentage_data = []
lower_bound_data = []
upper_bound_data = []

iterations = 1
steps_to_update_target_model = 1

def episode(environment,attacker,defender,episode_number):

    environment.reset()
    attacker.reset()
    defender.reset()

    attacker_sas = []
    defender_sas = []

    global iterations
    global steps_to_update_target_model
    iterations = 0
    # steps_to_update_target_model = 0
    while True:
        defender_action = defender.select_action(environment, episode_number)
        attacker_action = attacker.select_action(environment, episode_number)

        environment.do_defender_action(*defender_action)
        if defender.learning and defender.learning_algorithm == "dqn":
            defender_current_state = environment.defence_values

        attacker_current_reward, defender_current_reward = environment.do_attacker_action(*attacker_action)


        attacker_sas.append((attacker.current_node, attacker_action)) #, attacker_current_reward))
        defender_sas.append((0,defender_action)) #,defender_current_reward))

        if environment.attacker_attack_successful: #successful attack
            attacker.update_position(attacker_action[0])

        if environment.termination_condition():
            if defender.learning and defender.learning_algorithm == "dqn":
                defender_next_state = environment.defence_values
                defender.append_experience((defender_current_state, defender_action, defender_current_reward/100, defender_next_state, 1))
            break

        if defender.learning and defender.learning_algorithm == "dqn":
            defender_next_state = environment.defence_values
            defender.append_experience((defender_current_state, defender_action, defender_current_reward, defender_next_state, 0))

        if iterations % 10 == 0 and defender.learning_algorithm == "dqn":
            defender.dqn_training_step()

        # steps_to_update_target_model = steps_to_update_target_model + 1
        if steps_to_update_target_model  >= 100 and defender.learning_algorithm == "dqn":
            print('Copying main network weights to the target network weights')
            defender.target_model.set_weights(defender.model.get_weights())
            steps_to_update_target_model = 1

        iterations+=1
        steps_to_update_target_model+= 1


    attacker_final_reward = 0
    defender_final_reward = 0
    if environment.attacker_maxed_all_values or environment.defender_maxed_all_values:
        return (0,0)
    if environment.data_node_cracked:
        attacker_final_reward = 100
        defender_final_reward = -100
    elif environment.attacker_detected:
        attacker_final_reward = -100
        defender_final_reward = 100

    return attacker_final_reward, defender_final_reward, iterations

if __name__ == '__main__':
    # Define detection probabilities to simulate
    detection_probabilities = [2,4,6]
    # create subplots for step2win plot
    # fig, axs = plt.subplots(1,2,figsize=(10,4))

    neighbours_matrix = np.array([[1],
                           [2,3],
                           [3,4],
                           [4],
                           [5]], dtype=object)
    num_of_nodes = 6
    data_node = 5

    simulations = 3
    
    episodes = 200

    simulation_attacker_combinations = [True, None, "probabilistic"]
    simulation_defender_combinations = [True, "dqn", "epsilon_greedy"]

    # Create the shaded confidence region plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    for detection_probability in detection_probabilities:
        print("Detection Probability {}".format(detection_probability))
        print("Attacker combination: {}".format(simulation_attacker_combinations))
        print("Defender combination: {}".format(simulation_defender_combinations))

        attacker_win_percentages = []
        defender_win_percentages = []
        attacker_win_steps = []
        defender_win_steps = []
        defender_accumulated_rewards = []
        defender_rewards = []

        for sim in range(simulations):
            print("Simulation {}".format(sim))
            environment = Environment(num_of_nodes=num_of_nodes,values_per_node=3, neighbours_matrix = neighbours_matrix, detection_probability=detection_probability, data_node=data_node)
            attacker = Attacker(environment, learning = simulation_attacker_combinations[0], learning_algorithm = simulation_attacker_combinations[1], exploration= simulation_attacker_combinations[2])
            defender = Defender(environment, learning = simulation_defender_combinations[0], learning_algorithm = simulation_defender_combinations[1], exploration= simulation_defender_combinations[2])

            attacker_total_rewards = []
            defender_total_rewards = []
            steps_to_finish_game = []

            attacker_wins = 0
            defender_wins = 0
            ep = 0
            steps_to_update_target_model = 0
            defender_accumulated_reward = []
            defender_win_percentage = []
            defender_reward = []
            attacker_win = []
            defender_win = []
            defender_steps = []

            while ep < episodes:

                r_a,r_d,step = episode(environment, attacker, defender, ep)

                if r_a == 0: #tie
                    continue

                attacker_total_rewards.append(r_a)
                defender_total_rewards.append(r_d)
                steps_to_finish_game.append(step+1)
                if r_a > r_d:
                    attacker_wins+=1
                else:
                    defender_wins+=1

                defender_accumulated_reward.append(sum(defender_total_rewards))
                defender_win_percentage.append(100*defender_wins/(ep+1))
                defender_reward.append(r_d/100)

                ep += 1

            defender_accumulated_rewards.append(defender_accumulated_reward)
            defender_win_percentages.append(defender_win_percentage)
            defender_rewards.append(defender_reward)

            print("Attacker win percentage {}".format(attacker_wins/episodes))
            print("Defender win percentage {}".format(defender_wins/episodes))
                
            attacker_indices = np.where(np.array(attacker_total_rewards) == 100)[0]
            defender_indices = np.where(np.array(defender_total_rewards) == 100)[0]

            attacker_step = [steps_to_finish_game[i] for i in attacker_indices]
            defender_step = [steps_to_finish_game[i] for i in defender_indices]

            defender_steps.append(defender_step)

        mean_defender_wins = np.mean(defender_win_percentages, axis=0)
        std_defender_wins = np.std(defender_win_percentages, axis=0)

        # # step2win plot
        # axs[i].hist(defender_steps,bins = 8,  alpha = 0.7, rwidth=0.85)
        # axs[i].set_xlabel('Steps to win',fontsize=14)
        # axs[i].set_ylabel('Counts',fontsize=14)
        # axs[i].grid(axis='y', alpha = 0.75)


        # Calculate the upper and lower bounds for shading
        lower_bound = mean_defender_wins - std_defender_wins
        upper_bound = mean_defender_wins + std_defender_wins

        # Append data to the lists
        detection_probability_data.append(detection_probability)
        mean_defender_win_percentage_data.append(mean_defender_wins.tolist())
        lower_bound_data.append(lower_bound.tolist())
        upper_bound_data.append(upper_bound.tolist())

        # Fill the confidence region between the upper and lower bounds
        plt.fill_between(range(episodes), lower_bound, upper_bound, alpha=0.3, label=f'Initial Detection Probability {detection_probability*10}%')

        # Plot the mean values on top of the confidence region
        plt.plot(range(episodes), mean_defender_wins)

        end_time = time.time()
        running_time = end_time - start_time

        print("Running time: {:.2f} mins".format(running_time/60))


    # Add legend to the same figure
    plt.legend(fontsize=16)
    plt.xlabel('Episodes',fontsize=16)
    plt.ylabel('Defender Win Percentages',fontsize=16)
    plt.xticks(fontsize=14)  # Adjust the fontsize value for the x-axis tick labels
    plt.yticks(fontsize=14)  # Adjust the fontsize value for the y-axis tick labels

    plt.grid(True)
    # Adjust layout
    # plt.tight_layout()
    # Save the plot as an image file (optional)
    plt.savefig('defender_win_percentages.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('defender_steps.pdf',  dpi=300, bbox_inches='tight')

    # Show the final plot
    plt.show()


