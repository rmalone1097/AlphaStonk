import matplotlib.pyplot as plt

# The higher the decay factor, the less aggressively time is penalized. Decay factor = time at which a positive net trade will generate a negative reward.

class RewardModel():
    def __init__(self, times_list:list, net_list:list) -> None:
        self.times_list = times_list
        self.net_list = net_list
        self.reward_lists = []

    def decayOne(self, decay_factor):
        for net in self.net_list:
            reward_list = []
            for time in self.times_list:
                if net < 0:
                    reward = (-net - (-net * time) / decay_factor) + net*2
                else:
                    reward = net - (net * time) / decay_factor
                reward_list.append(reward)

            self.reward_lists.append(reward_list)
    
    def plotRewards(self):
        fig, ax = plt.subplots()
        for i, reward_list in enumerate(self.reward_lists):
            ax.plot(self.times_list, reward_list, label='Net: ' + str(self.net_list[i]))
        legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
        plt.grid()

        plt.show()
    
new = RewardModel([5, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000], [-0.1,-0.05, 0.01, 0.05, 0.1, 0.15, 0.20])
new.decayOne(1000)
new.plotRewards()