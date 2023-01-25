import matplotlib.pyplot as plt
import mpmath

# The higher the decay factor, the less aggressively time is penalized. Decay factor = time at which a positive net trade will generate a negative reward.

class RewardModel():
    def __init__(self, price_list:list) -> None:
        self.price_list = price_list
        self.reward_lists = []

    def decayOne(self, decay_factor, minimum_roi):
        for net in self.net_list:
            reward_list = []
            for time in self.times_list:
                if net < 0:
                    reward = (-net - (-net * time) / decay_factor) + net*2 - minimum_roi
                else:
                    reward = -net * (mpmath.acot(time - decay_factor) - mpmath.pi / 2)
                reward_list.append(reward)

            self.reward_lists.append(reward_list)
        print(self.reward_lists)
    
    def plotRewards(self):
        fig, ax = plt.subplots()
        for i, reward_list in enumerate(self.reward_lists):
            ax.plot(self.times_list, reward_list, label='Net: ' + str(self.net_list[i]))
        legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
        plt.grid()

        plt.show()
    
new = RewardModel([5, 10, 30, 50, 75, 100, 125, 150, 200, 300, 390, 400, 500], [-0.1,-0.05, 0.01, 0.05, 0.1, 0.15, 0.20, 0.5])
new.decayOne(390, 0)
new.plotRewards()