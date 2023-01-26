import matplotlib.pyplot as plt
import mpmath

class RewardModel():
    def __init__(self, action:int, price_list:list) -> None:
        self.price_list = price_list
        self.action = action
        self.reward_list = []

    def decayOne(self, decay_factor, minimum_roi):
        for net in self.net_list:
            reward_list = []
            for time in self.times_list:
                if net < 0:
                    reward = (-net - (-net * time) / decay_factor) + net*2 - minimum_roi
                else:
                    reward = -net * (mpmath.acot(time - decay_factor) - mpmath.pi / 2)
                reward_list.append(reward)

    
    def plotRewards(self):
        fig, ax = plt.subplots()
        for i, reward_list in enumerate(self.reward_list):
            ax.plot(self.times_list, reward_list, label='Net: ' + str(self.net_list[i]))
        legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
        plt.grid()

        plt.show()