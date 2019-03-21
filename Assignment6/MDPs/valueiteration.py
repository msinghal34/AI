class Mdp:
    def __init__(self, mdpfileName):
        file = open(mdpfileName, 'r')
        lines = file.read().splitlines()
        file.close()

        for line in lines:
            words = line.split()
            initialWord = words[0]
            if initialWord == "numStates":
                self.numStates = int(words[1])
            elif initialWord == "numActions":
                self.numActions = int(words[1])
                self.transitionList = [
                    [[] for _ in range(self.numActions)] for _ in range(self.numStates)]
            elif initialWord == "start":
                self.start = int(words[1])
            elif initialWord == "end":
                if words[1] == "-1":
                    self.end = []
                else:
                    self.end = [int(word) for word in words[1:]]
            elif initialWord == "discount":
                self.discount = float(words[1])
            elif initialWord == "transition":
                self.transitionList[int(words[1])][int(words[2])].append(
                    tuple([int(words[2]), int(words[3]), float(words[4]), float(words[5])]))
            else:
                print("Error: Initial Word is wrong")
        # Removing all empty actions from each state
        for state, transitions in enumerate(self.transitionList):
            for action, actions in enumerate(transitions):
                if (actions == []):
                    del self.transitionList[state][action]

    def printMdp(self):
        print(self.numStates)
        print(self.numActions)
        print(self.start)
        print(self.end)
        print(self.transitionList)
        print(self.discount)

    def valueIteration(self):
        old_values = [0.0]*self.numStates
        action_values = [-1]*self.numStates
        num_iters = 0
        condition = True
        while (condition):
            condition = False
            new_values = [0.0]*self.numStates
            # For each state
            for state in range(self.numStates):
                # If state is not an end state
                if (self.end.count(state) == 0):
                    state_transitions = self.transitionList[state]
                    sum_list = []
                    # For all possible actions from this state
                    for actions in state_transitions:
                        a_sum = 0.0
                        a_action = actions[0][0]
                        for action in actions:
                            a_sum += action[3]*(action[2] +
                                                self.discount*old_values[action[1]])
                        sum_list.append((a_sum, a_action))
                    # Update the maximum value of the state and corresponding action
                    new_values[state], action_values[state] = max(sum_list)
                    # Updating the condition
                    if (abs(new_values[state] - old_values[state]) > 1e-16):
                        condition = True
            num_iters += 1
            old_values = new_values
        for value, action in zip(old_values, action_values):
            print(value, action)
        print("iterations", num_iters)
        return


string = input("Enter file number: ")
mdp = Mdp("data/mdp/mdpfile" + string + ".txt")
# mdp.printMdp()
# input("Press Enter to calculate optimal policy")
mdp.valueIteration()
