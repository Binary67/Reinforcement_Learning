# Action Value Methods

**[1. E-Greedy Action Value Method](https://github.com/Binary67/Reinforcement_Learning/blob/master/Action%20Value%20Methods/1.%20E-Greedy%20Action%20Selection.py)**
> E-Greedy Action Value Method encourages agent to explore the search space. Initially, the agent will perform badly on searching for optimal action. However, the agent will eventually perform better as they continue to explore and improve their chance of recognizing optimal action.
    
**[2. Optimistic Initial Values](https://github.com/Binary67/Reinforcement_Learning/blob/master/Action%20Value%20Methods/2.%20Optimistic%20Initial%20Values%20Action%20Selection.py)**
> Optimistic Initial Values is another method to encourage exploration. Instead of setting the initial action value to zero, we initialize it to +5. So whichever actions are selected, the rewards will be less than starting estimates.The agent will continue explore for other actions until the estimates coverge.

**[3. Upper-Confidence-Bound Action Selection](https://github.com/Binary67/Reinforcement_Learning/blob/master/Action%20Value%20Methods/3.%20Upper%20Confidence%20Bound%20Action%20Selection.py)**
> Upper-Confidence-Bound Action Selection (UCB) helps to select action among the non-greedy actions (non-optimal action) according to their potential for actually being optimal. This action selection method take into account on how close the estimates for the action to be optimal and how uncertainty are those estimates.
