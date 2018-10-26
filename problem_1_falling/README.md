# The falling objects challenge

I have tried implementing the DQN Algorithm in order to solve the problem, however the agent doesn't seem to learn a lot.
You need the DQN_Model.py, the QNetwork file and the DQNAgent folder to run the agent implementation.
I have tried multiple approaches. The first one was to train the agent with negative reward as the game was implemented. Meaning that anytime the agent collied with a falling object, a -1 reward was received. I have trained the agent for 1000 steps episodes, hoping to maximize the reward (get it as close to 0 as possible). This approached failed. After that I tried giving the egent reward +1 for every move that doesn't result in a collision with an object, and letting the episode be unitl the first collison occurs. I observed that the agent started to get epsiodes with increasing number of steps ( around 1000 steps before a collision would occur), but somehow, there were still 40-60 steps episodes, meaning that the agent didn't learn how to doge every object. 

## Running the tests

```
python test_agent.py -a <module_name>+<class_name>
```
Example:
```
python test_agent.py -a demo_agent+DemoAgent
```
