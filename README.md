# Guide to Reproduce Evaluation Results
For all results, there are two ways of reproducing them. Either RL training can be re-done from scratch and new results created (that should overall be similar to the results depicted in the thesis) or the raw data from the thesis experiments can be used to reproduce the same plots/figures as shown in the thesis.

## 6.4 Adaptivity of DQN, DDPG, and PPO Agents


## 6.6.1 Worst-Offender Cache
Run `sim/run_sim_without_train.py` twice after modifiying two lines of code. First, set the `base_dir` variable. 
Then, for one run, set `gin.bind_parameter('RulePerformanceTable.use_cache', False)`, and for the other run, use `gin.bind_parameter('RulePerformanceTable.use_cache', True)`. 
This runs the simulation with the median mitigation parameters learned by the DQN-pthresh agent in scenario S1 for 100 adaptation steps, once with WOC disabled, and once with WOC enabled.
Note the directory names, wherein the results are stored, and adapt two lines
