# Guide to Reproduce Evaluation Results
For all results, there are two ways of reproducing them. Either RL training can be re-done from scratch and new results created (that should overall be similar to the results depicted in the thesis) or the raw data from the thesis experiments can be used to reproduce the same plots/figures as shown in the thesis.

## Use Raw Data From Thesis Experiments
1. Set `project_dir` variable in  `plotting/eval_result_paths.py`
2. Get the archived experiment data and copy it to the respective directories in `data`
3. Run the main methods in `plotting/plot_results.py` and `plotting/flowgen/rawdata_vis.py` with the section variables set to reproduce the most relevant evaluation plots from the data of the thesis experiments.

## Re-run Experiments and Produce New Plots
1. Set `project_dir` variable in  `plotting/eval_result_paths.py`
2. Change the hyperparameters of the Gin Config file of the agent you want to run, i.e., `{ddpg, dqn, ppo}.gin` to use the parameters you want.
3. Run `cd <project-root> && python -m training.runner --gin_file=<project-root>/data/configs/{ddpg, dqn, ppo}.gin` to launch the training process.
4. You can supervise the training process using TensorBoard, data for it is logged in `<TrainLoop.root_dir>/tensorflow`.
5. After training finished, change `plotting/eval_result_paths.py` to the paths of the new data and rawdata episodes.
6. Run `plotting/plot_results.py` and `plotting/flowgen/rawdata_vis.py` with the section variables set to reproduce the most relevant evaluation plots (or change their main methods to create new plots, see methods in the respective scripts).

### Notes for WOC Evaluation in 6.6.1
Since no training is done for the WOC evaluation, the procedure differs here.
1. In `sim/run_sim_without_train.py`, set the `base_dir` variable
2. Set `gin.bind_parameter('RulePerformanceTable.use_cache', False)`
3. Run `sim/run_sim_without_train.py`
4. Set `gin.bind_parameter('RulePerformanceTable.use_cache', True)`
5. Run `sim/run_sim_without_train.py`

This runs the simulation with the median mitigation parameters learned by the DQN-pthresh agent in scenario S1 for 100 adaptation steps, once with WOC disabled, and once with WOC enabled.
Note the directory names, wherein the results are stored, and adapt the paths in `plotting/eval_result_paths.py` accordingly. Then create the plots by running `plotting/plot_results.py` with `section_to_plot` set to `6.6.1`.