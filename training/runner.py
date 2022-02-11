import gin
import tf_agents
from absl import app, flags, logging
import tensorflow
import dask
import pathlib
from gyms.hhh.env import register_hhh_gym
from training.train_loop import get_train_loop

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
FLAGS = flags.FLAGS


def main(_):
    """
    Entry point for RL training. Need to specify "gin_file" command line parameter that points to a valid Gin Config.
    (e.g. "<project-root>/data/configs/dqn.gin")
    """
    env_name = register_hhh_gym()

    logging.set_verbosity(logging.INFO)
    tensorflow.get_logger().setLevel('INFO')

    physical_devices = tensorflow.config.list_physical_devices('GPU')
    for gpu in physical_devices:  # don't allocate all available mem on start, but grow by demand
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    # register external functions/classes to use in gin config file
    # activations
    gin.external_configurable(tensorflow.keras.activations.relu, 'tf.keras.activations.relu')
    gin.external_configurable(tensorflow.keras.activations.tanh, 'tf.keras.activations.tanh')
    for gin_file in FLAGS.gin_file:
        gin.add_config_file_search_path(pathlib.Path(gin_file).parent)
        gin.parse_config_file(gin_file)

    gin.config._OPERATIVE_CONFIG_LOCK = dask.utils.SerializableLock()  # required for PPO when instantiating envs

    tf_agents.system.multiprocessing.enable_interactive_mode()

    get_train_loop(env_name=env_name).train()

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required('gin_file')
    app.run(main)
