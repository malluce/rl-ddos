import gin
import tf_agents
from absl import app, flags, logging
import tensorflow
import dask
from gyms.hhh.env import register_hhh_gym
from training.train_loop import get_train_loop

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
FLAGS = flags.FLAGS


def main(_):
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

    # sequential model and layers for CNN
    gin.external_configurable(tensorflow.keras.models.Sequential, 'tf.keras.models.Sequential')
    gin.external_configurable(tensorflow.keras.layers.Conv2D, 'tf.keras.layers.Conv2D')
    gin.external_configurable(tensorflow.keras.layers.MaxPool2D, 'tf.keras.layers.MaxPool2D')
    gin.external_configurable(tensorflow.keras.layers.Flatten, 'tf.keras.layers.Flatten')
    gin.external_configurable(tensorflow.keras.layers.Dense, 'tf.keras.layers.Dense')

    for gin_file in FLAGS.gin_file:
        gin.parse_config_file(gin_file)

    gin.config._OPERATIVE_CONFIG_LOCK = dask.utils.SerializableLock()  # required for PPO when instantiating envs

    tf_agents.system.multiprocessing.enable_interactive_mode()

    get_train_loop(env_name=env_name).train()

    return 0


if __name__ == '__main__':
    try:
        flags.mark_flag_as_required('gin_file')
        app.run(main)
    except KeyboardInterrupt:
        pass
