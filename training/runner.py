import gin
from absl import app, flags, logging
import tensorflow as tf

from gyms.hhh.env import register_hhh_gym
from training.train_loop import get_train_loop

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
FLAGS = flags.FLAGS


def main(_):
    env_name = register_hhh_gym()
    logging.set_verbosity(logging.INFO)
    tf.get_logger().setLevel('INFO')
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:  # don't allocate all available mem on start, but grow by demand
        tf.config.experimental.set_memory_growth(gpu, True)
    for gin_file in FLAGS.gin_file:
        gin.parse_config_file(gin_file)
    get_train_loop(env_name=env_name).train()
    return 0


if __name__ == '__main__':
    try:
        flags.mark_flag_as_required('gin_file')
        app.run(main)
    except KeyboardInterrupt:
        pass
