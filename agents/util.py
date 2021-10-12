import os


def get_dirs(root_dir, timestamp, alg_name):
    dirs = {}
    root_dir = os.path.expanduser(root_dir)
    root_dir = os.path.join(root_dir, alg_name + '_' + timestamp)
    dirs['root'] = root_dir
    tf_dir = os.path.join(root_dir, 'tensorflow')
    dirs['tf'] = tf_dir
    tf_train_dir = os.path.join(tf_dir, 'train')
    dirs['tf_train'] = tf_train_dir
    dirs['tf_eval'] = os.path.join(tf_dir, 'eval')
    dirs['chkpt'] = tf_train_dir
    dirs['collect_policy_chkpt'] = os.path.join(tf_train_dir, 'collect_policy')
    dirs['policy_chkpt'] = os.path.join(tf_train_dir, 'policy')
    return dirs
