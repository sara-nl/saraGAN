import tensorflow as tf

class ExtendedEMA(tf.train.ExponentialMovingAverage):
    """
    This class expands on the tf.train.ExponentialMovingAverage class. 
    It allows to store a backup of the current (non-averaged) parameters in the session,
    in order to be able to swap out the training parameters with the averaged parameters.
    This can be used for e.g. computing validation/test metrics using the averaged parameters, rather than the training parameters.
    """

    def __init__(self, var_list, decay, num_updates=None, zero_debias=False, name='ExponentialMovingAverage'):
        """
        Transfers the weights from a tf.train.ExponentialMovingAverage to an active TensorFlow session.
        Arguments:
            var_list: the list of variables that you want to copy from the ema to the tensors in the active TF session
            others arguments: see official tf.train.ExponentialMovingAverage docs
        """

        super().__init__(decay, num_updates, zero_debias, name)

        self.var_list = var_list
        with tf.variable_scope('BackupVariables'):
            self.backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                                initializer=var.initialized_value())
                                for var in self.var_list]

    def assign_ema_weights(self):
        """
        Transfers the weights from a tf.train.ExponentialMovingAverage to an active TensorFlow session.
        Returns:
            An Operations that executes the assignments from the variables in the ExtendedEMA to the tensors in the active TF session
        """
        # Make sure that weights are backed up first
        with tf.control_dependencies([self._save_weight_backups()]):
            return tf.group([tf.assign(var, self.average(var)) for var in self.var_list])

    def _save_weight_backups(self):
        """
        Save the weights (listed in self.var_list) from the current model in the session to the self.backup_vars.
        Generally doesnt need to be called explicitely, since self.assign_ema_weights() already does.
        Returns:
            An Operation that executes assignment from the variables in the model beging trained to the self.backup_vars.
        """
        return tf.group([tf.assign(bck, var) for var, bck in zip(self.var_list, self.backup_vars)])

    def restore_original_weights(self):
        """
        Restore the weights (listed in self.var_list) from the self.backup_vars to the current model in the session.
        Returns:
            An Operation that executes assignment from the variables in self.backup_vars to the model being trained.
        """
        return tf.group([tf.assign(var, bck) for var, bck in zip(self.var_list, self.backup_vars)])

    def apply(self):
        """
        Overwrites the normal tf.train.ExponentialMovingAverage with a version that uses self.var_list as the variable list.
        """
        return super().apply(self.var_list)
    
