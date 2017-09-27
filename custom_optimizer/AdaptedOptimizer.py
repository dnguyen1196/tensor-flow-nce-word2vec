from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util import nest

'''
NOTE: CustomBaseOptimizer is just like optimizer.Optimizer
 But I copied over to do testing/tinkering/ see how it works!
'''
from custom_optimizer.CustomBaseOptimizer import CustomBaseOptimizer

class AdaptedOptimizer(CustomBaseOptimizer):
    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def __init__(self, learning_rate, use_locking=False, name="AdaptedOptimizer"):
        """Construct a new gradient descent optimizer.
        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescent".
        """
        super(AdaptedOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate

    def _apply_dense(self, grad, var):
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op

    def _resource_apply_dense(self, grad, handle):
        return training_ops.resource_apply_gradient_descent(
            handle.handle, math_ops.cast(self._learning_rate_tensor,
                                         grad.dtype.base_dtype),
            grad, use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return resource_variable_ops.resource_scatter_add(
            handle.handle, indices, -grad * self._learning_rate)

    def _apply_sparse_duplicate_indices(self, grad, var):
        delta = ops.IndexedSlices(
            grad.values *
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.indices, grad.dense_shape)
        return var.scatter_sub(delta, use_locking=self._use_locking)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):

        # print ("heno");
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        print ("vars_with_grad <class>: ", vars_with_grad.__class__)
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        return self.apply_gradients(grads_and_vars, global_step=global_step,
                                    name=name)

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name="learning_rate")