import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = array_ops.shape(x)[1]
    ones_shape = array_ops.stack([cols, 1])
    ones = array_ops.ones(ones_shape, x.dtype)
    return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            partition_strategy="mod",
                            name=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """

    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "compute_sampled_logits",weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # as the name implies, this flatten into a row tensor

        sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=labels,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes)

        # sampled_values is a tuple of tensors (3 to be exact)
        # shown right the line below
        print("sampled_values: ", sampled_values)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        sampled, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in sampled_values)

        # pylint: enable=unpacking-non-sequence
        # cast a tensor to a new type
        sampled = math_ops.cast(sampled, dtypes.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        # But the num_true in our example is 1

        # Concatenate in the horizontal (0) direction, since both are row vectors
        all_ids = array_ops.concat([labels_flat, sampled], 0) # [num_true + num_negative_sampled, 1]

        # Retrieve the true weights and the logits of the sampled weights.
        # weights shape is [num_classes, dim]
        # This gets all the embeddings into 1 matrix?
        all_embeddings = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)

        print("all_embeddings: ", all_embeddings)
        # all_embeddings is the word embeddings for all words in 1 batch including true labels
        # and wrong labels so its dimension is [num true + num negative, embedding dimension]

        # true_embeddings shape is [batch_size * num_true, dim]
        true_embeddings = array_ops.slice(
            all_embeddings, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))

        print("true_embeddings: ", true_embeddings)
        sampled_embeddings = array_ops.slice(
            all_embeddings, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        print("sampled_embeddings: ", sampled_embeddings)
        # sampled embeddings should be [num_neg, embedding_dimension]

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]
        # So each entry will be vi'sj where vi is a center sample word and s is a sampled word
        # Have to transpose sampled_embeddings so that we get the right value
        sampled_logits = math_ops.matmul(inputs, sampled_embeddings, transpose_b=True)
        print("inputs: ", inputs) # [batch_size, dim]
        print("sampled_logits: ", sampled_logits) # [batch_size, negative_num_count]

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_biases = embedding_ops.embedding_lookup(biases, all_ids, partition_strategy=partition_strategy)

        # true_biases is a [batch_size * num_true] tensor (but num true is 1 by default)
        # sampled_biases is a [num_sampled] float tensor
        true_biases = array_ops.slice(all_biases, [0], array_ops.shape(labels_flat))
        sampled_biases = array_ops.slice(all_biases, array_ops.shape(labels_flat), [-1])

        # Slice is a multi-dimensional array cutting tool
        print("true_biases: ", true_biases)
        print("sampled_biases: ", sampled_biases)

        # TODO: this part does the matrix multiplication to find true w'x product
        # inputs shape is [batch_size, dim]
        # true_embeddings shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_embeddings)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)

        # TODO: why do we need different syntax to find true vs sampled logits?
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_embeddings, new_true_w_shape))

        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

        true_biases = array_ops.reshape(true_biases, [-1, num_true])

        # TODO: add the true and sampled logits with the respective biases
        # add the logits with the biases
        true_logits += true_biases
        sampled_logits += sampled_biases

        print("true_logits: ", true_logits)
        print("sampled_logis: ", sampled_logits)
        # Construct output logits and labels. The true labels/logits start at col 0.
        # This creates a column vector?
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        print("out_logits: ", out_logits)

        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        # Oh ok, think of out labels as the probability of appearing so 1/num_true for
        # correct context and 0 for sampled context
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

        print("out_labels: ", out_labels)

    return out_logits, out_labels


def _compute_logits_special(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            partition_strategy="mod",
                            name=None,
                            factor_matrix=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """

    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "compute_sampled_logits",weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # as the name implies, this flatten into a row tensor

        sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=labels,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes)

        # sampled_values is a tuple of tensors (3 to be exact)
        # shown right the line below
        print("sampled_values: ", sampled_values)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        sampled, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in sampled_values)

        # pylint: enable=unpacking-non-sequence
        # cast a tensor to a new type
        sampled = math_ops.cast(sampled, dtypes.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        # But the num_true in our example is 1

        # Concatenate in the horizontal (0) direction, since both are row vectors
        all_ids = array_ops.concat([labels_flat, sampled], 0) # [num_true + num_negative_sampled, 1]

        # Retrieve the true weights and the logits of the sampled weights.
        # weights shape is [num_classes, dim]
        # This gets all the embeddings into 1 matrix?
        all_embeddings = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)

        print("all_embeddings: ", all_embeddings)
        # all_embeddings is the word embeddings for all words in 1 batch including true labels
        # and wrong labels so its dimension is [num true + num negative, embedding dimension]

        # true_embeddings shape is [batch_size * num_true, dim]
        true_embeddings = array_ops.slice(
            all_embeddings, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))

        print("true_embeddings: ", true_embeddings)
        sampled_embeddings = array_ops.slice(
            all_embeddings, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        print("sampled_embeddings: ", sampled_embeddings)
        # sampled embeddings should be [num_neg, embedding_dimension]

        # inputs has shape [batch_size, dim]
        # sampled_embeddings has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]
        # So each entry will be vi'sj where vi is a center sample word and s is a sampled word
        # Have to transpose sampled_embeddings so that we get the right value
        # sampled_logits = math_ops.matmul(inputs, sampled_embeddings, transpose_b=True)

        # NOTE:
        #
        #
        #
        #       SAMPLED LOGITS HERE
        #
        #
        # TODO: replace the usual sampled_logits to include factor_matrix
        # so its x'Bv instead of the usual x'v
        print("factor_matrix: ", factor_matrix)
        matrix_vector_product = math_ops.matmul(factor_matrix, sampled_embeddings, transpose_b=True)
        sampled_logits = math_ops.matmul(inputs, matrix_vector_product)
        print("inputs: ", inputs) # [batch_size, dim]
        print("sampled_logits: ", sampled_logits) # [batch_size, negative_num_count]

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_biases = embedding_ops.embedding_lookup(biases, all_ids, partition_strategy=partition_strategy)

        # true_biases is a [batch_size * num_true] tensor (but num true is 1 by default)
        # sampled_biases is a [num_sampled] float tensor
        true_biases = array_ops.slice(all_biases, [0], array_ops.shape(labels_flat))
        sampled_biases = array_ops.slice(all_biases, array_ops.shape(labels_flat), [-1])

        # Slice is a multi-dimensional array cutting tool
        print("true_biases: ", true_biases)
        print("sampled_biases: ", sampled_biases)

        # TODO: this part does the matrix multiplication to find true w'x product
        # inputs shape is [batch_size, dim]
        # new_true_w_shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_embeddings)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)

        # CALCULATING the true word x word cross product here
        # TODO: why do we need different syntax to find true vs sampled logits?
        # Because we just want one column (batch_size, 1) of v'c where v is the word
        # vector and c is the context vector, so need some form of row wise
        # inner product
        # TODO: is multiply here doing the same work as in mathops.matmul??
        # row wise dot product is the element wise product of inputs and true embeddings
        # This supports broad casting, so what it probably means is multiplying X and B
        # where B is a matrix then it forms a column vector of inner product of rows of x
        # What is the point of expanding the dimension (so that there will be a dimension match?)

        # context_matrix = array_ops.reshape(true_embeddings, new_true_w_shape)
        context_matrix = true_embeddings
        input_word_matrix = array_ops.expand_dims(inputs, 1)

        print("context_matrix: ", context_matrix)
        print("input_word_matrix: ", input_word_matrix)

        B_vector = math_ops.matmul(factor_matrix, true_embeddings, transpose_b=True)
        row_wise_dots = math_ops.multiply(input_word_matrix, array_ops.transpose(B_vector))

        # row_wise_dots = math_ops.multiply(
        #     array_ops.expand_dims(inputs, 1),
        #     array_ops.reshape(true_embeddings, new_true_w_shape))
        # TODO: what exactly is tf doing here, how does it calculate the true logits
        # and why does so differently from sampled logits
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

        true_biases = array_ops.reshape(true_biases, [-1, num_true])

        # TODO: add the true and sampled logits with the respective biases
        # add the logits with the biases
        #
        true_logits += true_biases
        sampled_logits += sampled_biases

        print("true_logits: ", true_logits)
        print("sampled_logis: ", sampled_logits)
        # Construct output logits and labels. The true labels/logits start at col 0.
        # This creates a column vector?
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        print("out_logits: ", out_logits)

        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        # Oh ok, think of out labels as the probability of appearing so 1/num_true for
        # correct context and 0 for sampled context
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

        print("out_labels: ", out_labels)

    return out_logits, out_labels


def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
        _sentinel=None,
        labels=None,
        logits=None,
        name=None):
    """Computes sigmoid cross entropy given `logits`.

    Measures the probability error in discrete classification tasks in which each
    class is independent and not mutually exclusive.  For instance, one could
    perform multilabel classification where a picture can contain both an elephant
    and a dog at the same time.

    For brevity, let `x = logits`, `z = labels`.  The logistic loss is

          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

    For x < 0, to avoid overflow in exp(-x), we reformulate the above

          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

    Hence, to ensure stability and avoid overflow, the implementation uses this
    equivalent formulation

        max(x, 0) - x * z + log(1 + exp(-abs(x)))

    `logits` and `labels` must have the same type and shape.

    Args:
      _sentinel: Used to prevent positional parameters. Internal, do not use.
      labels: A `Tensor` of the same type and shape as `logits`.
      logits: A `Tensor` of type `float32` or `float64`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
      logistic losses.

    Raises:
      ValueError: If `logits` and `labels` do not have the same shape.
    """
    # pylint: disable=protected-access
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                             labels, logits)
    # pylint: enable=protected-access

    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        labels = ops.convert_to_tensor(labels, name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

        # The logistic loss formula from above is
        #   x - x * z + log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   -x * z + log(1 + exp(x))
        # Note that these two expressions can be combined into the following:
        #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # To allow computing gradients at zero, we define custom versions of max and
        # abs functions.
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)
        return math_ops.add(
            relu_logits - logits * labels,
            math_ops.log1p(math_ops.exp(neg_abs_logits)),
            name=name)

def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
            num_classes):
    """Computes and returns the noise-contrastive estimation training loss.

    See [Noise-contrastive estimation: A new estimation principle for
    unnormalized statistical
    models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
    Also see our [Candidate Sampling Algorithms
    Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

    A common use case is to use this method for training, and calculate the full
    sigmoid loss for evaluation or inference. In this case, you must set
    `partition_strategy="div"` for the two losses to be consistent, as in the
    following example:

    ```python
    if mode == "train":
      loss = tf.nn.nce_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...,
          partition_strategy="div")
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
      loss = tf.reduce_sum(loss, axis=1)
    ```

    Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
    so your labels must be sorted in order of decreasing frequency to achieve
    good results.  For more details, see
    @{tf.nn.log_uniform_candidate_sampler}.

    Note: In the case where `num_true` > 1, we assign to each target class
    the target probability 1 / `num_true` so that the target probabilities
    sum to 1 per-example.

    Note: It would be useful to allow a variable number of target classes per
    example.  We hope to provide this functionality in a future release.
    For now, if you have a variable number of target classes, you can pad them
    out to a constant number by either repeating them or by padding
    with an otherwise unused class.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          [num_classes, dim].  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  If set to
          `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
          learning to generate log-odds instead of log probabilities.  See
          our [Candidate Sampling Algorithms Reference]
          (https://www.tensorflow.org/extras/candidate_sampling.pdf).
          Default is False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).

    Returns:
      A `batch_size` 1-D tensor of per-example NCE losses.
    """

    # For the example using myWord2Vec, we only need the first 6 parameters
    logits, labels = _compute_sampled_logits(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes)

    sampled_losses = sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name="sampled_losses")

    print("nce_helper - nce_loss - sampled_losses: ", sampled_losses)
    # sampled_losses is batch_size x {true_loss, sampled_losses...}
    # We sum out true and sampled losses.
    # As we sum all the rows -> each row correspond to 1 value of the
    # nce loss function associated with just 1 word
    return _sum_rows(sampled_losses)

def nce_loss_special(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             factor_matrix):
    """Computes and returns the noise-contrastive estimation training loss.

    See [Noise-contrastive estimation: A new estimation principle for
    unnormalized statistical
    models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
    Also see our [Candidate Sampling Algorithms
    Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

    A common use case is to use this method for training, and calculate the full
    sigmoid loss for evaluation or inference. In this case, you must set
    `partition_strategy="div"` for the two losses to be consistent, as in the
    following example:

    ```python
    if mode == "train":
      loss = tf.nn.nce_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...,
          partition_strategy="div")
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
      loss = tf.reduce_sum(loss, axis=1)
    ```

    Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
    so your labels must be sorted in order of decreasing frequency to achieve
    good results.  For more details, see
    @{tf.nn.log_uniform_candidate_sampler}.

    Note: In the case where `num_true` > 1, we assign to each target class
    the target probability 1 / `num_true` so that the target probabilities
    sum to 1 per-example.

    Note: It would be useful to allow a variable number of target classes per
    example.  We hope to provide this functionality in a future release.
    For now, if you have a variable number of target classes, you can pad them
    out to a constant number by either repeating them or by padding
    with an otherwise unused class.

    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          [num_classes, dim].  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  If set to
          `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
          learning to generate log-odds instead of log probabilities.  See
          our [Candidate Sampling Algorithms Reference]
          (https://www.tensorflow.org/extras/candidate_sampling.pdf).
          Default is False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).

    Returns:
      A `batch_size` 1-D tensor of per-example NCE losses.
    """

    logits, labels = _compute_logits_special(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        factor_matrix=factor_matrix)

    sampled_losses = sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name="sampled_losses")

    print("nce_helper - nce_loss - sampled_losses: ", sampled_losses)
    # sampled_losses is batch_size x {true_loss, sampled_losses...}
    # We sum out true and sampled losses.
    # As we sum all the rows -> each row correspond to 1 value of the
    # nce loss function associated with just 1 word
    return _sum_rows(sampled_losses)