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
    cols = array_ops.shape(x)[1]
    ones_shape = array_ops.stack([cols, 1])
    ones = array_ops.ones(ones_shape, x.dtype)
    return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_word_vector_product(nce_weights,
                                 biases,
                                 contexts,
                                 center_word_embeddings,
                                 num_sampled,
                                 num_classes,
                                 factor_matrix,
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
      nce_weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      contexts: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      center_word_embeddings: A `Tensor` of shape `[batch_size, dim]`.  The forward
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

    if isinstance(nce_weights, variables.PartitionedVariable):
        nce_weights = list(nce_weights)
    if not isinstance(nce_weights, list):
        nce_weights = [nce_weights]

    with ops.name_scope(name, "compute_sampled_logits", nce_weights + [biases, center_word_embeddings, contexts]):
        if contexts.dtype != dtypes.int64:
            contexts = math_ops.cast(contexts, dtypes.int64)
        context_row_vec = array_ops.reshape(contexts, [-1]) # Turn contexts vector into row

        # as the name implies, this flatten into a row tensor
        negative_samples_id = candidate_sampling_ops.log_uniform_candidate_sampler(
            true_classes=contexts,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes)

        negative_sampled_ids, true_expected_count, sampled_expected_count = (
            array_ops.stop_gradient(s) for s in negative_samples_id)

        negative_sampled_ids = math_ops.cast(negative_sampled_ids, dtypes.int64)

        # Concatenate in the horizontal (0) direction, since both are row vectors
        all_word_ids = array_ops.concat([context_row_vec, negative_sampled_ids], 0)
        print("all_word_ids: ", all_word_ids)

        # Retrieve all the word embeddings
        # Todo: find out how does embedding lookup work?
        all_word_embeddings = embedding_ops.embedding_lookup(
            nce_weights, all_word_ids, partition_strategy=partition_strategy)

        # Extract true embeddings and sampled_embeddings
        context_word_embeddings = array_ops.slice(
            all_word_embeddings, [0, 0], array_ops.stack([array_ops.shape(context_row_vec)[0], -1]))

        negative_sample_embeddings = array_ops.slice(
            all_word_embeddings, array_ops.stack([array_ops.shape(context_row_vec)[0], 0]), [-1, -1])


        # Compute word vector products between true words and sampled words
        # This is B * W' (
        # todo: check the math here
        factor_embedding_product = math_ops.matmul(factor_matrix, negative_sample_embeddings, transpose_b=True)
        negative_sample_cross_product = math_ops.matmul(center_word_embeddings, factor_embedding_product)

        # Retrieve all biases and then slice to get true and negative sampled biases
        all_biases = embedding_ops.embedding_lookup(biases, all_word_ids, partition_strategy=partition_strategy)
        true_biases = array_ops.slice(all_biases, [0], array_ops.shape(context_row_vec))
        negative_sample_biases = array_ops.slice(all_biases, array_ops.shape(context_row_vec), [-1])

        # TODO: this part does the matrix multiplication to find true w'x product
        # TODO: figure out the bug
        dim = array_ops.shape(context_word_embeddings)[1:2]

        # new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        context_matrix = context_word_embeddings # [num_labels, dim]

        # Todo: Why do we need to expand 1 more dimension here?
        # todo: why do we directly use inputs (it's an argument in the function and is just a vector of ints)
        # input_word_matrix = array_ops.expand_dims(center_word_embeddings, 1)
        input_word_matrix = center_word_embeddings

        print("context_matrix: ", context_matrix)
        print("input_word_matrix: ", input_word_matrix)

        # Multiply B with true embeddings
        factorized_center_word_embeddings = math_ops.matmul(factor_matrix, context_word_embeddings, transpose_b=True)

        # Since we arrange input word and context words in pairs
        # We need to do row wise cross products
        row_by_row_center_context_cross_products = math_ops.multiply(input_word_matrix, array_ops.transpose(factorized_center_word_embeddings))

        # Reshape the above cross products
        dots_as_matrix = array_ops.reshape(row_by_row_center_context_cross_products, array_ops.concat([[-1], dim], 0))
        true_cross_product = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

        print("true_cross_product: ", true_cross_product)
        # Add with biases
        # Reshap true biases into one column vector because num_true is 1?
        true_biases = array_ops.reshape(true_biases, [-1, num_true]) # This is just reshaping
        true_cross_product += true_biases
        negative_sample_cross_product += negative_sample_biases

        # This creates a column vector
        # Since all of these are column vectors
        out_cross_products = array_ops.concat([true_cross_product, negative_sample_cross_product], 1)

        # Return a label column vector
        out_labels = array_ops.concat([
            array_ops.ones_like(true_cross_product) / num_true,
            array_ops.zeros_like(negative_sample_cross_product)
        ], 1)

    return out_cross_products, out_labels


def compute_sigmoid_cross_entropy(  # pylint: disable=invalid-name
        _sentinel=None,
        labels=None,
        cross_products=None,
        name=None):
    # pylint: disable=protected-access
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                             labels, cross_products)
    # pylint: enable=protected-access
    with ops.name_scope(name, "logistic_loss", [cross_products, labels]) as name:
        cross_products = ops.convert_to_tensor(cross_products, name="logits")
        labels = ops.convert_to_tensor(labels, name="labels")
        try:
            labels.get_shape().merge_with(cross_products.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (cross_products.get_shape(), labels.get_shape()))

        # The logistic loss formula from above is
        #   x - x * z + log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   -x * z + log(1 + exp(x))
        # Note that these two expressions can be combined into the following:
        #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # To allow computing gradients at zero, we define custom versions of max and
        # abs functions.
        zeros = array_ops.zeros_like(cross_products, dtype=cross_products.dtype)
        cond = (cross_products >= zeros)
        relu_logits = array_ops.where(cond, cross_products, zeros)
        neg_abs_logits = array_ops.where(cond, -cross_products, cross_products)
        return math_ops.add(
            relu_logits - cross_products * labels,
            math_ops.log1p(math_ops.exp(neg_abs_logits)),
            name=name)


def nce_loss_multi_corpus(weights,
                          biases,
                          contexts_vectors,
                          center_word_embeddings,
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
      contexts_vectors: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.
      center_word_embeddings: A `Tensor` of shape `[batch_size, dim]`.  The forward
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
    cross_products, contexts = _compute_word_vector_product(
        nce_weights=weights,
        biases=biases,
        contexts=contexts_vectors,
        center_word_embeddings=center_word_embeddings,
        num_sampled=num_sampled,
        num_classes=num_classes,
        factor_matrix=factor_matrix)

    sampled_losses = compute_sigmoid_cross_entropy(
        labels=contexts, cross_products=cross_products, name="sampled_losses")

    return _sum_rows(sampled_losses)