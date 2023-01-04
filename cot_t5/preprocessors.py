import tensorflow as tf

def _explanation_targets(answer, explanations, prefix='The answer is'):
  # Add prefix before each explanation.
  return tf.strings.reduce_join(
      tf.concat([explanations, [answer]], axis=0),
      separator=' %s ' % prefix)


def esnli_cot(
    dataset,
    inputs_template='explain nli premise: {premise}, hypothesis: {hypothesis}',
    targets_list=['entailment', 'neutral', 'contradiction'],
    drop_explanations=False,
):
  """Convert the e-SNLI dataset to a text-to-text dataset.
  Args:
    dataset: a tf.data.Dataset to process.
    inputs_template: str, a template for the data sample.
    targets_list: verbalizer for each label.
    drop_explanations: bool, whether to drop the explanations from the target.
  Returns:
    a tf.data.Dataset
  """
  def my_fn(x):
    """Helper function to transform e-Snli dataset to inputs/targets."""

    h = "{hypothesis}"
    prefix, suffix = inputs_template.split("{premise}")
    if h in prefix:
        h_p, h_s = prefix.split(h)
        prefix = [h_p, x['hypothesis'], h_s]
        suffix = [suffix]
    else:
        h_p, h_s = suffix.split(h)
        prefix = [prefix]
        suffix = [h_p, x['hypothesis'], h_s]
    
    if not drop_explanations:
        prefix = ["Answer with explanation, "] + prefix

    inputs_string = tf.strings.join(
        prefix + [x['premise']] + suffix,
        separator='')

    class_label = tf.gather(targets_list, x['label'])

    if drop_explanations:
      targets_string = class_label
    else:
      explanations = [x.get('explanation_%d' % i, '') for i in range(1, 4)]
      explanations = tf.boolean_mask(
          explanations, tf.not_equal(explanations, ''))
      targets_string = _explanation_targets(class_label, explanations)

    return {'inputs': inputs_string, 'targets': targets_string}

  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
