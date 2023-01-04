from functools import partial

import seqio
from t5.data import get_default_vocabulary

import cot_t5.preprocessors as preprocessors

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True),
    "targets": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True)
}

MAX_EXAMPLES_PER_DATASET = 500_000

ESNLI_TEMPLATES = {
    "esnli_using_only_the_above" : {
        "inputs_template": "{premise} Using only the above description and what you know about the world, \"{hypothesis}\" is definitely correct, incorrect, or inconclusive?",
           "targets_list": ["Correct", "Inconclusive", "Incorrect"],
        },
    "esnli_given_should_we_assume" : {
        "inputs_template": "Given {premise} Should we assume that \"{hypothesis}\" is true? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
        },
    "esnli_given_does_it_follow": {
        "inputs_template": "Given that {premise} Does it follow that {hypothesis} Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
        },
    "esnli_question": {
        "inputs_template": "{premise} Question: {hypothesis} True, False, or Neither?",
           "targets_list": ["True", "Neither", "False"]
    },
    "esnli_based_on_previous_passage": {
        "inputs_template": "{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
    "esnli_are_we_justified": {
        "inputs_template": "{premise} Are we justified in saying that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
    "esnli_take_the_following": {
        "inputs_template": "Take the following as truth: {premise} Then the following statement: \"{hypothesis}\" is true, false, or inconclusive?",
           "targets_list": ["True", "Inconclusive", "False"]
    },
    "esnli_given_that_therefore": {
        "inputs_template": "Given that {premise} Therefore, it must be true that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
    "esnli_suppose_can_we_infer": {
        "inputs_template": "Suppose {premise} Can we infer that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
    "esnli_assume_it_is_true": {
        "inputs_template": "Assume it is true that {premise} Therefore, \"{hypothesis}\" is guaranteed, possible, or impossible?",
           "targets_list": ["Guaranteed", "Possible", "Impossible"]
    },
    "esnli_suppose_its_true": {
        "inputs_template": "Suppose it's true that {premise} Then, is \"{hypothesis}\" always, sometimes, or never true?",
           "targets_list": ["Always", "Sometimes", "Never"]
    },
    "esnli_question_does_it_imply": {
        "inputs_template": "{premise} Question: Does this imply that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
    "esnli_keeping_in_mind": {
        "inputs_template": "{premise} Keeping in mind the above text, consider: {hypothesis} Is this always, sometimes, or never correct?",
           "targets_list": ["Always", "Sometimes", "Never"]
    },
    "esnli_based_on_that_info": {
        "inputs_template": "{premise} Based on that information, is the claim: \"{hypothesis}\" true, false, or inconclusive?",
           "targets_list": ["True", "Inconclusive", "False"]
    },
    "esnli_is_it_guaranteed": {
        "inputs_template": "Given {premise} Is it guaranteed true that \"{hypothesis}\"? Yes, no, or maybe?",
           "targets_list": ["Yes", "Maybe", "No"]
    },
}

for task_name, template_args in ESNLI_TEMPLATES.items():
    TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(tfds_name="esnli:0.1.0"),
        preprocessors=[
            partial(preprocessors.esnli_cot, **template_args),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[])
