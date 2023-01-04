import seqio

import cot_t5.tasks
# import t0.seqio_cache.tasks


seqio.MixtureRegistry.add(
    "esnli_cot",
    [task for task in cot_t5.tasks.ESNLI_TEMPLATES],
    default_rate=lambda t: cot_t5.tasks.MAX_EXAMPLES_PER_DATASET//len(cot_t5.tasks.ESNLI_TEMPLATES),
)

# seqio.MixtureRegistry.add(
#     "t0_train+esnli_cot",
#     ["t0_train", "esnli_cot"],
#     # default_rate=lambda t: mixture_cap[t.name],
# )