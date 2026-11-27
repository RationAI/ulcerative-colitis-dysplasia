"""Cohort dataset readers behind one interface.

preprocessing/create_dataset.py dispatches on config.dataset.cohort to the
matching reader module. Each reader exposes create_dataset(config) ->
(dataset_df indexed by slide_id, missing items); downstream steps only ever
see the shared dataset.csv contract (case_id + slide_path, plus cohort
specific columns), never the reader itself.
"""

from . import huncrc, ikem

#: config.dataset.cohort -> reader module
READERS = {
    "ikem": ikem,
    "huncrc": huncrc,
}


def get_reader(cohort: str):
    try:
        return READERS[cohort]
    except KeyError:
        known = ", ".join(sorted(READERS))
        raise ValueError(
            f"Unknown dataset.cohort {cohort!r}; known cohorts: {known}"
        ) from None
