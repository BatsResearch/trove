import copy
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

import pytrec_eval


def get_measures_max_cutoff(measures: Union[str, List[str], Set[str]]) -> int:
    """Returns Maximum cutoff value among several metrics.

    Args:
        measures (Union[str, List[str], Set[str]]): a collection of measures (e.g., ``mrr_cut_10,ndcg_cut_10``).

    Returns:
        maximum cutoff value among these measures.
    """
    measures = expand_and_combine_eval_measures(
        measures=copy.deepcopy(measures),
        additional_supported_measures="mrr_cut",
        reconstruct=False,
    )
    max_k = -1
    for k_values in measures.values():
        for k in k_values:
            if int(k) == float(k):
                num_k = int(k)
            else:
                num_k = int(k) + 1
            if num_k > max_k:
                max_k = num_k
    return max_k


def expand_and_combine_eval_measures(
    measures: Union[str, List[str], Set[str]],
    supported_measures: Optional[Union[str, List[str], Set[str]]] = None,
    additional_supported_measures: Optional[Union[str, List[str], Set[str]]] = None,
    reconstruct: bool = True,
) -> Union[Set[str], Dict[str, Set[str]]]:
    """Preprocess and combine all IR metric names into a standard format.

    Adapted combination of these two functions:

        * https://github.com/cvangysel/pytrec_eval/blob/0d8571efc2dea5a792358410d5affbe0ad383602/py/__init__.py#L66
        * https://github.com/cvangysel/pytrec_eval/blob/0d8571efc2dea5a792358410d5affbe0ad383602/py/__init__.py#L76

    Args:
        measures (Union[str, List[str], Set[str]]): IR metric names to process.
        supported_measures (Optional[Union[str, List[str], Set[str]]]): supported measure names.
            If not provided, use supported measure names from ``pytrec_eval``. I.e., ``pytrec_eval.supported_measures``.
        additional_supported_measures (Optional[Union[str, List[str], Set[str]]]): Additional
            metric names to include in ``supported_measures``.
        reconstruct (bool): If True, reconstruct processed metric names into ``meas.p1,p2,p3`` format.
            If False, return a mapping from base metric names to a set of cuttoff values.

    Returns:
        Standardized metric names either formatted as strings or a mapping from base metric name to cutoff values.
    """
    if isinstance(measures, str):
        measures = {measures}
    elif isinstance(measures, list):
        measures = set(measures)
    elif not isinstance(measures, set):
        msg = f"'measures' should of type `Union[List[str], Set[str], str]`. Got: '{type(measures)}'"
        raise TypeError(msg)

    if supported_measures is None:
        supported_measures = copy.deepcopy(pytrec_eval.supported_measures)
    elif isinstance(supported_measures, str):
        supported_measures = {supported_measures}
    elif isinstance(supported_measures, list):
        supported_measures = set(supported_measures)
    elif not isinstance(supported_measures, set):
        msg = f"'supported_measures' should of type `Optional[Union[List[str], Set[str], str]]`. Got: '{type(supported_measures)}'"
        raise TypeError(msg)

    supported_nicknames = copy.deepcopy(pytrec_eval.supported_nicknames)

    if isinstance(additional_supported_measures, str):
        additional_supported_measures = {additional_supported_measures}

    if isinstance(additional_supported_measures, (list, set)):
        supported_measures = supported_measures.union(additional_supported_measures)
    elif additional_supported_measures is not None:
        msg = (
            f"'additional_supported_measures' should of type `Optional[Union[List[str], Set[str], str]]`."
            f" Got: '{type(additional_supported_measures)}'"
        )
        raise TypeError(msg)

    # from here: https://github.com/cvangysel/pytrec_eval/blob/0d8571efc2dea5a792358410d5affbe0ad383602/py/__init__.py#L66
    # Expand nicknames (e.g., official, all_trec)
    result = set()
    for _measure in measures:
        if _measure in supported_nicknames:
            result.update(supported_nicknames[_measure])
        else:
            result.add(_measure)
    measures = result

    # from here: https://github.com/cvangysel/pytrec_eval/blob/0d8571efc2dea5a792358410d5affbe0ad383602/py/__init__.py#L76
    RE_BASE = r"{}[\._]([0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*)"

    # break apart measures in any of the following formats and combine
    #  1) meas          -> {meas: {}}  # either non-parameterized measure or use default params
    #  2) meas.p1       -> {meas: {p1}}
    #  3) meas_p1       -> {meas: {p1}}
    #  4) meas.p1,p2,p3 -> {meas: {p1, p2, p3}}
    #  5) meas_p1,p2,p3 -> {meas: {p1, p2, p3}}
    param_meas = defaultdict(set)
    for measure in measures:
        if measure not in supported_measures and measure not in supported_nicknames:
            matches = (
                (m, re.match(RE_BASE.format(re.escape(m)), measure))
                for m in supported_measures
            )
            match = next(filter(lambda x: x[1] is not None, matches), None)
            if match is None:
                raise ValueError("unsupported measure {}".format(measure))
            base_meas, meas_args = match[0], match[1].group(1)
            param_meas[base_meas].update(meas_args.split(","))
        elif measure not in param_meas:
            param_meas[measure] = set()

    if not reconstruct:
        return param_meas

    # re-construct in meas.p1,p2,p3 format for trec_eval
    fmt_meas = set()
    for meas, meas_args in param_meas.items():
        if meas_args:
            meas = "{}.{}".format(meas, ",".join(sorted(meas_args)))
        fmt_meas.add(meas)

    return fmt_meas
