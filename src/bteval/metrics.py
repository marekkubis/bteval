import warnings


def aggregate_robustness(robust: int, non_robust: int, zero_division) -> float:
    """Aggregates robust and non-robust cases."""

    try:
        return robust / (robust + non_robust)
    except ZeroDivisionError:
        if zero_division == "warn":
            warnings.warn("Robustness score ill-defined and being set to 0.0")
            return 0.0

        return zero_division


def remove_const_text_samples(y_true, y_before, y_after, x_before, x_after):
    "Remove samples where reference and back transcribed texts are the same."

    if x_before is None or x_after is None:
        return y_true, y_before, y_after

    res_true = []
    res_before = []
    res_after = []

    for yt, yb, ya, xb, xa in zip(y_true, y_before, y_after, x_before, x_after):
        if xb != xa:
            res_true.append(yt)
            res_before.append(yb)
            res_after.append(ya)

    return res_true, res_before, res_after


def score_robustness(
    robust_case_func,
    non_robust_case_func,
    y_true,
    y_before,
    y_after,
    x_before,
    x_after,
    zero_division,
) -> float:
    """Scores robustness in accordance with robust_case_func and non_robust_case_func."""

    y_true, y_before, y_after = remove_const_text_samples(
        y_true, y_before, y_after, x_before, x_after
    )

    robust = 0
    non_robust = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if robust_case_func(t, b, a):
            robust += 1
        elif non_robust_case_func(t, b, a):
            non_robust += 1

    return aggregate_robustness(robust, non_robust, zero_division=zero_division)


def c_const_case(y_true, y_before, y_after) -> bool:
    return y_before == y_true and y_after == y_true


def i_const_case(y_true, y_before, y_after) -> bool:
    return y_before != y_true and y_after != y_true and y_before == y_after


def const_case(y_true, y_before, y_after) -> bool:
    return y_before == y_after


def c_to_i_case(y_true, y_before, y_after) -> bool:
    return y_before == y_true and y_after != y_true


def i_to_i_case(y_true, y_before, y_after) -> bool:
    return y_before != y_true and y_after != y_true and y_before != y_after


def i_to_c_case(y_true, y_before, y_after) -> bool:
    return y_before != y_true and y_after == y_true


def changed_case(y_true, y_before, y_after) -> bool:
    return y_before != y_after


def r1_robust_case(y_true, y_before, y_after) -> bool:
    return c_const_case(y_true, y_before, y_after)


def r1_non_robust_case(y_true, y_before, y_after) -> bool:
    return c_to_i_case(y_true, y_before, y_after)


def r1_irrelevant_case(y_true, y_before, y_after) -> bool:
    return (
        i_const_case(y_true, y_before, y_after)
        or i_to_i_case(y_true, y_before, y_after)
        or i_to_c_case(y_true, y_before, y_after)
    )


def r1_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_1$ score.

    $R_1$ tracks the volume of samples that become incorrect due to the use of an ASR system. It is
    suitable for monitoring the regressions of the ASR-NLU pair across consecutive revisions of the
    ASR model.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC

        non-robust cases: C->I

        irrelevant cases: constI, I->I, I->C
    """
    return score_robustness(
        r1_robust_case,
        r1_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def r13_robust_case(y_true, y_before, y_after) -> bool:
    return c_const_case(y_true, y_before, y_after)


def r13_non_robust_case(y_true, y_before, y_after) -> bool:
    return c_to_i_case(y_true, y_before, y_after) or i_to_c_case(
        y_true, y_before, y_after
    )


def r13_irrelevant_case(y_true, y_before, y_after) -> bool:
    return i_const_case(y_true, y_before, y_after) or i_to_i_case(
        y_true, y_before, y_after
    )


def r13_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_{13}$ score.

    $R_{13}$ penalizes positive changes which makes it a reasonable choice for tracking the
    robustness of NLU models that should act consistently in the presence of the input typed by the
    user and the input that comes from an ASR system. Contrary to $R_{123}$ it does not take into
    account the impact of I->I changes.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC

        non-robust cases: C->I, I->C

        irrelevant cases: constI, I->I
    """
    return score_robustness(
        r13_robust_case,
        r13_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def r13p_robust_case(y_true, y_before, y_after) -> bool:
    return c_const_case(y_true, y_before, y_after) or i_to_c_case(
        y_true, y_before, y_after
    )


def r13p_non_robust_case(y_true, y_before, y_after) -> bool:
    return c_to_i_case(y_true, y_before, y_after)


def r13p_irrelevant_case(y_true, y_before, y_after) -> bool:
    return i_const_case(y_true, y_before, y_after) or i_to_i_case(
        y_true, y_before, y_after
    )


def r13p_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_{13+}$ score.

    $R_{13+}$ is a counterpart of measuring the difference in accuracy for back-transcribed
    utterances and reference texts. This approach is sufficient for testing an NLU model in
    isolation, but it does not take into account that the behavior of downstream modules of a
    dialogue system that consume the outcome of an NLU model can deteriorate due to the change in
    labeling of incorrect results.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC, I->C

        non-robust cases: C->I

        irrelevant cases: constI, I->I
    """
    return score_robustness(
        r13p_robust_case,
        r13p_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def r12_robust_case(y_true, y_before, y_after) -> bool:
    return c_const_case(y_true, y_before, y_after) or i_const_case(
        y_true, y_before, y_after
    )


def r12_non_robust_case(y_true, y_before, y_after) -> bool:
    return c_to_i_case(y_true, y_before, y_after) or i_to_i_case(
        y_true, y_before, y_after
    )


def r12_irrelevant_case(y_true, y_before, y_after) -> bool:
    return i_to_c_case(y_true, y_before, y_after)


def r12_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_{12}$ score.

    $R_{12}$ penalizes changes between incorrect labels but neglects the impact of I->C changes. It
    is a rational choice for assessment of an NLU model that precedes a downstream module dedicated
    to correcting incorrect NLU outcomes such as a rule-based post-processor.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC, constI

        non-robust cases: C->I, I->I

        irrelevant cases: I->C
    """
    return score_robustness(
        r12_robust_case,
        r12_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def r123_robust_case(y_true, y_before, y_after) -> bool:
    return c_const_case(y_true, y_before, y_after) or i_const_case(
        y_true, y_before, y_after
    )


def r123_non_robust_case(y_true, y_before, y_after) -> bool:
    return (
        c_to_i_case(y_true, y_before, y_after)
        or i_to_i_case(y_true, y_before, y_after)
        or i_to_c_case(y_true, y_before, y_after)
    )


def r123_irrelevant_case(y_true, y_before, y_after) -> bool:
    return False


def r123_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_{123}$ score.

    $R_{123}$ penalizes changing incorrect outcomes to correct ones. It should be preferred to
    $R_{123+}$, if the downstream module relies on the outcome of NLU regardless of its status.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC, constI

        non-robust cases: C->I,  I->I, I->C

        irrelevant cases: -
    """
    return score_robustness(
        r123_robust_case,
        r123_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def r123p_robust_case(y_true, y_before, y_after) -> bool:
    return (
        c_const_case(y_true, y_before, y_after)
        or i_const_case(y_true, y_before, y_after)
        or i_to_c_case(y_true, y_before, y_after)
    )


def r123p_non_robust_case(y_true, y_before, y_after) -> bool:
    return c_to_i_case(y_true, y_before, y_after) or i_to_i_case(
        y_true, y_before, y_after
    )


def r123p_irrelevant_case(y_true, y_before, y_after) -> bool:
    return False


def r123p_score(
    y_true, y_before, y_after, x_before=None, x_after=None, zero_division="warn"
) -> float:
    """The $R_{123+}$ score.

    $R_{123+}$ promotes changing incorrect outcomes to correct ones, which is reasonable if we
    assume that the downstream module behaves correctly when presented with a correct input.

    Args:
        y_true: 1d array-like.
            The expected outcome of the NLU model (ground truth).

        y_before: 1d array-like.
            The outcome of the NLU model for the text before back transcription.

        y_after: 1d array-like.
            The outcome of the NLU model for the text after back transcription.

        x_before: 1d array-like, optional.
            Reference, i.e. the text before back transcription.

        x_after: 1d array-like, optional.
            Hypothesis, i.e. the text after back transcription.

        zero_division: str or float, optional, default='warn'.
            Sets the value to return when there is a zero division.

    Returns:
        score: float

    Notes:
        robust cases: constC, constI, I->C

        non-robust cases: C->I,  I->I

        irrelevant cases: -
    """
    return score_robustness(
        r123p_robust_case,
        r123p_non_robust_case,
        y_true,
        y_before,
        y_after,
        x_before,
        x_after,
        zero_division=zero_division,
    )


def c_to_i_count(y_true, y_before, y_after) -> int:
    """The number of model outputs that change from correct to incorrect after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if c_to_i_case(t, b, a):
            c += 1

    return c


def i_to_i_count(y_true, y_before, y_after) -> int:
    """The number of model outputs that change from incorrect to incorrect after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if i_to_i_case(t, b, a):
            c += 1

    return c


def i_to_c_count(y_true, y_before, y_after) -> int:
    """The number of model outputs that change from incorrect to correct after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if i_to_c_case(t, b, a):
            c += 1

    return c


def changed_count(y_true, y_before, y_after) -> int:
    """The number of model outputs that change after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if changed_case(t, b, a):
            c += 1

    return c


def i_const_count(y_true, y_before, y_after) -> int:
    """The number of incorrect model outputs that remain unchanged after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if i_const_case(t, b, a):
            c += 1

    return c


def c_const_count(y_true, y_before, y_after) -> int:
    """The number of correct model outputs that remain unchanged after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if c_const_case(t, b, a):
            c += 1

    return c


def const_count(y_true, y_before, y_after) -> int:
    """The number of model outputs that remain unchanged after back transcription."""
    c = 0

    for t, b, a in zip(y_true, y_before, y_after):
        if const_case(t, b, a):
            c += 1

    return c
