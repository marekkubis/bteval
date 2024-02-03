import numpy as np
import torch
from bteval import r1_score, r12_score, r13_score, r13p_score, r123_score, r123p_score
from pytest import approx, warns


def test_r1_score():
    # robust: 0, non-robust: 0, irrelevant: 3
    with warns(UserWarning):
        assert (
            r1_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Confirm", "Confirm", "Confirm"],
            )
            == 0
        )

    # robust: 0, non-robust: 1, irrelevant: 2
    assert r1_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Confirm", "Confirm"],
    ) == approx(0)

    # robust: 1, non-robust: 0, irrelevant: 2
    assert r1_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1)

    # robust: 1, non-robust: 1, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(0.5)

    # robust: 2, non-robust: 0, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Request", "Confirm"],
    ) == approx(1.0)

    # robust: 0, non-robust: 2, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 0, non-robust: 3, irrelevant: 0
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)


def test_r13_score():
    # robust: 0, non-robust: 0, irrelevant: 3
    with warns(UserWarning):
        assert (
            r13_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Confirm", "Confirm", "Confirm"],
            )
            == 0
        )

    # robust: 0, non-robust: 1, irrelevant: 2
    assert r13_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Confirm", "Confirm"],
    ) == approx(0)
    assert r13_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Confirm", "Inform"],
    ) == approx(0)

    # robust: 1, non-robust: 0, irrelevant: 2
    assert r13_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1)

    # robust: 1, non-robust: 1, irrelevant: 1
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(0.5)
    assert r13_score(
        ["Inform", "Deny", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Inform"],
    ) == approx(0.5)

    # robust: 2, non-robust: 0, irrelevant: 1
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Request", "Confirm"],
    ) == approx(1.0)

    # robust: 0, non-robust: 2, irrelevant: 1
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Request"],
        ["Request", "Request", "Confirm"],
    ) == approx(0.0)

    # robust: 0, non-robust: 3, irrelevant: 0
    assert r13_score(
        ["Inform", "Request", "Request"],
        ["Inform", "Deny", "Request"],
        ["Request", "Request", "Confirm"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Deny"],
        ["Inform", "Request", "Inform"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r13_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)


def test_r13p_score():
    # robust: 0, non-robust: 0, irrelevant: 3
    with warns(UserWarning):
        assert (
            r13p_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Confirm", "Confirm", "Confirm"],
            )
            == 0
        )

    # robust: 0, non-robust: 1, irrelevant: 2
    assert r13p_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Confirm", "Confirm"],
    ) == approx(0)

    # robust: 1, non-robust: 0, irrelevant: 2
    assert r13p_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1)
    assert r13p_score(
        ["Inform", "Inform", "Inform"],
        ["Deny", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1)

    # robust: 1, non-robust: 1, irrelevant: 1
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(0.5)
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Deny", "Request", "Request"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(0.5)

    # robust: 2, non-robust: 0, irrelevant: 1
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Deny", "Request", "Request"],
        ["Inform", "Request", "Confirm"],
    ) == approx(1.0)

    # robust: 0, non-robust: 2, irrelevant: 1
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 0, non-robust: 3, irrelevant: 0
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Deny", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Deny", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r13p_score(
        ["Inform", "Request", "Inform"],
        ["Deny", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)


def test_r12_score():
    # robust: 0, non-robust: 0, irrelevant: 3
    with warns(UserWarning):
        assert (
            r12_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Inform", "Inform", "Inform"],
            )
            == 0
        )

    # robust: 0, non-robust: 1, irrelevant: 2
    assert r12_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Confirm", "Confirm"],
    ) == approx(0)
    assert r12_score(
        ["Inform", "Inform", "Inform"],
        ["Request", "Request", "Request"],
        ["Confirm", "Confirm", "Confirm"],
    ) == approx(0)

    # robust: 1, non-robust: 0, irrelevant: 2
    assert r12_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Inform", "Inform"],
    ) == approx(1)

    # robust: 1, non-robust: 1, irrelevant: 1
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Inform"],
    ) == approx(0.5)
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Request"],
        ["Inform", "Confirm", "Inform"],
    ) == approx(0.5)

    # robust: 2, non-robust: 0, irrelevant: 1
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Request"],
        ["Inform", "Deny", "Inform"],
    ) == approx(1.0)

    # robust: 0, non-robust: 2, irrelevant: 1
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 0, non-robust: 3, irrelevant: 0
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Deny"],
        ["Request", "Confirm", "Confirm"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Deny"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Deny"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)
    assert r12_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Deny"],
        ["Inform", "Request", "Deny"],
    ) == approx(1)


def test_r123_score():
    # robust: 0, non-robust: 3, irrelevant: 0
    assert r123_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
        ["Request", "Confirm", "Inform"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r123_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)
    assert r123_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
        ["Inform", "Confirm", "Inform"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r123_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Inform"],
        ["Inform", "Deny", "Confirm"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r123_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)


def test_r123p_score():
    # robust: 0, non-robust: 3, irrelevant: 0
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
        ["Request", "Confirm", "Deny"],
    ) == approx(0.0)

    # robust: 1, non-robust: 2, irrelevant: 0
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Confirm", "Confirm"],
    ) == approx(1 / 3)
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Confirm"],
        ["Inform", "Confirm", "Deny"],
    ) == approx(1 / 3)

    # robust: 2, non-robust: 1, irrelevant: 0
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Inform"],
        ["Inform", "Deny", "Confirm"],
    ) == approx(2 / 3)
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Deny", "Inform"],
        ["Inform", "Request", "Confirm"],
    ) == approx(2 / 3)

    # robust: 3, non-robust: 0, irrelevant: 0
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Request", "Confirm", "Request"],
        ["Request", "Confirm", "Request"],
    ) == approx(1)
    assert r123p_score(
        ["Inform", "Request", "Inform"],
        ["Request", "Confirm", "Request"],
        ["Inform", "Request", "Inform"],
    ) == approx(1)


def test_const_x_elimination():
    # robust: 0, non-robust: 0, irrelevant: 3
    with warns(UserWarning):
        assert (
            r1_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Inform", "Inform", "Inform"],
                ["a", "b", "c"],
                ["a", "b", "c"],
            )
            == 0
        )

    # robust: 0, non-robust: 1, irrelevant: 2
    assert r1_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Confirm", "Inform", "Inform"],
        ["a", "b", "c"],
        ["x", "b", "c"],
    ) == approx(0)

    # robust: 1, non-robust: 0, irrelevant: 2
    assert r1_score(
        ["Inform", "Inform", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Inform", "Inform"],
        ["a", "b", "c"],
        ["x", "b", "c"],
    ) == approx(1)

    # robust: 1, non-robust: 1, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Confirm", "Inform"],
        ["a", "b", "c"],
        ["x", "y", "c"],
    ) == approx(0.5)

    # robust: 2, non-robust: 0, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Inform", "Request", "Inform"],
        ["a", "b", "c"],
        ["x", "y", "c"],
    ) == approx(1.0)

    # robust: 0, non-robust: 2, irrelevant: 1
    assert r1_score(
        ["Inform", "Request", "Inform"],
        ["Inform", "Request", "Request"],
        ["Request", "Confirm", "Inform"],
        ["a", "b", "c"],
        ["x", "y", "c"],
    ) == approx(0.0)


def test_zero_division():
    with warns(UserWarning):
        assert (
            r1_score(
                ["Inform", "Inform", "Inform"],
                ["Request", "Request", "Request"],
                ["Inform", "Inform", "Inform"],
            )
            == 0
        )

    assert (
        r1_score(
            ["Inform", "Inform", "Inform"],
            ["Request", "Request", "Request"],
            ["Inform", "Inform", "Inform"],
            zero_division=0.0,
        )
        == 0.0
    )
    assert (
        r1_score(
            ["Inform", "Inform", "Inform"],
            ["Request", "Request", "Request"],
            ["Inform", "Inform", "Inform"],
            zero_division=1.0,
        )
        == 1.0
    )


def test_example():
    y_true = ["Inform", "Request", "Inform"]
    y_before = ["Inform", "Request", "Request"]
    y_after = ["Inform", "Confirm", "Confirm"]

    assert r1_score(y_true, y_before, y_after) == approx(0.5)


def test_numpy():
    y_true = np.asarray(["Inform", "Request", "Inform"])
    y_before = np.asarray(["Inform", "Request", "Request"])
    y_after = np.asarray(["Inform", "Confirm", "Confirm"])

    assert r1_score(y_true, y_before, y_after) == approx(0.5)


def test_torch():
    y_true = torch.tensor([1, 2, 1])
    y_before = torch.tensor([1, 2, 2])
    y_after = torch.tensor([1, 3, 3])

    assert r1_score(y_true, y_before, y_after) == approx(0.5)
