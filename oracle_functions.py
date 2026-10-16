"""Pure-Python functions for the public oracle examples."""

from collections.abc import Callable


def _validate_input(x: int) -> None:
    """Require an eight-bit integer input."""
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 255:
        raise ValueError("x must be an integer in the range 0 to 255")


def linear_public(x: int) -> int:
    """Return parity(178 & x) XOR 1 for the public linear example."""
    _validate_input(x)
    return (178 & x).bit_count() % 2 ^ 1


def simon_public(x: int) -> int:
    """Return min(x, x XOR 105) for the public Simon example."""
    _validate_input(x)
    return min(x, x ^ 105)


def simon2_public(x: int) -> int:
    """Return min(x XOR h) over the public Simon2 subspace H."""
    _validate_input(x)
    hidden_subspace = (0, 15, 49, 62)
    return min(x ^ h for h in hidden_subspace)


def permutation_public(x: int) -> int:
    """Return (37*x + 11) modulo 256 for the public permutation example."""
    _validate_input(x)
    return (37 * x + 11) % 256


def grover_public(x: int) -> int:
    """Return one only for the public Grover target x=173."""
    _validate_input(x)
    return int(x == 173)


def constant_public(x: int) -> int:
    """Return the constant c=1 for every valid input."""
    _validate_input(x)
    return 1


def validate_public_examples() -> None:
    """Raise AssertionError unless all Section 4 public values match."""
    reference_cases: tuple[tuple[Callable[[int], int], dict[int, int]], ...] = (
        (linear_public, {0: 1, 1: 1, 2: 0, 255: 1}),
        (simon_public, {5: 5, 108: 5, 105: 0, 200: 161}),
        (simon2_public, {0: 0, 15: 0, 49: 0, 62: 0, 1: 1, 200: 199}),
        (permutation_public, {0: 11, 1: 48, 2: 85, 255: 230}),
        (grover_public, {173: 1}),
        (constant_public, {0: 1, 255: 1}),
    )

    for function, expected_values in reference_cases:
        for x, expected in expected_values.items():
            actual = function(x)
            if actual != expected:
                raise AssertionError(
                    f"{function.__name__}({x}) returned {actual}; expected {expected}"
                )

def validate_invalid_inputs() -> None:
    """Verify that all public functions reject invalid inputs."""
    functions = (
        linear_public,
        simon_public,
        simon2_public,
        permutation_public,
        grover_public,
        constant_public,
    )
    invalid_inputs = (-1, 256, 1.5, True, "5", None)

    for function in functions:
        for x in invalid_inputs:
            try:
                function(x)
            except ValueError:
                continue
            raise AssertionError(
                f"{function.__name__} accepted invalid input {x!r}"
            )


if __name__ == "__main__":
    validate_public_examples()
    validate_invalid_inputs()
    print("All public oracle reference values validated.")
