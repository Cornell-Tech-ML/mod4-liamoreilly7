from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, Set, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)

    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol for a variable in the computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable.

        Args:
        ----
            x: The derivative value to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Return a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computation graph.

        Returns
        -------
            True if the variable is a leaf, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant.

        Returns
        -------
            True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for backpropagation.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            An iterable of tuples containing parent variables and their corresponding derivatives.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited: Set[int] = set()
    sorted_vars: List[Variable] = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for parent in v.parents:
                dfs(parent)

        visited.add(v.unique_id)
        sorted_vars.append(v)

    dfs(variable)
    return reversed(sorted_vars)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The variable for which to compute the derivative.
        deriv: The derivative to propagate back through the graph.

    """
    sorted_vars = topological_sort(variable)

    derivatives = {id(variable): deriv}

    for var in sorted_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[id(var)])

        else:
            for scalar, d in var.chain_rule(derivatives[id(var)]):
                if id(scalar) not in derivatives:
                    derivatives[id(scalar)] = 0
                derivatives[id(scalar)] += d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
