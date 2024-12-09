from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off
max_reduce = FastOps.reduce(operators.max, -1e9)


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    new_height = height // kh
    new_width = width // kw

    output = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
    )
    # Reshape to combine the kernel dimensions
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input (Tensor): batch x channel x height x width
        kernel (Tuple[int, int]): height x width of pooling

    Returns:
    -------
        Tensor: Pooled tensor

    """
    batch, channel, _, _ = input.shape
    tiled_input, _, _ = tile(input, kernel)
    pooled_tensor = tiled_input.mean(dim=4)
    pooled_tensor = pooled_tensor.view(
        batch, channel, pooled_tensor.shape[2], pooled_tensor.shape[3]
    )
    return pooled_tensor


# TODO: Implement for Task 4.3.


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (Tensor): input tensor
        dim (int): dimension to apply argmax

    Returns:
    -------
        Tensor: tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dimension: Tensor) -> Tensor:
        """Forward of max should be max reduction."""
        max_reduction = max_reduce(input_tensor, int(dimension.item()))
        ctx.save_for_backward(input_tensor, max_reduction)
        return max_reduction

    @staticmethod
    def backward(ctx: Context, gradient_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax."""
        input_tensor, max_reduction = ctx.saved_values
        return (gradient_output * (max_reduction == input_tensor)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input (Tensor): input tensor
        dim (int): dimension to apply softmax

    Returns:
    -------
        Tensor: softmax tensor

    """
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input (Tensor): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
    -------
        Tensor: log of softmax tensor

    """
    exp_input = input.exp()
    sum_exp_input = exp_input.sum(dim)
    log_sum_exp_input = sum_exp_input.log()
    return input - log_sum_exp_input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input (Tensor): batch x channel x height x width
        kernel (Tuple[int, int]): height x width of pooling

    Returns:
    -------
        Tensor: pooled tensor

    """
    batch_size, num_channels, _, _ = input.shape
    tiled_input, pooled_height, pooled_width = tile(input, kernel)
    pooled_input = max_reduce(tiled_input, 4)
    return pooled_input.contiguous().view(
        batch_size, num_channels, pooled_height, pooled_width
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input (Tensor): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
    -------
        Tensor: tensor with random positions dropped out

    """
    if not ignore:
        rand_tensor = rand(input.shape)
        random_drop = rand_tensor > rate
        return input * random_drop
    else:
        return input
