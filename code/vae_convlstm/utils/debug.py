import math
from typing import Tuple

# Define unit labels for memory sizes
MEMORY_UNITS = {
    0: 'B',
    1: 'KiB',
    2: 'MiB',
    3: 'GiB',
    4: 'TiB'
}

def format_elapsed_time(start_time: float, end_time: float) -> str:
    elapsed_seconds = end_time - start_time
    minutes, seconds = divmod(elapsed_seconds, 60)
    return f'{int(minutes)} m {seconds:.4f} s'


def format_memory_size(bytes: int) -> Tuple[int, str]:
    if abs(bytes) < 1024:
        return round(bytes, 2), 'B'

    scale = math.log2(abs(bytes)) // 10
    scaled_value = bytes / 1024 ** scale
    unit = MEMORY_UNITS[scale]

    if int(scaled_value) == scaled_value:
        return int(scaled_value), unit

    # Rounding to 2 decimal places, as required
    return round(scaled_value, 2), unit

def format_readable_memory_size(bytes: int) -> str:
    value, unit = format_memory_size(bytes)
    return f'{value} {unit}'


def print_model_size(model):
    total_parameters = 0
    total_memory_bytes = 0

    for param in model.parameters():
        total_parameters += param.numel()
        total_memory_bytes += param.data.element_size() * param.numel()

    formatted_memory_size = format_readable_memory_size(total_memory_bytes)
    print(f'Total number of model parameters: {total_parameters} (~{formatted_memory_size})')

    return total_parameters, total_memory_bytes