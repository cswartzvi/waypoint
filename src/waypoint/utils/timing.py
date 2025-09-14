from datetime import datetime


def format_duration(start: datetime | None, end: datetime | None) -> str:
    """Formats the duration between two datetime objects into a human-readable string."""
    if start is None or end is None:
        return "N/A"

    duration = end - start
    total_seconds = duration.total_seconds()

    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds_int = divmod(int(remainder), 60)

    # Calculate fractional seconds with microseconds
    fractional_seconds = total_seconds - int(total_seconds) + seconds_int

    parts = []
    if hours > 0:
        parts.append(f"{hours} h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes} m")
    parts.append(f"{fractional_seconds:.2f} s")

    return " ".join(parts)
