from datetime import datetime

import pytest

from waypoint.utils.timing import format_duration


class TestFormatDuration:
    @pytest.mark.parametrize(
        "start,end,expected",
        [
            (None, None, "N/A"),
            (None, datetime(2023, 1, 1, 12, 0, 0), "N/A"),
            (datetime(2023, 1, 1, 12, 0, 0), None, "N/A"),
            (datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 0, 0), "0.00 s"),
            (datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 0, 5), "5.00 s"),
            (datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 1, 0), "1 m 0.00 s"),
            (datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 13, 0, 0), "1 h 0 m 0.00 s"),
            (
                datetime(2023, 1, 1, 12, 0, 0),
                datetime(2023, 1, 1, 13, 30, 15, 123456),
                "1 h 30 m 15.12 s",
            ),
            (
                datetime(2023, 1, 1, 12, 0, 0),
                datetime(2023, 1, 2, 14, 45, 30, 654321),
                "26 h 45 m 30.65 s",
            ),
        ],
    )
    def test_format_duration(self, start, end, expected):
        assert format_duration(start, end) == expected
