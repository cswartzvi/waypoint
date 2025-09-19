from waypoint import flow
from waypoint.results import result_mapper


@result_mapper
def save_text(data: str) -> bytes:
    """Simple text result mapper."""

    return data.encode("utf-8")

save_text("example.txt")

@flow(result=save_text("example.txt"), store="./testing_results")
def main() -> str:
    return "Hello, Waypoint!"


if __name__ == "__main__":
    result = main()
    print(result)
