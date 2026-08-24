"""Job assignment problem definition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    """Represents a job with processing times per machine.

    Each job can be executed on any machine, but takes different time on each.
    """

    id: int
    processing_times: list[float]  # processing_times[i] = time on machine i

    @property
    def num_machines(self) -> int:
        """Get number of machines this job can run on."""
        return len(self.processing_times)

    def time_on_machine(self, machine_id: int) -> float:
        """Get processing time on specific machine.

        Args:
            machine_id: Machine index (0-based)

        Returns:
            Processing time
        """
        return self.processing_times[machine_id]


def create_example_problem() -> list[Job]:
    """Create the example problem with 33 jobs and 5 machines.

    This replicates the PHP example problem instance.

    Returns:
        List of jobs
    """
    # Processing times for each job on each of 5 machines
    # Format: [job_id, [time_on_machine_0, time_on_machine_1, ...]]
    job_data = [
        [1, [63.0, 37.0, 25.0, 36.0, 58.0]],
        [2, [69.0, 59.0, 46.0, 42.0, 34.0]],
        [3, [65.0, 76.0, 66.0, 63.0, 56.0]],
        [4, [74.0, 53.0, 41.0, 59.0, 52.0]],
        [5, [63.0, 71.0, 43.0, 60.0, 55.0]],
        [6, [68.0, 61.0, 53.0, 50.0, 37.0]],
        [7, [54.0, 58.0, 51.0, 43.0, 48.0]],
        [8, [64.0, 68.0, 48.0, 40.0, 48.0]],
        [9, [66.0, 54.0, 59.0, 58.0, 62.0]],
        [10, [58.0, 72.0, 56.0, 49.0, 60.0]],
        [11, [66.0, 67.0, 57.0, 52.0, 38.0]],
        [12, [71.0, 63.0, 44.0, 38.0, 54.0]],
        [13, [75.0, 49.0, 54.0, 49.0, 53.0]],
        [14, [59.0, 69.0, 62.0, 56.0, 44.0]],
        [15, [74.0, 61.0, 60.0, 45.0, 60.0]],
        [16, [61.0, 55.0, 57.0, 59.0, 58.0]],
        [17, [73.0, 64.0, 44.0, 39.0, 50.0]],
        [18, [66.0, 72.0, 64.0, 44.0, 54.0]],
        [19, [65.0, 71.0, 62.0, 57.0, 55.0]],
        [20, [72.0, 56.0, 47.0, 53.0, 52.0]],
        [21, [65.0, 50.0, 65.0, 60.0, 60.0]],
        [22, [53.0, 55.0, 62.0, 55.0, 65.0]],
        [23, [73.0, 63.0, 49.0, 45.0, 50.0]],
        [24, [64.0, 68.0, 58.0, 61.0, 59.0]],
        [25, [67.0, 60.0, 62.0, 51.0, 60.0]],
        [26, [64.0, 64.0, 51.0, 51.0, 60.0]],
        [27, [66.0, 75.0, 53.0, 46.0, 50.0]],
        [28, [75.0, 61.0, 48.0, 46.0, 60.0]],
        [29, [63.0, 65.0, 44.0, 58.0, 60.0]],
        [30, [58.0, 55.0, 61.0, 66.0, 60.0]],
        [31, [65.0, 70.0, 54.0, 61.0, 60.0]],
        [32, [63.0, 67.0, 60.0, 60.0, 60.0]],
        [33, [69.0, 74.0, 67.0, 60.0, 50.0]],
    ]

    return [Job(id=job_id, processing_times=times) for job_id, times in job_data]


__all__ = ["Job", "create_example_problem"]
