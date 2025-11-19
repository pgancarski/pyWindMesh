import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        print("Timer started.")

    def stop(self):
        """Stop the timer and print the elapsed time."""
        if self.start_time is None:
            print("Timer was not started.")
            return
        
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"Elapsed time: {elapsed:.3f} seconds")

    def elapsed(self):
        """Return the elapsed time without stopping."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time)
