import subprocess
import time


class ProfileAMDEnergy:
    def __init__(self, tag, date, model, system_name, num_gpus, num_tokens, batch_size):
        self.tag = tag
        self.date = date
        self.model = model
        self.system_name = system_name
        self.num_gpus = num_gpus
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.path = f"{self.model}/{self.date}/{self.system_name}-{self.tag}-{self.num_gpus}gpus-{self.num_tokens}tokens-{self.batch_size}batch"

    def start_profiling(self) -> subprocess.Popen:
        # Start AMDuProf profiling
        proc = subprocess.Popen(
            [
                "AMDuProfCLI",
                "timechart",
                "--event",
                "power",
                "--interval",
                "100",
                "--duration",
                "99999",
                "-o",
                f"./{self.path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc

    def stop_profiling(self, proc):
        # Stop AMDuProf profiling and save the results
        proc.terminate()
        proc.wait()
