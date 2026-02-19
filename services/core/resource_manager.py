import psutil
class SystemMonitor:
    def __init__(self, cpu_limit=90, ram_min_mb=500):
        self.cpu_limit = cpu_limit
        self.ram_min_mb = ram_min_mb
    def check_health(self):
        cpu = psutil.cpu_percent(interval=0.5)
        if cpu > self.cpu_limit: return False
        mem = psutil.virtual_memory()
        if (mem.available / 1024 / 1024) < self.ram_min_mb: return False
        return True
