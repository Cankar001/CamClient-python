import subprocess
import io

class MonitorControl():
    def __init__(self):
        self.monitor_awake = True

    def runningOnRaspberry(self) -> bool:
        try:
            with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
                if 'raspberry pi' in m.read().lower():
                    return True
        except Exception:
            pass

        return False

    def awakeMonitor(self):
        if self.runningOnRaspberry():
            subprocess.call(['vcgencmd', 'display_power', 1])
            self.monitor_awake = True

    def shutdownMonitor(self):
        if self.runningOnRaspberry():
            subprocess.call(['vcgencmd', 'display_power', 0])
            self.monitor_awake = False

    def isMonitorAwake(self) -> bool:
        return self.monitor_awake

