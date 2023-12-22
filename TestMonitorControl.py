import MonitorControl
import time

if __name__ == '__main__':
	monitor = MonitorControl.MonitorControl()
	time.sleep(2)
	print('Shutting down monitor')
	monitor.shutdownMonitor()
	time.sleep(2)
	print('Awaking monitor')
	monitor.awakeMonitor()
	print('Demo successful')


