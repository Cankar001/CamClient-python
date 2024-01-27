import MonitorControl
import time

if __name__ == '__main__':
	monitor = MonitorControl.MonitorControl()
	time.sleep(2)
	print('Shutting down monitor')
	monitor.shutdownMonitor()
	time.sleep(1)

	# Now that the monitor is off, test if the script still executes
	file = open("test.txt", "a")
	file.write("Testing, if the script still executes if the monitor is off!\n")
	file.close()

	time.sleep(10)
	print('Awaking monitor')
	monitor.awakeMonitor()
	print('Demo successful')


