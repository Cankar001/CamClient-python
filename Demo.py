import os

import Camera as c

if __name__ == '__main__':
    cam = c.Camera(mirror=True, seconds_until_monitor_off=60, show_debug_text=True)
    cam.stream()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
