import psutil, time
process_found = None
for process in psutil.process_iter():
    if "python" in process.name():
        process_cmd_line = ' '.join(process.cmdline())
        if "test_selenium" in process_cmd_line:
            process_found = process
            break

while(True):
    if not process_found.is_running():
        break
    elif process_found.cpu_percent(1) > 80:
        process_found.kill()
        break
    time.sleep(10)
