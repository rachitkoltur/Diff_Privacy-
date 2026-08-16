"""
keylogger.py
============
Records keystroke timestamps as ground truth for an EMG data-collection session
(see DATA_COLLECTION_PROTOCOL.md). Writes keylog.csv with columns:

    time,key,event

where `time` is a high-resolution monotonic clock in seconds (aligned to the EMG
recorder's start), `key` is the key label, and `event` is "press" or "release".

Use only with informed consent, on your own device, for your own study. Press ESC
to stop.

    pip install pynput
    python keylogger.py keylog.csv
"""
import sys, time, csv

try:
    from pynput import keyboard
except ImportError:
    sys.exit("Install pynput first:  pip install pynput")


def main(out_path="keylog.csv"):
    t0 = time.monotonic()
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["time", "key", "event"])
    print("Recording to %s. Press ESC to stop." % out_path)

    def label(key):
        try:
            return key.char
        except AttributeError:
            return str(key)

    def on_press(key):
        if key == keyboard.Key.esc:
            return False
        w.writerow(["%.6f" % (time.monotonic() - t0), label(key), "press"])
        f.flush()

    def on_release(key):
        w.writerow(["%.6f" % (time.monotonic() - t0), label(key), "release"])
        f.flush()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as L:
        L.join()
    f.close()
    print("Saved %s." % out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "keylog.csv")
