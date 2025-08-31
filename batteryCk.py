import socket
import time
import argparse

def get_battery_via_socket(timeout=3.0, retries=3, bind_port=9000):
    TELLO_ADDR = ("192.168.10.1", 8889)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)

    # try to bind to a fixed local port for consistency; fall back to ephemeral if in use
    try:
        s.bind(("", bind_port))
    except OSError:
        pass  # ephemeral port will be used

    def send_and_recv(msg):
        s.sendto(msg.encode("utf-8"), TELLO_ADDR)
        return s.recvfrom(1024)[0].decode("utf-8", errors="ignore").strip()

    # Enter SDK mode (retry a few times)
    ok = False
    for _ in range(retries):
        try:
            resp = send_and_recv("command")
            if resp.lower() == "ok":
                ok = True
                break
        except socket.timeout:
            pass
        time.sleep(0.1)

    if not ok:
        s.close()
        raise RuntimeError("failed to enter SDK mode (no 'ok'). check Wi-Fi connection to TELLO.")

    # Ask for battery
    for _ in range(retries):
        try:
            resp = send_and_recv("battery?")
            s.close()
            # Tello returns a number like "87"
            return int(resp)
        except (ValueError, socket.timeout):
            time.sleep(0.1)

    s.close()
    raise RuntimeError("failed to read battery (timeout or invalid response).")

def get_battery_via_djitellopy():
    """
    Uses djitellopy if installed. More convenient, handles some quirks for you.
    """
    from djitellopy import Tello
    t = Tello()
    try:
        t.connect()  # must be on TELLO Wi-Fi
        level = t.get_battery()  # returns int
        return int(level)
    finally:
        # end() safely closes sockets even if connect() failed
        try:
            t.end()
        except Exception:
            pass

def ascii_gauge(pct, width=20):
    pct = max(0, min(100, int(pct)))
    fill = (pct * width) // 100
    return f"[{'#'*fill}{'.'*(width-fill)}] {pct}%"

def main():
    ap = argparse.ArgumentParser(description="Show DJI Tello battery level.")
    ap.add_argument("--loop", type=float, default=0.0, help="poll every N seconds (0 = once)")
    ap.add_argument("--timeout", type=float, default=3.0, help="UDP timeout seconds for socket mode")
    ap.add_argument("--retries", type=int, default=3, help="retries for socket mode")
    args = ap.parse_args()

    def read_once():
        # try djitellopy first, fall back to socket
        try:
            try:
                level = get_battery_via_djitellopy()
            except ModuleNotFoundError:
                level = get_battery_via_socket(timeout=args.timeout, retries=args.retries)
        except Exception as e:
            print("error:", e)
            return None
        return level

    if args.loop > 0:
        print("polling battery. press Ctrl+C to stop.")
        while True:
            level = read_once()
            if level is not None:
                print(ascii_gauge(level))
            time.sleep(args.loop)
    else:
        level = read_once()
        if level is not None:
            print(ascii_gauge(level))

if __name__ == "__main__":
    main()
