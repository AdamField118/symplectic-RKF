"""Live viewer for the black hole: a web page you can fly around in.

    python visualization/viewer.py [port]

Run it on a GPU node. It starts the renderer (./blackhole --serve) and a small
web server; open the address it prints in a browser. On Turing that's an Open
OnDemand address, so it works from your laptop with nothing else to set up.

The renderer draws a frame whenever the browser has room for one: at most two
frames are on their way at a time, so the GPU can draw the next while the last
is travelling, but frames never pile up and the picture keeps up with the
mouse. Each frame goes to the browser as a JPEG over a websocket. Drag to move
around the hole at the same distance, scroll to change the distance.

Options after the port go to the renderer, for example `--sky starfield.rgb`.
"""
import io
import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FLYTHROUGH_SECONDS = 8.0

# What the page can change, and where it starts
state = {
    "r": 20.0, "theta": 0.525 * math.pi, "phi": 0.0,
    "a": 0.5, "fov": 60.0, "step": 0.08,
    "disk": True, "doppler": True,
    "spin_gas": True, "disk_speed": 8.0,
    "width": 1280, "height": 720, "quality": 85,
    "playing": False, "path_t": 0.0,
}
LIMITS = {"r": (3.0, 200.0), "theta": (0.05, math.pi - 0.05), "a": (0.0, 0.998),
          "fov": (20.0, 120.0), "step": (0.01, 0.3), "disk_speed": (0.0, 40.0),
          "width": (64, 3840), "height": (64, 2160), "quality": (30, 95)}
lock = threading.Lock()


class Credits:
    """How many more frames the browser has room for, at most IN_FLIGHT."""
    IN_FLIGHT = 2

    def __init__(self):
        self.count = 0
        self.changed = threading.Condition()

    def give(self):
        with self.changed:
            self.count = min(self.IN_FLIGHT, self.count + 1)
            self.changed.notify()

    def take(self):
        with self.changed:
            while self.count == 0:
                self.changed.wait()
            self.count -= 1


credits = Credits()


def clamp(key):
    lo, hi = LIMITS[key]
    state[key] = min(hi, max(lo, state[key]))


# The renderer
class Renderer:
    def __init__(self, extra_args):
        exe = os.path.join(HERE, "blackhole")
        if not os.path.exists(exe):
            sys.exit(f"{exe} isn't built yet: run make -C visualization")
        self.proc = subprocess.Popen([exe, "--serve", *extra_args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read(self, n):
        data = self.proc.stdout.read(n)
        if len(data) != n:
            raise RuntimeError("the renderer stopped (see its messages above)")
        return data

    def _send(self, line):
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def frame(self, w, h, r, theta, phi, t, a, fov, step, disk, doppler):
        self._send(f"frame {w} {h} {r!r} {theta!r} {phi!r} {t!r} {a!r} {fov!r} {step!r} "
                   f"{int(disk)} {int(doppler)}")
        w, h, ms = struct.unpack("<iif", self._read(12))
        return Image.frombytes("RGB", (w, h), self._read(w * h * 3)), ms

    def path(self, t, r):
        self._send(f"path {t!r} {r!r}")
        return struct.unpack("<3d", self._read(24))


# The web server
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")


class OnDemandPrefix:
    """Open OnDemand passes requests on as /node/<host>/<port>/...; strip that
    off so the routes below match."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app
        self.pattern = re.compile(r"^(/(?:node|rnode)/[^/]+/[^/]+)(.*)")

    def __call__(self, environ, start_response):
        match = self.pattern.match(environ.get("PATH_INFO", ""))
        if match:
            environ["SCRIPT_NAME"], rest = match.groups()
            environ["PATH_INFO"] = rest or "/"
        return self.wsgi_app(environ, start_response)


app.wsgi_app = OnDemandPrefix(app.wsgi_app)


@app.route("/")
def index():
    return send_from_directory(os.path.join(HERE, "viewer"), "index.html")


@socketio.on("connect")
def on_connect():
    with lock:
        emit("settings", {k: v for k, v in state.items() if k != "path_t"})


@socketio.on("ready")
def on_ready():
    credits.give()


@socketio.on("orbit")
def on_orbit(d):
    """Move around the hole by (dphi, dtheta) radians, at the same distance."""
    with lock:
        state["playing"] = False
        state["phi"] = (state["phi"] + float(d.get("dphi", 0))) % (2 * math.pi)
        state["theta"] += float(d.get("dtheta", 0))
        clamp("theta")


@socketio.on("zoom")
def on_zoom(d):
    with lock:
        state["r"] *= float(d.get("factor", 1))
        clamp("r")


@socketio.on("set")
def on_set(d):
    with lock:
        for key, value in d.items():
            if key in ("disk", "doppler", "spin_gas", "playing"):
                state[key] = bool(value)
            elif key in ("width", "height", "quality"):
                state[key] = int(value)
                clamp(key)
            elif key in LIMITS:
                state[key] = float(value)
                clamp(key)
            elif key == "reset":
                state.update(r=20.0, theta=0.525 * math.pi, phi=0.0, playing=False, path_t=0.0)


def render_loop(renderer):
    try:
        draw_frames(renderer)
    except Exception as e:  # without the renderer there's nothing to show
        print(f"error: {e}", file=sys.stderr, flush=True)
        os._exit(1)


def draw_frames(renderer):
    gas_time = 0.0
    last = time.perf_counter()
    frames, window_start = 0, time.perf_counter()
    fps = 0.0
    while True:
        credits.take()
        start = time.perf_counter()
        dt, last = start - last, start
        with lock:
            if state["spin_gas"]:
                gas_time += min(dt, 0.25) * state["disk_speed"]
            if state["playing"]:
                state["path_t"] = (state["path_t"] + min(dt, 0.25) / FLYTHROUGH_SECONDS) % 1.0
                _, state["theta"], state["phi"] = renderer.path(state["path_t"], state["r"])
            s = dict(state)

        image, render_ms = renderer.frame(s["width"], s["height"], s["r"], s["theta"], s["phi"],
                                          gas_time, s["a"], s["fov"], s["step"], s["disk"],
                                          s["doppler"])
        got = time.perf_counter()
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=s["quality"])
        jpeg = buf.getvalue()
        encoded = time.perf_counter()

        frames += 1
        if encoded - window_start >= 1.0:
            fps = frames / (encoded - window_start)
            frames, window_start = 0, encoded
        socketio.emit("frame", {
            "jpeg": jpeg,
            "stats": {
                "fps": fps, "render_ms": render_ms,
                "transfer_ms": (got - start) * 1e3 - render_ms,
                "encode_ms": (encoded - got) * 1e3, "kb": len(jpeg) / 1024,
                "width": image.width, "height": image.height,
            },
            "camera": {"r": s["r"], "theta": s["theta"], "phi": s["phi"], "playing": s["playing"]},
        })


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4630
    extra = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else sys.argv[1:]
    renderer = Renderer(extra)
    threading.Thread(target=render_loop, args=(renderer,), daemon=True).start()

    host = socket.getfqdn().split(".")[0] + ".int.turing.wpi.edu"
    print(f"On Turing, open  https://ondemand.turing.wpi.edu/node/{host}/{port}/", flush=True)
    print(f"Elsewhere, open  http://localhost:{port}/  (through an ssh tunnel if it's remote)",
          flush=True)
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
