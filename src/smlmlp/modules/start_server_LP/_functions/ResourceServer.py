#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-03
# Author        : Baptiste LEFEVRE
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : computer

"""
Server for managing computer properties.
"""

import json
import os
import socket
import threading
import time
import uuid
from typing import Optional


class ResourceServer:
    def __init__(
        self,
        sock_path: str,
        total_tokens: int,
        admin_key: str = "change-me",
        socket_mode: int = 0o666,
        log_prefix: Optional[str] = None,
    ):
        self.sock_path = sock_path
        self.total_tokens = int(total_tokens)
        self.admin_key = admin_key
        self.socket_mode = socket_mode
        self.log_prefix = log_prefix or f"[{sock_path}]"

        self.reservations = {}
        self.lock = threading.Lock()
        self._server_socket = None
        self._accept_thread = None
        self._stop_event = threading.Event()

    def _log(self, *parts):
        print(self.log_prefix, *parts, flush=True)

    def purge_expired(self):
        now = time.time()
        dead = [
            rid for rid, r in self.reservations.items()
            if r["expires_at"] is not None and r["expires_at"] < now
        ]
        for rid in dead:
            self._log("EXPIRE", rid, self.reservations[rid])
            del self.reservations[rid]

    def used_tokens(self):
        return sum(r["tokens"] for r in self.reservations.values())

    def free_tokens(self):
        return self.total_tokens - self.used_tokens()

    def status_payload(self):
        return {
            "ok": True,
            "socket_path": self.sock_path,
            "total_tokens": self.total_tokens,
            "used_tokens": self.used_tokens(),
            "free_tokens": self.free_tokens(),
            "reservations": dict(self.reservations),
        }

    def _handle_request(self, req):
        op = req["op"]

        with self.lock:
            self.purge_expired()

            if op == "free":
                return {"ok": True, "free_tokens": self.free_tokens()}

            if op == "status":
                return self.status_payload()

            if op == "claim_up_to":
                min_tokens = int(req["min_tokens"])
                max_tokens = int(req["max_tokens"])
                owner = req.get("owner", "unknown")
                pid = int(req.get("pid", 0))
                lease_seconds = int(req.get("lease_seconds", 300))

                free_now = self.free_tokens()
                if free_now < min_tokens:
                    return {
                        "ok": False,
                        "granted_tokens": 0,
                        "reason": "not_enough_tokens",
                        "free_tokens": free_now,
                    }

                granted = min(max_tokens, free_now)
                rid = uuid.uuid4().hex
                now = time.time()
                self.reservations[rid] = {
                    "owner": owner,
                    "pid": pid,
                    "tokens": granted,
                    "created_at": now,
                    "expires_at": now + lease_seconds if lease_seconds > 0 else None,
                }
                self._log("CLAIM", rid, self.reservations[rid])
                return {
                    "ok": True,
                    "reservation_id": rid,
                    "granted_tokens": granted,
                    "free_tokens": self.free_tokens(),
                }

            if op == "refresh":
                rid = req["reservation_id"]
                lease_seconds = int(req.get("lease_seconds", 300))
                if rid not in self.reservations:
                    return {"ok": False, "reason": "unknown_reservation"}
                self.reservations[rid]["expires_at"] = time.time() + lease_seconds
                return {"ok": True}

            if op == "release":
                rid = req["reservation_id"]
                existed = self.reservations.pop(rid, None) is not None
                if existed:
                    self._log("RELEASE", rid)
                return {"ok": existed}

            if op == "force_release":
                if req.get("admin_key") != self.admin_key:
                    return {"ok": False, "reason": "bad_admin_key"}
                rid = req["reservation_id"]
                existed = self.reservations.pop(rid, None) is not None
                if existed:
                    self._log("FORCE_RELEASE", rid)
                return {"ok": existed}

            return {"ok": False, "reason": "unknown op"}

    def _handle_connection(self, conn):
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk

            req = json.loads(data.decode())
            resp = self._handle_request(req)
            conn.sendall((json.dumps(resp) + "\n").encode())

        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "reason": str(e)}) + "\n").encode())
            except Exception:
                pass
        finally:
            conn.close()

    def _accept_loop(self):
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except OSError:
                break
            threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()

    def start(self):
        if self._server_socket is not None:
            raise RuntimeError("server already started")

        os.makedirs(os.path.dirname(self.sock_path) or ".", exist_ok=True)
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self.sock_path)
        os.chmod(self.sock_path, self.socket_mode)
        self._server_socket.listen()

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._log("Listening", "total_tokens=", self.total_tokens)
        return self

    def serve_forever(self):
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self._server_socket is None:
            return

        self._stop_event.set()
        try:
            self._server_socket.close()
        except Exception:
            pass
        self._server_socket = None

        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

        self._log("Stopped")


def main():
    server = ResourceServer(
        sock_path="/tmp/ramtokens.sock",
        total_tokens=96 * 1024,
        admin_key="change-me",
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
