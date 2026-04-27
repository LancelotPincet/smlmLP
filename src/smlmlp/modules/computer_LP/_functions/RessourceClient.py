#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-03-03
# Author        : Baptiste LEFEVRE
# GitHub        : https://github.com/LancelotPincet
# Library       : smlmLP
# Module        : computer

"""
System resource manager for RAM, VRAM, CPU, GPU, and disk.
"""

import json
import os
import socket
import time
import math



class RessourceClient:
    def __init__(self, file, backup_obj):
        self.file = file
        self.backup_obj = backup_obj
        self.broken_com = False
    
    def total(self):
        try:
            r = self.status()['total_tokens']
            self.broken_com = False
        except:
            r = self.backup_obj.total()
            self.broken_com = True
        return r
    
    def free(self):
        try:
            r = self.status()['free_tokens']
            self.broken_com = False
        except:
            r = self.backup_obj.free()
            self.broken_com = True
        return r

    def used(self):
        try:
            r =  self.status()['used_tokens']
            self.broken_com = False
        except:
            r = self.backup_obj.used()
            self.broken_com = True
        return r
        
    def status(self):
        try:
            r = status(self.file)
            self.broken_com = False
        except:
            self.broken_com = True
            r = {
                "ok": True,
                "total_tokens": self.backup_obj.total(),
                "used_tokens": self.backup_obj.used(),
                "free_tokens": self.backup_obj.free(),
                "reservations": 0,
            }
        return r
        
    def take(self, value, owner, minimum=None, wait_timeout=1*60*60, lease_seconds=2*60*60):
        self.status()
        if self.broken_com:
            return {"ok": True, "reason": "could not reach server", "granted_tokens":value}
        
        value = math.ceil(value)
        
        if minimum is None:
            minimum = value
        tokens = {'ok':False}
        time_quit = time.time() + wait_timeout
        while (not tokens['ok']) and (time_quit>time.time()):
            tokens = claim_up_to(
                min_tokens=minimum,
                max_tokens=value,
                owner=owner,
                lease_seconds=lease_seconds,
                file=self.file,
                )
        if tokens is None:
            return None # it failed. maybe wait longer or accept less memory
        
        return tokens
    
    def release(self, token):
        self.status()
        if self.broken_com:
            return {"ok": False, "reason": "could not reach server"}
        
        if isinstance(token, dict):
            try:
                res_id = token['reservation_id']
            except:
                raise ValueError('dict token must have a reservation_id field')
        elif isinstance(token, str):
            res_id = token
        else:
            raise ValueError('token must be a str or dict')
        
        
        return release(res_id, self.file)
        


def rpc(payload, file):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(file)
        connection = True
    except:
        connection = False
        raise Warning('Connection with server failed')
    
    if connection:
        s.sendall((json.dumps(payload) + "\n").encode())
    
        data = b""
        while not data.endswith(b"\n"):
            data += s.recv(65536)
        s.close()
    
        return json.loads(data.decode())
    else:
        return None



def free_tokens(file):
    return rpc({"op": "free"}, file)


def status(file):
    return rpc({"op": "status"}, file)


def claim_up_to(min_tokens, max_tokens, owner, file, lease_seconds=300):
    return rpc({
        "op": "claim_up_to",
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "owner": owner,
        "pid": os.getpid(),
        "lease_seconds": lease_seconds,
    }, file)


def refresh(reservation_id, file, lease_seconds=300):
    return rpc({
        "op": "refresh",
        "reservation_id": reservation_id,
        "lease_seconds": lease_seconds,
    },file)


def release(reservation_id, file):
    return rpc({
        "op": "release",
        "reservation_id": reservation_id,
    }, file)