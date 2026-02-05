#!/usr/bin/env python3
"""Streamlink IGT Timer Capture Server - Fixed for Render.com"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set
import base64

import numpy as np
import cv2
import websockets
from websockets.legacy.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Must use 0.0.0.0, not localhost
PORT = int(os.environ.get("PORT", 10000))
HOST = "0.0.0.0"

@dataclass
class StreamConfig:
    url: str
    quality: str = "best"

class StreamCapture:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.subscribers: Set[WebSocketServerProtocol] = set()
        self.config: Optional[StreamConfig] = None
        self.capture_thread = None
        self.loop = None
        
        self.igt_config = {'x_start': 1616, 'y_start': 84, 'char_width': 20, 'char_height': 25}
        self.digit_positions = [
            {'x_start': 88}, {'x_start': 112}, {'x_start': 144}, {'x_start': 168},
            {'x_start': 200}, {'x_start': 224}, {'x_start': 248}
        ]
        self._shutdown_event = threading.Event()

    def is_yellow_pixel(self, r, g, b):
        return r > 200 and g > 240 and b > 60 and b < 150

    def extract_digit_pattern(self, image, x_start, y_start, width, height):
        pattern = np.zeros((height, width), dtype=np.uint8)
        roi = image[y_start:y_start+height, x_start:x_start+width]
        if roi.shape[0] != height or roi.shape[1] != width:
            return pattern
        for y in range(height):
            for x in range(width):
                b, g, r = roi[y, x]
                pattern[y, x] = 1 if self.is_yellow_pixel(r, g, b) else 0
        return pattern

    def load_digit_templates(self):
        templates = {}
        patterns = {
            '0': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,1,1],[1,0,1,0,1],[1,1,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
            '1': [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
            '2': [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,1,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
            '3': [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
            '4': [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]],
            '5': [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
            '6': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
            '7': [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0]],
            '8': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
            '9': [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]
        }
        for digit, pattern in patterns.items():
            arr = np.array(pattern, dtype=np.uint8)
            templates[digit] = cv2.resize(arr, (20, 25), interpolation=cv2.INTER_NEAREST)
        return templates

    def recognize_digit(self, pattern, templates):
        best_match, best_score = None, 0
        for digit, template in templates.items():
            if pattern.shape != template.shape:
                continue
            score = np.sum(pattern == template) / pattern.size
            if score > best_score:
                best_score, best_match = score, digit
        return best_match, best_score

    def extract_igt_timer(self, image):
        templates = self.load_digit_templates()
        cfg = self.igt_config
        digits, scores = [], []
        
        for pos in self.digit_positions:
            abs_x = cfg['x_start'] + pos['x_start']
            pattern = self.extract_digit_pattern(image, abs_x, cfg['y_start'], cfg['char_width'], cfg['char_height'])
            digit, score = self.recognize_digit(pattern, templates)
            digits.append(digit if digit else '?')
            scores.append(score if digit else 0)
        
        if len(digits) == 7:
            return {
                'time': f"{digits[0]}{digits[1]}:{digits[2]}{digits[3]}.{digits[4]}{digits[5]}{digits[6]}",
                'confidence': sum(scores) / 7,
                'digits': digits,
                'scores': scores
            }
        return None

    async def start_stream(self, url: str, quality: str = "best"):
        try:
            self.config = StreamConfig(url=url, quality=quality)
            
            # Start streamlink (outputs raw stream to stdout)
            streamlink_cmd = ['streamlink', url, quality, '--stdout', '--loglevel', 'warning']
            self.process = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.is_running = True
            self.loop = asyncio.get_event_loop()
            self.capture_thread = threading.Thread(target=self._capture_worker)
            self.capture_thread.start()
            
            logger.info(f"Started stream: {url}")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False

    def _capture_worker(self):
        """Read frames using FFmpeg"""
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-hide_banner', '-loglevel', 'error',
                '-i', 'pipe:0',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', '1920x1080',
                '-r', '30',
                'pipe:1'
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self.process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            frame_size = 1920 * 1080 * 3
            frame_count = 0
            last_time = time.time()
            
            while self.is_running and not self._shutdown_event.is_set():
                raw = self.ffmpeg_process.stdout.read(frame_size)
                if len(raw) != frame_size:
                    if self.is_running:
                        logger.warning(f"Incomplete frame: {len(raw)} bytes")
                    continue
                
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((1080, 1920, 3))
                self.current_frame = frame.copy()
                
                # OCR every 3rd frame
                timer_data = self.extract_igt_timer(frame) if frame_count % 3 == 0 else None
                
                # Send preview every 5th frame
                if frame_count % 5 == 0:
                    small = cv2.resize(frame, (960, 540))
                    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    msg = {
                        'type': 'FRAME',
                        'image': base64.b64encode(buf).decode(),
                        'timestamp': time.time(),
                        'resolution': '1920x1080',
                        'timer': timer_data
                    }
                    if self.loop:
                        asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"FPS: {30 / (time.time() - last_time):.1f}")
                    last_time = time.time()
                    
        except Exception as e:
            logger.error(f"Capture error: {e}")

    async def broadcast(self, message: dict):
        if not self.subscribers:
            return
        disconnected = set()
        msg_str = json.dumps(message)
        for ws in self.subscribers:
            try:
                await ws.send(msg_str)
            except:
                disconnected.add(ws)
        self.subscribers -= disconnected

    def stop(self):
        self.is_running = False
        self._shutdown_event.set()
        for p in [self.ffmpeg_process, self.process]:
            if p:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except:
                    p.kill()
        if self.capture_thread:
            self.capture_thread.join(timeout=3)
        logger.info("Stream stopped")


class WebSocketServer:
    def __init__(self):
        self.capture = StreamCapture()

    async def process_request(self, path, request_headers):
        """Handle HTTP requests - allows health checks and browser connections"""
        if path == "/health":
            return (
                200,
                [("Content-Type", "text/plain")],
                b"OK - WebSocket Server Running\n"
            )
        elif path == "/":
            html = b"""<!DOCTYPE html>
<html>
<head><title>WebSocket Server</title></head>
<body>
<h1>WebSocket Server Running</h1>
<p>Connect using WebSocket protocol: <code>ws://""" + f"{HOST}:{PORT}".encode() + b"""</code></p>
<p>Health check: <a href="/health">/health</a></p>
</body>
</html>"""
            return (
                200,
                [("Content-Type", "text/html")],
                html
            )
        # Return None to let websockets handle WebSocket upgrade
        return None

    async def handle_client(self, websocket: WebSocketServerProtocol, path):
        """Handle client connections with path parameter"""
        logger.info(f"Client connected: {websocket.remote_address} | Path: {path}")
        self.capture.subscribers.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get('action')
                    
                    if action == 'START_STREAM':
                        success = await self.capture.start_stream(data.get('url'), data.get('quality', 'best'))
                        await websocket.send(json.dumps({'type': 'STATUS', 'action': 'START_STREAM', 'success': success}))
                    
                    elif action == 'STOP_STREAM':
                        self.capture.stop()
                        await websocket.send(json.dumps({'type': 'STATUS', 'action': 'STOP_STREAM', 'success': True}))
                    
                    elif action == 'PING':
                        await websocket.send(json.dumps({'type': 'PONG'}))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                    await websocket.send(json.dumps({'type': 'ERROR', 'message': 'Invalid JSON'}))
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {e}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.capture.subscribers.discard(websocket)

    async def start(self):
        # Setup graceful shutdown
        loop = asyncio.get_running_loop()
        stop = loop.create_future()
        
        def shutdown():
            logger.info("Received shutdown signal, closing...")
            self.capture.stop()
            stop.set_result(None)
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown)
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on {HOST}:{PORT}")
        
        async with websockets.serve(
            self.handle_client, 
            HOST, 
            PORT,
            process_request=self.process_request,  # Handle HTTP health checks
            ping_interval=20,
            ping_timeout=10,
            compression=None,  # Disable compression to avoid proxy issues
            subprotocols=None  # Disable subprotocol negotiation
        ) as server:
            logger.info(f"Server started successfully on ws://{HOST}:{PORT}")
            logger.info(f"Health check available at http://{HOST}:{PORT}/health")
            await stop
            server.close()
            await server.wait_closed()


async def main():
    await WebSocketServer().start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
