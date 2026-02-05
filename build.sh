#!/usr/bin/env bash
# Install system dependencies
apt-get update && apt-get install -y ffmpeg
pip install -r requirements.txt