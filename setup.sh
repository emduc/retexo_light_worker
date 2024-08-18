#!/bin/bash
sudo yum install -y python3
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user


# dependencies
pip install -r requirements.txt
pip install --no-deps torchdata==0.7.0

# Set environment variable (if new machine type, check with ifconfig the if name)
export GLOO_SOCKET_IFNAME=ens5

# Configure AWS CLI
aws configure