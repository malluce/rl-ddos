#!/usr/bin/bash

sudo apt-get update
sudo apt-get install -y build-essential zlib1g-dev libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libgdm-dev libdb4o-cil-dev libpcap-dev

# install python from tgz
echo "===== installing python ====="
sudo apt-get install -y python3.8
sudo apt-get install -y python3-tk
sudo apt-get install -y python3-pip 

echo "===== upgrading pip ====="
pip3 install --upgrade pip

pip install -r /vagrant/requirements.txt
pip install pandas

# cd to synced folder
cd /vagrant/

echo "export PYTHONPATH=$PYTHONPATH:/vagrant/" >> ~/.bashrc

# allow to overcommit memory
sudo -s
echo 1 >/proc/sys/vm/overcommit_memory