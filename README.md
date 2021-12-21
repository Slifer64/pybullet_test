# Pybullet examples

## Create a virtual env and install pip
```bash
python3 -m venv --without-pip ./env
source env/bin/activate
curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
python get-pip.py pip
#rm get-pip.py
pip install -e .

sudo apt install --fix-broken 
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

## Source virtual env
```bash
source env/bin/activate
```