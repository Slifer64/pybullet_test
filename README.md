# Pybullet examples

## Create a virtual env and install pip
```bash
python3 -m venv --without-pip ./env
source env/bin/activate
curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
python get-pip.py pip
#rm get-pip.py
pip install -e .
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

sudo apt install --fix-broken 
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

## Source virtual env
```bash
source env/bin/activate
```