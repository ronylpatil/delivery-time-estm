echo [$(date)]: "START"
echo [$(date)]: "Creating virtual env with python 3.9"
python -m virtualenv ./venv

echo [$(date)]: "activate venv"
source ./venv/Scripts/activate

echo [$(date)]: "upgrading pip and setuptools"
pip install --upgrade pip setuptools

echo [$(date)]: "installing dev requirements"
pip install -r requirements.txt

echo [$(date)]: "END"

# ye init setup likne k baad just "bash init_setup.sh" cmd run kr do