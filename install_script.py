import subprocess
import sys
import pip

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def _install(package):
    subprocess.check_call([sys.executable, "-U", "pip", "install", package])

if __name__ == '__main__':
    install('mplfinance')
    install('pandas_ta')
    install('polygon-api-client')
    install('finnhub-python')
    install('tensorflow_probability')
    install('alpaca-py')

    pip.main(['install', '--upgrade', 'tensorflow'])
    pip.main(['install', '-U', 'ray[rllib]==2.3.0'])