import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    install('mplfinance')
    install('pandas_ta')
    install('polygon-api-client')
    install('finnhub-python')
    install('tensorflow_probability')
    install('alpaca-py')