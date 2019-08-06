import subprocess

subprocess.call(['pip', 'install', 'numpy'])
subprocess.call(['pip', 'install', 'scipy'])
subprocess.call(['pip', 'install', 'matplotlib'])
subprocess.call(['unzip', 'mnist_dataset/mnist_train.zip', '-d', 'mnist_dataset/'])