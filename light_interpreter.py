import pickle
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='check csgo-data style dataset')
parser.add_argument('-rp', help='the absolute path the file to be checked', type=str)
args = parser.parse_args()

root_path = args.rp

with open(root_path, 'rb') as filezin:
    loss_dict = pickle.load(filezin)

plt.subplot(111)
plt.plot(loss_dict["losses"])
plt.plot(loss_dict["losses_val"])
plt.ylabel("losses")
plt.xlabel("epochs")
plt.show()
