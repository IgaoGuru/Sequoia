import pickle
import argparse
import matplotlib.pyplot as plt
from numpy import mean as npmean


parser = argparse.ArgumentParser(description='check csgo-data style dataset')
parser.add_argument('-rp', help='the absolute path the file to be checked', type=str)
args = parser.parse_args()

root_path = args.rp

with open(root_path, 'rb') as filezin:
    loss_dict = pickle.load(filezin)

print(f"trained for {loss_dict['num_epochs']} epochs.")
print(f"learning rate: {loss_dict['lr']}")
print("\n")
print(f"median train loss: {npmean(loss_dict['losses'])}")
print(f"median train accuracy: {npmean(loss_dict['accuracies'])}")
print(f"median validation loss: {npmean(loss_dict['losses_val'])}")
print(f"median validation accuracy: {npmean(loss_dict['accuracies_val'])}")
print(loss_dict.keys())

plt.subplot(111)
plt.plot(loss_dict["losses"])
plt.plot(loss_dict["losses_val"])
plt.ylabel("losses")
plt.xlabel("epochs")
plt.show()
