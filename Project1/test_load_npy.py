import numpy as np

def main():
    data = np.load("hw1.npy", allow_pickle=True).item()

    print("Available keys in hw1.npy:")
    print(data.keys())

    for key in data:
        print(f"{key}: type={type(data[key])}, shape={np.shape(data[key])}")


main()