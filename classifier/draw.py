#!/usr/bin/env python3
#coding=utf-8
import matplotlib.pyplot as plt

accuracy = accuracy = [0.63508, 0.66633, 0.69052, 0.69052, 0.711169, 0.69052, 0.71270, 0.71169, 0.71673, 0.72984, 0.71875, 0.71875, 0.71471, 0.72077, 0.74496, 0.76714, 0.75806, 0.75403, 0.75, 0.77218, 0.76915, 0.76815, 0.74395, 0.76310, 0.77621, 0.77822, 0.75403, 0.77319, 0.76815, 0.76613, 0.76109, 0.75101, 0.77722, 0.78125, 0.78629, 0.77722, 0.76512, 0.77722, 0.77823, 0.76411, 0.78528, 0.76512, 0.75806, 0.77825, 0.77218]
# accuracy = [accuracy1[i] for i in range(0, len(accuracy1), 2)]

train_rounds = list(range(1, len(accuracy) + 1))

plt.plot(train_rounds, accuracy, marker='o')
plt.xlabel('Train Rounds')
plt.ylabel('Accuracy')
plt.title("'bert-base-chinese', 32, 512, False, 5e-6, 1e-5, Model1")
plt.axis([0, len(accuracy), 0.6, 0.9])
plt.grid(True)
plt.show()