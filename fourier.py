import numpy as np
import matplotlib.pyplot as plt
# def f(x):
#     # return np.sin(x)  + (((x + 0.5) ** 1.2) % 3) * 0.5
#     return x ** 2

# num_t = 100
# x = np.linspace(0, 10, num_t)
# y = f(x)

# plt.plot(x, y)
# plt.show()


# fy = np.fft.fft(y)
# plt.subplot(1,2,1)
# plt.plot(np.fft.fftshift(np.real(fy)))
# plt.title("Real part of Fourier Transform")
# plt.subplot(1,2,2)
# plt.plot(np.fft.fftshift(np.imag(fy)))
# plt.title("Imaginary part of Fourier Transform")
# plt.show()

num_graphs = 25

num_sin_waves = 10000
graph_x_amount = 1000
graph_range = np.linspace(0, 1, graph_x_amount)

def forward_fourier(components, num_t=graph_x_amount, start=0, end=1):
    x = np.linspace(start, end, num_t)
    s_weight, s_bias, bias = components
    y = bias
    for i in range(len(s_weight)):
        weighted_i = i / len(s_weight)
        y += s_weight[i] * np.sin(2 * np.pi * (x * weighted_i + s_bias[i]))
    return y

def generate_f(num_f):
    s_weight = np.random.rand(num_f) * 2 - 1
    s_bias = np.random.rand(num_f)
    bias = np.random.rand()
    return s_weight, s_bias, bias


# fig, axs = plt.subplots(num_graphs, 2)
fig, axs = plt.subplots(5, 5)

for i in range(num_graphs):
    f = generate_f(num_sin_waves)
    # axs[i, 0].scatter(f[:,0], f[:,1])
    # axs[i, 1].plot(graph_range, forward_fourier(f))
    axs[i//5, i%5].plot(graph_range, forward_fourier(f))



plt.show()

