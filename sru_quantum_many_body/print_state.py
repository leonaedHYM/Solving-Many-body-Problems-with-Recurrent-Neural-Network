import numpy as np



def printf(f, Ns):
    form = '{0:0'+str(Ns)+'b}'
    states = np.load(f).squeeze()
    ini = ""
    for i in range(2**Ns):
        ini += '+({:.3f})+j*({:.3f})\\blue{{$\ket{{{}}}$}} \\\\\n'.format(states[0][i],
                                                                        states[0][i+2**Ns],
                                                                form.format(i))
    print(ini)


if __name__ == '__main__':
    Ns = 5
    f = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns={}/prediction_500.npy'.format(Ns)
    printf(f, Ns)
