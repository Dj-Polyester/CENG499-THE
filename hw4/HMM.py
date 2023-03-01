import numpy as np


class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def forward_log(self, O: list):
        T = len(O)
        N = len(self.A)
        loga = np.zeros((T, N), dtype=np.float128)
        logA = np.log(self.A)
        logB = np.log(self.B)
        loga[0] = np.log(self.Pi)+logB[:, O[0]]
        for t in range(1, T):
            oldLoga2dColWise = np.meshgrid(loga[t-1], loga[t-1])[1]
            oldLoga2dColWisePlusLogA = oldLoga2dColWise + logA
            loga[t] = np.log((np.e**oldLoga2dColWisePlusLogA).sum(axis=0)) + \
                logB[:, O[t]]
        return np.log((np.e**loga[T-1]).sum())

    def viterbi_log(self, O: list):
        T = len(O)
        N = len(self.A)

        logdelta = np.zeros((T, N), dtype=np.float128)
        sai = np.zeros((T, N), dtype=np.float128)
        logA = np.log(self.A)
        logB = np.log(self.B)
        logdelta[0] = np.log(self.Pi)+logB[:, O[0]]
        for t in range(1, T):
            oldLogDelta2dColWise = np.meshgrid(logdelta[t-1], logdelta[t-1])[1]
            oldLogDelta2dColWisePlusLogA = oldLogDelta2dColWise + logA

            logdelta[t] = oldLogDelta2dColWisePlusLogA.max(
                axis=0) + logB[:, O[t]]
            sai[t] = oldLogDelta2dColWisePlusLogA.argmax(axis=0)

        maxdelta = logdelta[T-1].max()

        mostLikelyStates = np.zeros(T, dtype=np.int32)
        mostLikelyStates[T-1] = logdelta[T-1].argmax()
        for i in reversed(range(T-1)):
            mostLikelyStates[i] = sai[i+1, mostLikelyStates[i+1]]
        return (maxdelta, list(mostLikelyStates))
