import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer

import numpy as np
import math
from scipy.fftpack import fft, ifft

class DftTransform(Layer):

    def __init__(self, n, **kwargs):
        """
        Perform Discrete Fourier Transform (DFT) Analysis and synthesis of the input
        
        Arguments
        - n (int) : Size of the FFT.
        """
        
        self.N = n

        super(DftTransform, self).__init__(**kwargs)
        
    def build(self, input_shape):
    	# Create a trainable weight variable for this layer.
        self.Window = self.add_weight(name='Window', shape=(1, input_shape[1]), initializer='uniform', trainable=True)

        super(DftTransform, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mX, pX = self.dftAnal(x, self.Window, self.N)
        M = self.Window.get_shape().as_list()[1]
        y = self.dftSynth(mX, pX, M)
        
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape)
        #return (input_shape[0], int(self.N/2))
        
    def dftTrans(self, x):
        N = x.get_shape().as_list()[1]
        nv = K.arange(0, N)
        kvr = np.arange(0, N)
        
        X_list = None

        def conjugate(x):
            # x will be a numpy array with the contents of the placeholder below
            return np.conjugate(x)

        com = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
        for kr in kvr:
            s = K.exp(com * 2 * np.pi * kr / N * tf.cast(nv, tf.complex64))
            final_result = tf.py_func(conjugate, [s], tf.complex64)
            sum1 = K.sum(tf.cast(x, tf.complex64) * K.reshape(final_result, (-1, self.N)), axis=1)
            
            if X_list is None:
                X_list = K.expand_dims(sum1, axis=1)
            else:
                X_list = K.concatenate([X_list, K.expand_dims(sum1, axis=1)])

        X = X_list
        #X = K.reshape(X, (-1, self.N))
        
        return X

    def idft(self, x, N):
        ## Your code here
        #N = X.shape[0]
        #N = x.get_shape().as_list()[0]
        nv = np.arange(0, N)
        kvr = K.arange(0, N)
        
        y_list = None
        com = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
        for n in nv:
            s = K.exp(com * 2 * np.pi * n / N * tf.cast(kvr, tf.complex64))
            #y.append(1.0/N * K.sum(tf.cast(x, tf.complex64) * s))

            sum1 = 1.0/N * K.sum(tf.cast(x, tf.complex64) * K.expand_dims(s, axis=0), axis=1)
            
            if y_list is None:
                y_list = K.expand_dims(sum1, axis=1)
            else:
                y_list = K.concatenate([y_list, K.expand_dims(sum1, axis=1)])

        y = y_list
        
        return y

    def dftAnal(self, x, w, N):
        hN = int(N/2)                                                            # size of positive spectrum
        hM1 = int(math.floor((w.get_shape().as_list()[1]+1)/2))                  # half analysis window size by rounding
        hM2 = int(math.floor(w.get_shape().as_list()[1]/2))                      # half analysis window size by floor
        
        #fftbuffer = np.zeros(N)                                                 # initialize buffer for FFT
        #y = np.zeros(x.get_shape().as_list()[1])                                # initialize output array

        #----analysis--------
        xw = x * w                                                     # window the input sound
        
        #fftbuffer[:hM1] = xw[hM2:]                                              # zero-phase window in fftbuffer
        #fftbuffer[-hM2:] = xw[:hM2]
        hm_buffer1 = xw[:, hM2:]
        hm_zeros = K.zeros_like(xw)
        hm_buffer2 = xw[:, :hM2]
        fftbuffer = K.concatenate([hm_buffer1, hm_zeros[:, :N - (hM1+hM2)], hm_buffer2])
        
        #X = self.dftTrans(fftbuffer)                                         # compute FFT
        X = tf.spectral.fft(tf.cast(fftbuffer, tf.complex64))
        
        absX = K.abs(X[:, :hN])                                                  # compute ansolute value of positive side
        #absX[absX<np.finfo(float).eps] = np.finfo(float).eps                    # if zeros add epsilon to handle log
        absX = absX + np.finfo(float).eps
        mX = tf.cast(20 * (K.log(absX) / K.log(10.0)), tf.complex64)             # magnitude spectrum of positive frequencies in dB
          
        #pX = np.unwrap(np.angle(X[:hN]))                                        # unwrapped phase spectrum of positive frequencies
        def unwrap(x):
            return np.unwrap(x)

        def angle(x):
            return np.angle(x)

        def unwrap_angle(x):
            return np.unwrap(np.angle(x)).astype(np.complex64)

        #tempPX = tf.py_func(angle, [X[:hN]], tf.float32)
        pX = tf.py_func(unwrap_angle, [X[:, :hN]], tf.complex64)
        pX = K.reshape(pX, (-1, hN))

        return mX, pX

    def dftSynth(self, mX, pX, M):
        hN = mX.get_shape().as_list()[1]                                         # size of positive spectrum
        N = hN*2
        hM1 = int(math.floor((M+1)/2))                                           # half analysis window size by rounding
        hM2 = int(math.floor(M/2))                                               # half analysis window size by floor

        Y = np.zeros(N, dtype = complex)                                         # clean output spectrum
        #Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                                    # generate positive frequencies
        #Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1])                   # generate negative frequencies
        tens = K.ones_like(mX, dtype=tf.complex64) * 10
        nines = K.ones_like(mX, dtype=tf.complex64)[:, :-1] * 10
        
        posCom = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
        negCom = tf.complex(tf.constant([0.0]), tf.constant([-1.0]))
        
        hNY1 = 10**(mX/20) * K.exp(posCom * pX)
        hNY_zeros =  K.expand_dims(K.zeros_like(mX, dtype=tf.complex64)[:, hN-1], axis=1)
        hNY2 = 10**(mX[:, :0:-1]/20) * K.exp(negCom * pX[:, :0:-1])
        Y = K.concatenate([hNY1, hNY_zeros, hNY2])
        
        #fftbuffer = np.real(self.idft(Y))                                       # compute inverse FFT
        #fftbuffer = tf.real(self.idft(Y, self.N))
        fftbuffer = tf.real(tf.spectral.ifft(Y))
        
        #y[:hM2] = fftbuffer[-hM2:]                                              # undo zero-phase window
        #y[hM2:] = fftbuffer[:hM1]
        y1 = fftbuffer[:, -hM2:]
        #y_zeros =  K.expand_dims(K.zeros_like(mX, dtype=tf.complex64)[:, hN-1], axis=1)
        y2 = fftbuffer[:, :hM1]
        y = K.concatenate([y1, y2])

        return y


