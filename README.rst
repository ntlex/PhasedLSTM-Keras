PhasedLSTM-Keras
================

Keras implementation of Phased LSTM [https://arxiv.org/abs/1610.09513], from NIPS 2016.

To install run:
```pip install phased_lstm_keras```

Works both with Theano and Tensorflow backend (tested on Theano 0.9 and Tensorflow 1.0).

Classification performance compared to standard Keras LSTM for MNIST dataset:

.. image:: mnist_plstm_lstm_comparison_acc.png
   :height: 100px
   :width: 100px
   :alt: Accuracy [red: PLSTM, black: LSTM]
   :align: center
   
.. image:: mnist_plstm_lstm_comparison_loss.png
   :height: 100px
   :width: 100px
   :alt: Loss [red: PLSTM, black: LSTM]
   :align: center

Using the timegate with frozen weights
================

* Creating an initializer for the timegate:
    ```python
    # Opening the gate every 8 timesteps
    def timegate_init(shape, dtype=None):
        return K.constant(np.vstack((
                     np.zeros(shape[1]) + 0.8, # period
                     np.zeros(shape[1]) + 0.01, # shift
                     np.zeros(shape[1]) + 0.05)), dtype=dtype) # ratio
    ```
* Setting the `timegate_initializer` and marking the `trainable_timegame` as `False`:
    ```python
    PhasedLSTM(150, return_sequences=True, timegate_initializer=timegate_init, trainable_timegate=False)
    ```