## PhasedLSTM-Keras

Keras implementation of Phased LSTM [https://arxiv.org/abs/1610.09513], from NIPS 2016.

This is an extension to the [fferroni/PhasedLSTM-Keras](https://github.com/fferroni/PhasedLSTM-Keras), to allow switching off the training of the timegate and enable it to work on fixed weights for the *shift*, *period* and *ratio* parameters.

### Usage

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