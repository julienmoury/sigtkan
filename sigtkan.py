import keras
from keras.layers import Layer, Dense, Dropout, Flatten
from keras import ops

from keras_efficient_kan import KANLinear
from keras_sig import SigLayer

from .grns import GRKAN, GRN

class SigTKANCell(Layer):
    """
    SigTKAN Cell - Manual implementation combining signatures with TKAN-like functionality
    This is a custom implementation for educational purposes
    """
    def __init__(self, units, sig_level=2, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        
        # Signature layer for path signature computation
        self.sig_layer = SigLayer(self.sig_level)
        
        # KAN layer for non-linear approximation
        self.kan_layer = KANLinear(units, dropout=dropout, use_bias=False, use_layernorm=False)
        
        # Signature to attention weight conversion
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        
        # LSTM-style gates for recurrent processing
        self.input_gate = Dense(units, activation='sigmoid', name='input_gate')
        self.forget_gate = Dense(units, activation='sigmoid', name='forget_gate')
        self.cell_gate = Dense(units, activation='tanh', name='cell_gate')
        self.output_gate = Dense(units, activation='sigmoid', name='output_gate')
        
        # Recurrent connections
        self.recurrent_input = Dense(units, use_bias=False, name='recurrent_input')
        self.recurrent_forget = Dense(units, use_bias=False, name='recurrent_forget')
        self.recurrent_cell = Dense(units, use_bias=False, name='recurrent_cell')
        self.recurrent_output = Dense(units, use_bias=False, name='recurrent_output')
        
        self.dropout_layer = Dropout(dropout)

    def build(self, input_shape):
        # Note: input_shape ici peut varier selon le timestep dans la manual loop
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        """
        Manual cell call - processes one timestep
        inputs: current input at timestep t
        states: [hidden_state, cell_state] from previous timestep
        """
        h_prev, c_prev = states
        
        # Compute signature features from input sequence
        # inputs shape: (batch, seq_len, features)
        sig_features = self.sig_layer(inputs)
        
        # Get attention weights from signature
        attention_weights = self.sig_to_weight(sig_features)
        
        # Apply KAN transformation with signature attention
        kan_output = self.kan_layer(inputs)
        attended_kan = kan_output * ops.expand_dims(attention_weights, axis=-1)
        
        # Aggregate over sequence dimension to get current input
        current_input = ops.mean(attended_kan, axis=1)  # (batch, units)
        current_input = self.dropout_layer(current_input, training=training)
        
        # LSTM-style gate computations
        # Combine current input with previous hidden state
        i = self.input_gate(current_input) + self.recurrent_input(h_prev)
        f = self.forget_gate(current_input) + self.recurrent_forget(h_prev)
        c_candidate = self.cell_gate(current_input) + self.recurrent_cell(h_prev)
        o = self.output_gate(current_input) + self.recurrent_output(h_prev)
        
        # Apply activations
        i = ops.sigmoid(i)
        f = ops.sigmoid(f)
        c_candidate = ops.tanh(c_candidate)
        o = ops.sigmoid(o)
        
        # Update cell state
        c_new = f * c_prev + i * c_candidate
        
        # Update hidden state
        h_new = o * ops.tanh(c_new)
        
        return h_new, [h_new, c_new]

class SigTKAN(Layer):
    """
    SigTKAN Layer with manual RNN loop implementation
    This manually implements the recurrent loop without inheriting from keras.layers.RNN
    """
    def __init__(self, units, sig_level=2, dropout=0.0, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        
        # Create the recurrent cell
        self.cell = SigTKANCell(units, sig_level, dropout)

    def build(self, input_shape):
        # Build the cell with the input shape
        self.cell.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """
        Manual RNN loop implementation
        This is our custom implementation of the recurrent computation
        """
        # Get input dimensions
        batch_size = ops.shape(inputs)[0]
        seq_length = ops.shape(inputs)[1]
        
        # Initialize states - this is how we manually initialize RNN states
        h_state = ops.zeros((batch_size, self.units), dtype=inputs.dtype)
        c_state = ops.zeros((batch_size, self.units), dtype=inputs.dtype)
        states = [h_state, c_state]
        
        # Prepare outputs list if returning sequences
        if self.return_sequences:
            outputs = []
        
        # Manual recurrent loop - this is the core of our custom RNN implementation
        for t in range(seq_length):
            # Prepare input for current timestep
            # For signature computation, we need the sequence up to current timestep
            if t == 0:
                # For first timestep, we need at least 2 points for signature
                current_sequence = ops.expand_dims(inputs[:, 0, :], axis=1)
                current_sequence = ops.concatenate([current_sequence, current_sequence], axis=1)
            else:
                # Use sequence from start to current timestep
                current_sequence = inputs[:, :t+1, :]
            
            # Apply mask if provided (standard RNN masking)
            if mask is not None:
                current_mask = mask[:, t]
                # Apply masking to states (this is how Keras handles masking)
                current_mask = ops.cast(current_mask, dtype=inputs.dtype)
                current_mask = ops.expand_dims(current_mask, axis=-1)
            else:
                current_mask = None
            
            # Call the cell for current timestep
            output, states = self.cell(current_sequence, states, training=training)
            
            # Apply masking to output and states if needed
            if current_mask is not None:
                output = output * current_mask
                states = [state * current_mask for state in states]
            
            # Store output if returning sequences
            if self.return_sequences:
                outputs.append(output)
        
        # Return appropriate format
        if self.return_sequences:
            # Stack all timestep outputs
            return ops.stack(outputs, axis=1)
        else:
            # Return only final output
            return output

    def compute_output_shape(self, input_shape):
        """
        Compute output shape for the layer
        """
        batch_size, seq_length, input_dim = input_shape
        if self.return_sequences:
            return (batch_size, seq_length, self.units)
        else:
            return (batch_size, self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
            "return_sequences": self.return_sequences,
        })
        return config

class SigTKANDense(Layer):
    """
    Dense version of SigTKAN for comparison
    """
    def __init__(self, unit, sig_level, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.dense_layer = Dense(unit)
        self.sig_to_weight = GRN(unit, activation = 'softmax', dropout = dropout)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, n_features = input_shape
        name = self.name
        self.time_weigthing_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weigthing_kernel",
        )
        super().build(input_shape)
        
    def call(self, inputs):
        inputs = self.time_weigthing_kernel * inputs
        sig = self.sig_layer(inputs)
        weights = self.sig_to_weight(sig)
        dense_out = self.dense_layer(inputs)
        dense_out = self.dropout(dense_out)
        return dense_out * ops.expand_dims(weights, axis=1)
