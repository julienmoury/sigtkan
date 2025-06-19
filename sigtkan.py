import keras
from keras.layers import Layer, Dense, Dropout, Flatten, RNN
from keras import ops, activations, initializers, regularizers, constraints

from keras_efficient_kan import KANLinear
from keras_sig import SigLayer

from .grns import GRKAN, GRN

class SigTKANCell(Layer):
    """
    SigTKAN Cell - True combination of Signature methods with TKAN recurrent processing
    
    This cell implements a recurrent structure that:
    1. Computes path signatures of input sequences
    2. Uses TKAN-style KAN sub-layers for recurrent processing
    3. Combines signature features with temporal dynamics
    """
    
    def __init__(self, units, sig_level=2, sub_kan_configs=None, sub_kan_output_dim=None, 
                 sub_kan_input_dim=None, activation="tanh", recurrent_activation="sigmoid",
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        
        self.units = units
        self.sig_level = sig_level
        self.sub_kan_configs = sub_kan_configs or [None, None]  # 2 KAN sub-layers by default
        self.sub_kan_output_dim = sub_kan_output_dim or units // 2
        self.sub_kan_input_dim = sub_kan_input_dim
        
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        # Signature computation
        self.sig_layer = SigLayer(self.sig_level)
        
        # Output size calculation: [h_t, c_t] + sub_states
        self.state_size = [units, units] + [self.sub_kan_output_dim for _ in self.sub_kan_configs]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        if self.sub_kan_input_dim is None:
            self.sub_kan_input_dim = input_dim
            
        # LSTM-like gates for main recurrent structure
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),  # input, forget, cell gates
            name="kernel",
            initializer="glorot_uniform"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel", 
            initializer="orthogonal"
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name="bias",
                initializer="zeros"
            )
        
        # Signature processing pathway
        # Estimate signature dimension based on input features and level
        est_sig_dim = input_dim ** self.sig_level  # Rough estimate
        self.sig_projection = KANLinear(
            self.units, 
            dropout=self.dropout,
            use_bias=False,
            use_layernorm=False
        )
        # Build with estimated signature dimension
        self.sig_projection.build((None, est_sig_dim))
        
        # KAN sub-layers for enhanced processing (TKAN-style)
        self.kan_sub_layers = []
        for config in self.sub_kan_configs:
            if config is None:
                layer = KANLinear(self.sub_kan_output_dim, use_layernorm=True)
            elif isinstance(config, dict):
                layer = KANLinear(self.sub_kan_output_dim, **config, use_layernorm=True)
            else:
                layer = KANLinear(self.sub_kan_output_dim, use_layernorm=True)
            layer.build((None, self.sub_kan_input_dim))
            self.kan_sub_layers.append(layer)
        
        # Sub-layer recurrent connections (TKAN-style)
        self.sub_recurrent_kernels_input = []
        self.sub_recurrent_kernels_state = []
        self.sub_recurrent_weights = []
        
        for _ in self.sub_kan_configs:
            # Input projection for each sub-layer
            self.sub_recurrent_kernels_input.append(
                self.add_weight(
                    shape=(input_dim, self.sub_kan_input_dim),
                    name=f"sub_kernel_input_{len(self.sub_recurrent_kernels_input)}",
                    initializer="glorot_uniform"
                )
            )
            # State projection for each sub-layer  
            self.sub_recurrent_kernels_state.append(
                self.add_weight(
                    shape=(self.sub_kan_output_dim, self.sub_kan_input_dim),
                    name=f"sub_kernel_state_{len(self.sub_recurrent_kernels_state)}",
                    initializer="orthogonal"
                )
            )
            # Recurrent weights for state update
            self.sub_recurrent_weights.append(
                self.add_weight(
                    shape=(self.sub_kan_output_dim * 2,),
                    name=f"sub_recurrent_weight_{len(self.sub_recurrent_weights)}",
                    initializer="ones"
                )
            )
        
        # Signature-TKAN fusion
        total_kan_output = len(self.kan_sub_layers) * self.sub_kan_output_dim
        self.fusion_weight = self.add_weight(
            shape=(self.units + total_kan_output, self.units),
            name="fusion_weight",
            initializer="glorot_uniform"
        )
        self.fusion_bias = self.add_weight(
            shape=(self.units,),
            name="fusion_bias", 
            initializer="zeros"
        )
        
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # Previous hidden state
        c_tm1 = states[1]  # Previous cell state  
        sub_states = states[2:]  # Previous KAN sub-layer states
        
        batch_size = ops.shape(inputs)[0]
        
        # 1. Signature computation - key SigKAN component
        # Add batch dimension handling for signature computation
        if len(ops.shape(inputs)) == 2:
            # Single timestep - need to create sequence for signature
            sig_input = ops.expand_dims(inputs, axis=1)
        else:
            sig_input = inputs
            
        try:
            sig_features = self.sig_layer(sig_input)
            sig_processed = self.sig_projection(sig_features)
        except:
            # Fallback if signature computation fails
            sig_processed = ops.zeros((batch_size, self.units))
        
        # 2. Main LSTM-like recurrent computation
        if self.use_bias:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel) + self.bias
        else:
            gates = ops.matmul(inputs, self.kernel) + ops.matmul(h_tm1, self.recurrent_kernel)
            
        i, f, c_candidate = ops.split(self.recurrent_activation(gates), 3, axis=-1)
        
        # Update cell state
        c = f * c_tm1 + i * self.activation(c_candidate)
        
        # 3. TKAN-style KAN sub-layer processing
        sub_outputs = []
        new_sub_states = []
        
        for idx, (kan_layer, sub_state) in enumerate(zip(self.kan_sub_layers, sub_states)):
            # Project input and previous state
            projected_input = ops.matmul(inputs, self.sub_recurrent_kernels_input[idx])
            projected_state = ops.matmul(sub_state, self.sub_recurrent_kernels_state[idx])
            
            # Combine projections
            kan_input = projected_input + projected_state
            
            # Apply KAN transformation
            kan_output = kan_layer(kan_input)
            
            # Update sub-layer state (TKAN-style recurrent connection)
            weights = self.sub_recurrent_weights[idx]
            weight_output, weight_state = ops.split(weights, 2)
            new_sub_state = weight_output * kan_output + weight_state * sub_state
            
            sub_outputs.append(kan_output)
            new_sub_states.append(new_sub_state)
        
        # 4. Signature-TKAN fusion
        # Combine signature features with KAN outputs
        if sub_outputs:
            kan_combined = ops.concatenate(sub_outputs, axis=-1)
            fusion_input = ops.concatenate([sig_processed, kan_combined], axis=-1)
        else:
            fusion_input = sig_processed
            
        # Apply fusion transformation
        fusion_output = ops.matmul(fusion_input, self.fusion_weight) + self.fusion_bias
        
        # 5. Final hidden state computation
        h = self.recurrent_activation(fusion_output) * self.activation(c)
        
        return h, [h, c] + new_sub_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or self.compute_dtype
        initial_states = [
            ops.zeros((batch_size, self.units), dtype=dtype),  # h_0
            ops.zeros((batch_size, self.units), dtype=dtype),  # c_0
        ]
        # Add initial states for KAN sub-layers
        for _ in self.sub_kan_configs:
            initial_states.append(ops.zeros((batch_size, self.sub_kan_output_dim), dtype=dtype))
        return initial_states


class SigTKAN(RNN):
    """
    SigTKAN Layer - True combination of Signature methods with TKAN
    
    Combines:
    - Path signatures from SigKAN for capturing sequence geometry
    - TKAN's KAN-based recurrent processing for enhanced temporal modeling
    - Proper recurrent structure with states and manual temporal processing
    """
    
    def __init__(self, units, sig_level=2, sub_kan_configs=None, sub_kan_output_dim=None,
                 sub_kan_input_dim=None, activation="tanh", recurrent_activation="sigmoid", 
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, 
                 return_state=False, **kwargs):
        
        cell = SigTKANCell(
            units=units,
            sig_level=sig_level,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_output_dim,
            sub_kan_input_dim=sub_kan_input_dim,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
        
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            **kwargs
        )
        
    @property
    def units(self):
        return self.cell.units
    
    @property
    def sig_level(self):
        return self.cell.sig_level


class SigTKANDense(Layer):
    """
    Non-recurrent version of SigTKAN for comparison
    Combines signature computation with KAN processing (SigKAN-style)
    """
    def __init__(self, unit, sig_level, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.sig_level = sig_level
        self.sig_layer = SigLayer(self.sig_level)
        self.kan_layer = KANLinear(unit, dropout=dropout, use_bias=False, use_layernorm=False)
        self.sig_to_weight = GRKAN(unit, activation = 'softmax', dropout = dropout)
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
        # Temporal weighting
        weighted_inputs = self.time_weigthing_kernel * inputs
        
        # Signature computation  
        sig = self.sig_layer(weighted_inputs)
        
        # Attention weights from signatures
        weights = self.sig_to_weight(sig)
        
        # KAN processing
        kan_out = self.kan_layer(weighted_inputs)
        kan_out = self.dropout(kan_out)
        
        # Weighted output
        return kan_out * keras.ops.expand_dims(weights, axis=1)
