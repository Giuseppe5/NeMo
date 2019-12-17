import brevitas.nn as quant_nn
from brevitas.core.quant import QuantType

jasper_activations = {
    "hardtanh": quant_nn.QuantHardTanh,
    "relu": quant_nn.QuantReLU,
    # "selu": quant_nn.SELU,
}

brevitas_QuantType = {
    'QuantType.INT' : QuantType.INT,
    'QuantType.FP' : QuantType.FP,
    'QuantType.BINARY': QuantType.BINARY,
    'QuantType.TERNARY': QuantType.TERNARY
}


def make_quantization_input(input_config):
    return quant_nn.QuantHardTanh(bit_width=            input_config['bit_width'],
                            min_val=              input_config['min_val'],
                            max_val=              input_config['max_val'],
                            quant_type =          brevitas_QuantType[input_config['quant_type']],
                            scaling_impl_type=    input_config['scaling_impl_type'],
                            scaling_stats_op=     input_config['scaling_stats_op'],
                            scaling_min_val=      input_config['scaling_min_val'],
                            return_quant_tensor=  False
                            )

def make_norm_scale(input_config):
    return quant_nn.QuantHardTanh(bit_width=            input_config['bit_width'],
                            min_val=              input_config['min_val'],
                            max_val=              input_config['max_val'],
                            quant_type =          brevitas_QuantType[input_config['quant_type']],
                            scaling_impl_type=    input_config['scaling_impl_type'],
                            scaling_stats_op=     input_config['scaling_stats_op'] ,
                            scaling_min_val=      input_config['scaling_min_val'],
                            return_quant_tensor=  True
                            )

def make_jasper_activation(activation, activation_config):
    brevitas_activation = jasper_activations[activation]
    return brevitas_activation(bit_width=activation_config['bit_width'],
                               max_val=activation_config['max_val'],
                               quant_type=brevitas_QuantType[activation_config['quant_type']],
                               scaling_impl_type=activation_config['scaling_impl_type'],
                               scaling_stats_op=activation_config['scaling_stats_op'],
                               scaling_min_val=activation_config['scaling_min_val'],
                               return_quant_tensor=False)


def make_quantconv1d(feat_in, classes, kernel_size, weight_config):
    return quant_nn.QuantConv1d(in_channels=feat_in, out_channels=classes, kernel_size=kernel_size,
                           bias=weight_config['bias'],
                           weight_bit_width=weight_config['weight_bit_width'],
                           weight_quant_type=brevitas_QuantType[weight_config['weight_quant_type']],
                           weight_narrow_range=weight_config['weight_narrow_range'],
                           weight_scaling_impl_type=weight_config['weight_scaling_impl_type'],
                           weight_scaling_stats_op=weight_config['weight_scaling_stats_op'],
                           weight_scaling_min_val=weight_config['weight_scaling_min_val'],
                           bias_bit_width=weight_config['bias_bit_width'],
                           bias_quant_type=brevitas_QuantType[weight_config['bias_quant_type']],
                           bias_narrow_range=weight_config['bias_narrow_range'],
                           compute_output_scale=weight_config['compute_output_scale'],
                           compute_output_bit_width=weight_config['compute_output_bit_width'],
                           return_quant_tensor=False)
