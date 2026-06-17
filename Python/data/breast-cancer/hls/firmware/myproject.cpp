#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t dense_input[30],
    result_t layer7_out[2]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=dense_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=dense_input,layer7_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 600>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 200>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 20>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 2>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[20];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[2];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    nnet::dense<input_t, layer2_t, config2>(dense_input, layer2_out, w2, b2); // dense

    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // dense_relu

    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense_1

    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // dense_1_relu

    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // dense_2

    nnet::softmax<layer6_t, result_t, softmax_config7>(layer6_out, layer7_out); // dense_2_softmax

}

