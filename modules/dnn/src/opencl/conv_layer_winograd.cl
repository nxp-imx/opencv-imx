/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
// Copyright 2019 NXP
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef KERNEL_WINOGRAD_3X3

#define KERNEL_ARG_DTYPE float
#define TYPE_FLOAT  1
#define TYPE_HALF   2

// macros below must be kept in sync with conv_layer_spatial.cl

#if defined(FUSED_CONV_RELU)
#define ACTIVATION_RELU_FUNCTION(x, c) ((Dtype)(x) > 0 ? (Dtype)(x) : ((Dtype)(x) * (negative_slope)))
#define FUSED_ARG KERNEL_ARG_DTYPE negative_slope,
#elif defined(FUSED_CONV_PRELU)
#define ACTIVATION_RELU_FUNCTION(x, c) ((Dtype)(x) > 0 ? (Dtype)(x) : ((Dtype)(x) * (negative_slope[c])))
#define FUSED_ARG __global const KERNEL_ARG_DTYPE* negative_slope,
#elif defined(FUSED_CONV_POWER)
#define ACTIVATION_RELU_FUNCTION(x, c) pow(x, (Dtype)power)
#define FUSED_ARG KERNEL_ARG_DTYPE power,
#elif defined(FUSED_CONV_TANH)
#define ACTIVATION_RELU_FUNCTION(x, c) tanh(x)
#define FUSED_ARG
#elif defined(FUSED_CONV_RELU6)
#define ACTIVATION_RELU_FUNCTION(x, c) (clamp((Dtype)(x), (Dtype)min_value, (Dtype)max_value))
#define FUSED_ARG KERNEL_ARG_DTYPE min_value, KERNEL_ARG_DTYPE max_value,
#else
#define ACTIVATION_RELU_FUNCTION(x, c) (x)
#define FUSED_ARG
#endif

#ifdef FUSED_CONV_ELTWISE
#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_, _channel_) do { \
    const Dtype _x_ = eltwise_data[(_offset_)] + (_data_); \
    (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(_x_, _channel_); \
} while(0)
#define ELTWISE_DATA_ARG __global Dtype* eltwise_data,
#else
#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_, _channel_) do { \
    const Dtype _x_ = (_data_); \
    (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(_x_, _channel_); \
} while(0)
#define ELTWISE_DATA_ARG
#endif

#if APPLY_BIAS
#define BIAS_KERNEL_ARG __global Dtype * biases_base,
#else
#define BIAS_KERNEL_ARG
#endif

#define TRANSFORM_INPUT( output_data, input_data, input_width )             \
    Dtype4 output_data[4];                                                  \
    do {                                                                    \
        Dtype4 input_row[4];                                                \
        input_row[0] = vload4(0, input_data);                               \
        input_row[1] = vload4(0, input_data + total_input_width);           \
        input_row[2] = vload4(0, input_data                                 \
            + total_input_width + total_input_width);                       \
        input_row[3] = vload4(0, input_data                                 \
            + total_input_width + total_input_width + total_input_width);   \
                                                                            \
        Dtype4 trans_1[4];                                                  \
        trans_1[0] = input_row[0] - input_row[2];                           \
        trans_1[1] = input_row[1] + input_row[2];                           \
        trans_1[2] = input_row[2] - input_row[1];                           \
        trans_1[3] = input_row[1] - input_row[3];                           \
                                                                            \
        for (int i = 0; i < 4; ++i)                                         \
        {                                                                   \
            output_data[i].s0 = trans_1[i].s0 - trans_1[i].s2;              \
            output_data[i].s1 = trans_1[i].s1 + trans_1[i].s2;              \
            output_data[i].s2 = trans_1[i].s2 - trans_1[i].s1;              \
            output_data[i].s3 = trans_1[i].s1 - trans_1[i].s3;              \
        }                                                                   \
    } while(0)

#define TRANSFORM_KERNEL( output_data, input_data, input_width )                \
    Dtype4 output_data[4];                                                      \
    do {                                                                        \
        Dtype3 input_row[3];                                                    \
        input_row[0] = vload3(0, input_data);                                   \
        input_row[1] = vload3(0, input_data + input_width);                     \
        input_row[2] = vload3(0, input_data + input_width + input_width);       \
                                                                                \
        Dtype3 trans_1[4];                                                      \
        trans_1[0] = input_row[0];                                              \
        trans_1[3] = input_row[2];                                              \
                                                                                \
        input_row[0] *= (Dtype3)0.5;                                            \
        input_row[1] *= (Dtype3)0.5;                                            \
        input_row[2] *= (Dtype3)0.5;                                            \
                                                                                \
        trans_1[1] = input_row[0] + input_row[1] + input_row[2];                \
        trans_1[2] = input_row[0] - input_row[1] + input_row[2];                \
                                                                                \
        for (int i = 0; i < 4; ++i)                                             \
        {                                                                       \
            output_data[i].s0 = trans_1[i].s0;                                  \
            output_data[i].s3 = trans_1[i].s2;                                  \
            trans_1[i] *= (Dtype3)0.5;                                          \
            output_data[i].s1 = trans_1[i].s0 + trans_1[i].s1 + trans_1[i].s2;  \
            output_data[i].s2 = trans_1[i].s0 - trans_1[i].s1 + trans_1[i].s2;  \
        }                                                                       \
    } while(0)

#define TRANSFORM_BACK( output_data, input_data )                               \
    Dtype4 output_data;                                                         \
    do {                                                                        \
        Dtype4 trans_1[2];                                                      \
        trans_1[0].s0 = input_data[0].s0 + input_data[1].s0 + input_data[2].s0; \
        trans_1[0].s1 = input_data[0].s1 + input_data[1].s1 + input_data[2].s1; \
        trans_1[0].s2 = input_data[0].s2 + input_data[1].s2 + input_data[2].s2; \
        trans_1[0].s3 = input_data[0].s3 + input_data[1].s3 + input_data[2].s3; \
                                                                                \
        trans_1[1].s0 = input_data[1].s0 - input_data[2].s0 - input_data[3].s0; \
        trans_1[1].s1 = input_data[1].s1 - input_data[2].s1 - input_data[3].s1; \
        trans_1[1].s2 = input_data[1].s2 - input_data[2].s2 - input_data[3].s2; \
        trans_1[1].s3 = input_data[1].s3 - input_data[2].s3 - input_data[3].s3; \
                                                                                \
        output_data.s0 = trans_1[0].s0 + trans_1[0].s1 + trans_1[0].s2;         \
        output_data.s1 = trans_1[0].s1 - trans_1[0].s2 - trans_1[0].s3;         \
        output_data.s2 = trans_1[1].s0 + trans_1[1].s1 + trans_1[1].s2;         \
        output_data.s3 = trans_1[1].s1 - trans_1[1].s2 - trans_1[1].s3;         \
    } while(0)

#define MULT_TRANSFORMED( trans_image, trans_kernel ) do {      \
    for (int i = 0; i < 4; ++i)                                 \
        trans_image[i] *= trans_kernel[i];                      \
} while(0)

/*
For every input channels the following is performed:
    1) 3x3 winograd transform (* - matrix mult, x - element-wise mult, ^T - transpose):
       TRANS_I = (X_I * I) * X_I^T
       TRANS_K = (X_K * K) * X_K^T
       OUT = (X_O * (TRANS_I x TRANS_K)) * X_O^T

       Transformation matrices:
       X_I = {{1, 0, -1, 0},
        {0, 1, 1, 0},
        { 0, -1, 1, 0},
        { 0, 1, 0, -1 }}

       X_K = {{ 1, 0, 0},
        { 0.5, 0.5, 0.5 },
        { 0.5, -0.5, 0.5 },
        { 0, 0, 1 }}

       X_O = {{1, 1, 1, 0},
        {0, 1, -1, -1 }}

    2) Accumulate result
    3) Shift pointers
       (Note: input transformation is the same for every channel)
*/

// 4x4 input patch, 3x3 kernel, 2x2 output patch
__kernel void ConvolveWinograd_i4_k3_o2(
    ELTWISE_DATA_ARG
    FUSED_ARG
    __global Dtype* image_data,
    int image_offset,
    __global Dtype* kernel_data,
    int kernel_offset,
    __global Dtype* bias,
    const int bias_offset,
    __global Dtype* convolved_image_base,
    const int convolved_image_base_offset,
    const int convolved_image_offset,
    const int total_input_width,
    const int total_input_size,
    const int output_width,
    const int output_height
)
{
    const int out_z = get_global_id(2) * ZPAR;
    const int out_y = get_global_id(1);
    const int out_x = get_global_id(0);
    const int out_y_2 = out_y << 1;
    const int out_x_2 = out_x << 1;

    if ((out_x_2 <= output_width) && (out_y_2 <= output_height))
    {
        __global Dtype* convolved_image = convolved_image_base + convolved_image_base_offset;
        __global Dtype* input_data_ptr = image_data + (out_y_2 * total_input_width) + out_x_2 + image_offset;

        Dtype4 p[ZPAR];
        for (int kern = 0; kern < ZPAR; ++kern)
        {
            const int local_kernel_offset = kernel_offset + (out_z + kern) * KERNEL_SIZE * CHANNELS;
            __global Dtype * kernel_data_ptr = kernel_data + local_kernel_offset;

            p[kern] = (Dtype4)0;
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                TRANSFORM_INPUT(trans_input, input_data_ptr, total_input_width);
                TRANSFORM_KERNEL(trans_kernel, kernel_data_ptr, KERNEL_WIDTH);
                MULT_TRANSFORMED(trans_input, trans_kernel);
                TRANSFORM_BACK(out, trans_input);

                p[kern] += out;

                input_data_ptr += total_input_size;
                kernel_data_ptr += KERNEL_SIZE;
            }

            if ((out_z + kern) < OUTPUT_Z)
            {
                const int offset =
                    ((out_z + kern) * output_height * output_width)
                    + (out_y_2 * output_width)
                    + out_x_2 + convolved_image_offset;
                const int biasIdx = out_z + kern + bias_offset;
            #if APPLY_BIAS
                p[kern] += (Dtype4)(bias[biasIdx]);
            #endif
                if ((out_x_2 < output_width) && (out_y_2 < output_height))
                    ACTIVATION_FUNCTION(convolved_image, offset, p[kern].s0, biasIdx);
                if (((out_x_2 + 1) < output_width) && (out_y_2 < output_height))
                    ACTIVATION_FUNCTION(convolved_image, offset + 1, p[kern].s1, biasIdx);
                if ((out_x_2 < output_width) && ((out_y_2 + 1) < output_height))
                    ACTIVATION_FUNCTION(convolved_image, offset + output_width, p[kern].s2, biasIdx);
                if (((out_x_2 + 1) < output_width) && ((out_y_2 + 1) < output_height))
                    ACTIVATION_FUNCTION(convolved_image, offset + output_width + 1, p[kern].s3, biasIdx);
            }
        }
    }
}

#endif // KERNEL_WINOGRAD_3X3
