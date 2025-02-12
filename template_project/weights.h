#include "nnom.h"

#define forward model_run

#define DENSE_35_KERNEL_0 {117, 62, 22, 102, -42, -37, 74, -107, -67, -115, -107, -15, 41, 86, 115, 27, -22, -43, 27, -26, -98, -28, 64, 62, -115, -63, 66, 71, 99, 104, 62, 46, 100, 50, -78, -40, 9, 113, -12, 39, 91, 118, -79, -106, -96, 62, -18, -86, -42, 102, -89, -73, 30, -76, 76, 98, 33, 56, -86, 101, 80, 82, -110, 6, -63, -58, 23, 34, 49, 16, 66, -84, 75, 67, 98, 38, -26, -15, -35, -116, 56, 32, -32, 65, 53, 20, -61, -34, 101, -69, -87, -76, -45, 12, 81, 65, 93, -10, 13, 36, -118, -118, 43, 70, 77, -105, 29, -98, 80, 102, -40, -54, 75, 60, -105, -90, -111, -42, -51, -64, -93, 54, 84, -89, -25, -76, 61, -46, -85, 42, -81, 12, -56, 82, -106, 80, 82, 48, 47, 86, 114, 1, 118, -74, 18, 87, -77, -83, -111, -70, 14, -53, -103, 36, -58, -116, -28, 14, -7, 86, 54, -108, 45, 32, -116, -46, -47, 82, -41, -54, 84, -59, 10, 54, 111, -42, -98, 117, 79, 23, 85, 56, 78, -59, -6, -96, 8, 103, 21, 85, 31, -45}

#define DENSE_35_KERNEL_0_SHIFT (8)

#define DENSE_35_BIAS_0 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define DENSE_35_BIAS_0_SHIFT (8)


#define DENSE_36_KERNEL_0 {-127, -2, -97, -40, 120, -124, -119, 45, 84, -92, -46, -43, -15, 91, -67, -14, -67, 107, 119, -9, 109, -28, 58, -10, -32, 8, -22, 23, 113, 63, 89, -104, 42, 122, -27, -54, -25, -92, 22, 50, 127, -61, 54, 38, -28, -61, 48, -6, 91, -113, 100, -35, -30, -72, -114, 69, -89, -106, 90, 26, 71, -103, -95, 2, -40, -89, 110, 5, 12, -50, -97, -84, 73, -24, 98, 69, -88, 115, 64, -127, -14, -87, 32, 71, -116, 93, 13, 46, 54, 69, -119, -79, -115, 118, 24, -47, -7, -101, 83, -87, 6, 108, -112, -98, 18, -95, -95, 123, 68, 3, -32, -1, 57, 107, 117, 126, -99, 60, -66, 37, -111, 42, 10, 12, -11, -57, -114, 120, -46, 8, -127, 4, 98, 108, 107, 32, -78, 114, 123, 13, -9, -58, 68, -123}

#define DENSE_36_KERNEL_0_SHIFT (8)

#define DENSE_36_BIAS_0 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define DENSE_36_BIAS_0_SHIFT (8)


#define DENSE_37_KERNEL_0 {-90, 76, 46, 26, -23, 4, -18, 111, -4, -3, 126, -48, -83, 73, 124, 51, 117, -111, 96, 48, -18, -23, -71, -116, -121, 116, -41, -121, -76, 28, -12, -124, 116, 36, 79, -126, -73, 83, 49, -111, -50, 75, 34, 71, -67, 32, -77, 109, 22, -63, 93, -24, -54, 118, -1, -98, 47, 75, -107, 24, 93, 57, -32, 63, 105, -67, -36, -13, -107, 24, 49, 46, 57, -19, -38, 3, -83, 75, 44, 101, -113, -109, 112, -125, 22, -73, -5, 92, 43, -16, -13, -120, 81, -3, 100, 47, -56, -62, -107, 116, 43, 63, -27, 57, -110, -127, 40, -126, -12, -24, 80, 46, 68, 88, 116, 40, 120, 84, -111, -64, -24, -128, 126, -115, 107, 71, -78, 30, 112, 109, 24, -44, 43, -2, 1, 81, -3, 89, -9, 58, -72, -96, 63, 53}

#define DENSE_37_KERNEL_0_SHIFT (8)

#define DENSE_37_BIAS_0 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

#define DENSE_37_BIAS_0_SHIFT (8)



/* output encoding for each layer */
#define INPUT_176_OUTPUT_SHIFT 7
#define DENSE_35_OUTPUT_SHIFT 6
#define ACTIVATION_396_OUTPUT_SHIFT 6
#define DENSE_36_OUTPUT_SHIFT 6
#define ACTIVATION_397_OUTPUT_SHIFT 6
#define DENSE_37_OUTPUT_SHIFT 7
#define ACTIVATION_398_OUTPUT_SHIFT 7

/* bias shift and output shift for each layer */
#define DENSE_35_OUTPUT_RSHIFT (INPUT_176_OUTPUT_SHIFT+DENSE_35_KERNEL_0_SHIFT-DENSE_35_OUTPUT_SHIFT)
#define DENSE_35_BIAS_LSHIFT   (INPUT_176_OUTPUT_SHIFT+DENSE_35_KERNEL_0_SHIFT-DENSE_35_BIAS_0_SHIFT)
#if DENSE_35_OUTPUT_RSHIFT < 0
#error DENSE_35_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_35_BIAS_LSHIFT < 0
#error DENSE_35_BIAS_RSHIFT must be bigger than 0
#endif
#define DENSE_36_OUTPUT_RSHIFT (ACTIVATION_396_OUTPUT_SHIFT+DENSE_36_KERNEL_0_SHIFT-DENSE_36_OUTPUT_SHIFT)
#define DENSE_36_BIAS_LSHIFT   (ACTIVATION_396_OUTPUT_SHIFT+DENSE_36_KERNEL_0_SHIFT-DENSE_36_BIAS_0_SHIFT)
#if DENSE_36_OUTPUT_RSHIFT < 0
#error DENSE_36_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_36_BIAS_LSHIFT < 0
#error DENSE_36_BIAS_RSHIFT must be bigger than 0
#endif
#define DENSE_37_OUTPUT_RSHIFT (ACTIVATION_397_OUTPUT_SHIFT+DENSE_37_KERNEL_0_SHIFT-DENSE_37_OUTPUT_SHIFT)
#define DENSE_37_BIAS_LSHIFT   (ACTIVATION_397_OUTPUT_SHIFT+DENSE_37_KERNEL_0_SHIFT-DENSE_37_BIAS_0_SHIFT)
#if DENSE_37_OUTPUT_RSHIFT < 0
#error DENSE_37_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_37_BIAS_LSHIFT < 0
#error DENSE_37_BIAS_RSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t dense_35_weights[] = DENSE_35_KERNEL_0;
static const nnom_weight_t dense_35_w = { (const void*)dense_35_weights, DENSE_35_OUTPUT_RSHIFT};
static const int8_t dense_35_bias[] = DENSE_35_BIAS_0;
static const nnom_bias_t dense_35_b = { (const void*)dense_35_bias, DENSE_35_BIAS_LSHIFT};
static const int8_t dense_36_weights[] = DENSE_36_KERNEL_0;
static const nnom_weight_t dense_36_w = { (const void*)dense_36_weights, DENSE_36_OUTPUT_RSHIFT};
static const int8_t dense_36_bias[] = DENSE_36_BIAS_0;
static const nnom_bias_t dense_36_b = { (const void*)dense_36_bias, DENSE_36_BIAS_LSHIFT};
static const int8_t dense_37_weights[] = DENSE_37_KERNEL_0;
static const nnom_weight_t dense_37_w = { (const void*)dense_37_weights, DENSE_37_OUTPUT_RSHIFT};
static const int8_t dense_37_bias[] = DENSE_37_BIAS_0;
static const nnom_bias_t dense_37_b = { (const void*)dense_37_bias, DENSE_37_BIAS_LSHIFT};

/* nnom model */
const int8_t NUM_INPUTS = 1;
const int8_t ALL_INPUTS_SIZE = 16;
const int8_t INPUT_LENGTHS[] = {16};
static int8_t nnom_input_data[ALL_INPUTS_SIZE];
static int8_t* nnom_input_1 = nnom_input_data;
const int8_t NUM_OUTPUTS = 1;
const int8_t ALL_OUTPUTS_SIZE = 12;
const int8_t OUTPUT_LENGTHS[] = {12};
static int8_t nnom_output_data[ALL_OUTPUTS_SIZE];
static int8_t* nnom_output_1 = nnom_output_data;
static int8_t* send_data = nnom_output_1;
static int8_t SEND_LENGTH = OUTPUT_LENGTHS[0];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[8];

	new_model(&model);
	layer[0] = Input(shape(16, 1, 1), nnom_input_1);
	layer[1] = model.hook(Dense(12, &dense_35_w, &dense_35_b), layer[0]);
	layer[2] = model.active(act_relu(), layer[1]);
	layer[3] = model.hook(Dense(12, &dense_36_w, &dense_36_b), layer[2]);
	layer[4] = model.active(act_relu(), layer[3]);
	layer[5] = model.hook(Dense(12, &dense_37_w, &dense_37_b), layer[4]);
	layer[6] = model.active(act_relu(), layer[5]);
	layer[7] = model.hook(Output(shape(12, 1, 1), nnom_output_1), layer[6]);
	model_compile(&model, layer[0], layer[7]);
	return &model;
}
