#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"

#include "pico-tflmicro/examples/mnist_uart/mnist_cnn_int8.h"

#include "pico-tflmicro/src/tensorflow/lite/micro/micro_interpreter.h"
#include "pico-tflmicro/src/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "pico-tflmicro/src/tensorflow/lite/schema/schema_generated.h"

#define IMG_SIZE (28 * 28)

static uint8_t rx_buf[IMG_SIZE];

// CNN precisa de arena maior
static uint8_t tensor_arena[128 * 1024];

static int argmax_u8(const uint8_t *v, int n)
{
    int best = 0;
    uint8_t best_val = v[0];

    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

static void read_exact(uint8_t *dst, size_t n)
{
    size_t i = 0;
    while (i < n) {
        int c = getchar_timeout_us(0);
        if (c == PICO_ERROR_TIMEOUT) {
            tight_loop_contents();
            continue;
        }
        dst[i++] = (uint8_t)c;
    }
}

int main()
{
    stdio_init_all();
    sleep_ms(1500);
    printf("READY\n");

    const tflite::Model *model = tflite::GetModel(mnist_cnn_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch\n");
        while (1) {}
    }

    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        sizeof(tensor_arena)
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        while (1) {}
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    while (true) {
        // Recebe 784 bytes (28x28)
        read_exact(rx_buf, IMG_SIZE);

        // Copia para o tensor [1,28,28,1]
        memcpy(input->data.uint8, rx_buf, IMG_SIZE);

        if (interpreter.Invoke() != kTfLiteOk) {
            putchar((char)0xFF);
            fflush(stdout);
            continue;
        }

        int pred = argmax_u8(output->data.uint8, 10);
        putchar((char)pred);
        fflush(stdout);
    }
}
