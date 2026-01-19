#include <stdio.h>
#include <string.h>

#include "pico/stdlib.h"

#include "pico-tflmicro/examples/mnist_uart/mnist_cnn_int8.h"

#include "pico-tflmicro/src/tensorflow/lite/micro/micro_interpreter.h"
#include "pico-tflmicro/src/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "pico-tflmicro/src/tensorflow/lite/schema/schema_generated.h"

#define IMG_SIZE (28 * 28)

/* ==== Protótipos das funções ==== */
static int argmax_u8(const uint8_t *v, int n);
static void read_exact(uint8_t *dst, size_t n);
static void print_scores(const uint8_t *scores, int n);

/* ==== Buffers globais ==== */
static uint8_t rx_buf[IMG_SIZE];

/* CNN precisa de arena maior */
static uint8_t tensor_arena[128 * 1024];

/* ==== Implementações ==== */
static int argmax_u8(const uint8_t *v, int n) {
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

static void read_exact(uint8_t *dst, size_t n) {
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

static void print_scores(const uint8_t *scores, int n) {
    printf("Scores: ");
    for (int i = 0; i < n; i++) {
        printf("%d:%u ", i, scores[i]);
    }
    printf("\n");
}

int main() {
    stdio_init_all();
    sleep_ms(4000);
    printf("READY\n");

    const tflite::Model *model = tflite::GetModel(mnist_cnn_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch\n");
        while (1) {
            tight_loop_contents();
        }
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
        while (1) {
            tight_loop_contents();
        }
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    printf("Input type: %d, bytes: %d\n", input->type, (int)input->bytes);
    printf("Output type: %d, bytes: %d\n", output->type, (int)output->bytes);
    printf("Waiting 784 bytes per inference...\n");

    while (true) {
        read_exact(rx_buf, IMG_SIZE);

        memcpy(input->data.uint8, rx_buf, IMG_SIZE);

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke failed\n");
            putchar((char)0xFF);
            fflush(stdout);
            continue;
        }

        int pred = argmax_u8(output->data.uint8, 10);

        printf("Inference done\n");
        printf("Prediction: %d\n", pred);
        print_scores(output->data.uint8, 10);
        printf("----\n");

        /* Mantém a saída crua (1 byte) para o script no PC */
        putchar((char)pred);
        fflush(stdout);
    }
}
