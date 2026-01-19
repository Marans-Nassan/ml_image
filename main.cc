#include <stdio.h>
#include <string.h>

#include "lib/ssd1306.h"
#include "lib/font.h"

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/sync.h"
#include "hardware/i2c.h"

#include "pico-tflmicro/examples/mnist_uart/mnist_cnn_int8.h"

#include "pico-tflmicro/src/tensorflow/lite/micro/micro_interpreter.h"
#include "pico-tflmicro/src/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "pico-tflmicro/src/tensorflow/lite/schema/schema_generated.h"

// Entrada: imagem 28x28 (784 bytes) + 1 byte de label.
#define IMG_SIZE (28 * 28)
#define LABEL_SIZE 1
#define RX_SIZE (IMG_SIZE + LABEL_SIZE)

// Pinos I2C do OLED na BitDogLab.
#define I2C_PORT_B i2c1
#define I2C_SDA_B 14
#define I2C_SCL_B 15
#define I2C_ADDRESS 0x3c

/* ==== Protótipos das funções ==== */
static int argmax_u8(const uint8_t *v, int n);
static void read_exact(uint8_t *dst, size_t n);
static void print_scores(const uint8_t *scores, int n);
static void init_i2c1(void);
static void init_oled(void);
static void core1_entry(void);

/* ==== Buffers globais ==== */
static uint8_t rx_buf[IMG_SIZE];

/* CNN precisa de arena maior */
static uint8_t tensor_arena[128 * 1024];

// Métricas acumuladas no MCU para exibir no display.
typedef struct {
    uint32_t confusion[10][10];
    uint32_t total;
    uint32_t correct;
    uint8_t last_pred;
    uint8_t last_true;
    bool has_last;
} metrics_t;

static ssd1306_t ssd;
static metrics_t metrics;
static mutex_t data_mutex;

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

// Lê exatamente n bytes da USB CDC (bloqueante).
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

// Debug: imprime os scores brutos.
static void print_scores(const uint8_t *scores, int n) {
    printf("Scores: ");
    for (int i = 0; i < n; i++) {
        printf("%d:%u ", i, scores[i]);
    }
    printf("\n");
}

// Inicializa o I2C usado pelo OLED.
static void init_i2c1(void) {
    i2c_init(I2C_PORT_B, 400 * 1000);
    gpio_set_function(I2C_SDA_B, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_B, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_B);
    gpio_pull_up(I2C_SCL_B);
}

// Inicializa o display OLED.
static void init_oled(void) {
    ssd1306_init(&ssd, WIDTH, HEIGHT, false, I2C_ADDRESS, I2C_PORT_B);
    ssd1306_config(&ssd);
    ssd1306_fill(&ssd, false);
    ssd1306_send_data(&ssd);
}

// Precision/recall/F1 por classe a partir da matriz de confusão.
static void class_metrics(const metrics_t *m, int cls, float *p, float *r, float *f1) {
    uint32_t row_sum = 0;
    uint32_t col_sum = 0;
    for (int i = 0; i < 10; i++) {
        row_sum += m->confusion[cls][i];
        col_sum += m->confusion[i][cls];
    }
    uint32_t tp = m->confusion[cls][cls];
    uint32_t fp = col_sum - tp;
    uint32_t fn = row_sum - tp;

    float precision = (tp + fp) ? (float)tp / (float)(tp + fp) : 0.0f;
    float recall = (tp + fn) ? (float)tp / (float)(tp + fn) : 0.0f;
    float f1_score = (precision + recall) ? (2.0f * precision * recall) / (precision + recall) : 0.0f;

    *p = precision;
    *r = recall;
    *f1 = f1_score;
}

// Média macro de precision/recall/F1 entre todas as classes.
static void macro_metrics(const metrics_t *m, float *p, float *r, float *f1) {
    float sp = 0.0f;
    float sr = 0.0f;
    float sf = 0.0f;
    for (int i = 0; i < 10; i++) {
        float cp, cr, cf;
        class_metrics(m, i, &cp, &cr, &cf);
        sp += cp;
        sr += cr;
        sf += cf;
    }
    *p = sp / 10.0f;
    *r = sr / 10.0f;
    *f1 = sf / 10.0f;
}

// Tarefa do core1: renderiza métricas no OLED (alternando classes).
static void core1_entry(void) {
    sleep_ms(500);
    init_i2c1();
    init_oled();

    int cls = 0;
    while (true) {
        metrics_t snapshot;
        mutex_enter_blocking(&data_mutex);
        snapshot = metrics;
        mutex_exit(&data_mutex);

        float acc = snapshot.total ? (float)snapshot.correct / (float)snapshot.total : 0.0f;
        float mp, mr, mf;
        macro_metrics(&snapshot, &mp, &mr, &mf);

        float cp, cr, cf;
        class_metrics(&snapshot, cls, &cp, &cr, &cf);

        char line[17];
        ssd1306_fill(&ssd, false);

        snprintf(line, sizeof(line), "Accuracy %4.1f%%", acc * 100.0f);
        ssd1306_draw_string(&ssd, line, 0, 0);

        if (snapshot.has_last) {
            snprintf(line, sizeof(line), "Last T=%d P=%d", snapshot.last_true, snapshot.last_pred);
        } else {
            snprintf(line, sizeof(line), "Last T=-- P=--");
        }
        ssd1306_draw_string(&ssd, line, 0, 8);

        snprintf(line, sizeof(line), "MacroP %0.2f", mp);
        ssd1306_draw_string(&ssd, line, 0, 16);
        snprintf(line, sizeof(line), "MacroR %0.2f", mr);
        ssd1306_draw_string(&ssd, line, 0, 24);
        snprintf(line, sizeof(line), "MacroF1 %0.2f", mf);
        ssd1306_draw_string(&ssd, line, 0, 32);

        snprintf(line, sizeof(line), "Class %d", cls);
        ssd1306_draw_string(&ssd, line, 0, 40);
        snprintf(line, sizeof(line), "P=%0.2f R=%0.2f", cp, cr);
        ssd1306_draw_string(&ssd, line, 0, 48);
        snprintf(line, sizeof(line), "F1=%0.2f", cf);
        ssd1306_draw_string(&ssd, line, 0, 56);

        ssd1306_send_data(&ssd);

        cls = (cls + 1) % 10;
        sleep_ms(1000);
    }
}

int main() {
    // Inicializa serial USB.
    stdio_init_all();
    sleep_ms(4000);
    printf("READY\n");

    // Métricas compartilhadas com o core1.
    mutex_init(&data_mutex);
    memset(&metrics, 0, sizeof(metrics));
    multicore_launch_core1(core1_entry);

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
    printf("Waiting 784 bytes + label per inference...\n");

    while (true) {
        uint8_t label = 0xFF;
        // Recebe imagem + label do PC.
        read_exact(rx_buf, IMG_SIZE);
        read_exact(&label, LABEL_SIZE);

        memcpy(input->data.uint8, rx_buf, IMG_SIZE);

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke failed\n");
            putchar((char)0xFF);
            fflush(stdout);
            continue;
        }

        int pred = argmax_u8(output->data.uint8, 10);

        // Atualiza métricas apenas se o label for válido (0..9).
        if (label < 10) {
            mutex_enter_blocking(&data_mutex);
            metrics.total++;
            if ((uint8_t)pred == label) {
                metrics.correct++;
            }
            metrics.confusion[label][pred]++;
            metrics.last_pred = (uint8_t)pred;
            metrics.last_true = label;
            metrics.has_last = true;
            mutex_exit(&data_mutex);
        }

        printf("Inference done\n");
        printf("Prediction: %d\n", pred);
        print_scores(output->data.uint8, 10);
        printf("----\n");

        /* Mantém a saída crua (1 byte) para o script no PC */
        putchar((char)pred);
        fflush(stdout);
    }
}
