# ml_image

Projeto TinyML para o RP2040 (BitDogLab) com CNN de MNIST, inferência local e exibição de métricas no OLED SSD1306.

## Visão geral
- **PC → RP2040:** envio de imagem via USB-Serial (CDC).
- **RP2040 (core0):** inferência com TFLite Micro.
- **RP2040 (core1):** cálculo de métricas e exibição no OLED.

## Estrutura principal
- `main.cc`: firmware (inferência + métricas + OLED).
- `img.py`: script do PC para enviar imagem + label.
- `lib/ssd1306.c`, `lib/ssd1306.h`, `lib/font.h`: driver do display e fonte.
- `mnist_cnn_int8.tflite`: modelo quantizado.
- `pico-tflmicro/`: biblioteca TFLM.

## Protocolo PC → RP2040
O MCU espera **785 bytes por inferência**:
1) **784 bytes** da imagem 28x28 (uint8, 0..255, linha-a-linha).  
2) **1 byte** de label (0..9).

No PC (`img.py`), isso é enviado assim:
```python
ser.write(img.tobytes() + bytes([DIGITO]))
```

## O que aparece no OLED
O display atualiza a cada 1s e alterna a classe exibida:
- **Accuracy**: acurácia geral (todas as amostras).
- **Last T/P**: último label verdadeiro (T) e predição (P).
- **MacroP/MacroR/MacroF1**: médias macro de precision/recall/F1.
- **Class N**: classe atual exibida.
- **P/R/F1**: métricas da classe atual.

## Funções principais do firmware (`main.cc`)
- `read_exact(...)`: lê N bytes da serial USB (bloqueante).
- `argmax_u8(...)`: encontra a classe com maior score.
- `init_i2c1()`: inicializa I2C do OLED (pinos 14/15).
- `init_oled()`: inicializa SSD1306.
- `class_metrics(...)`: precision/recall/F1 por classe.
- `macro_metrics(...)`: médias macro das métricas.
- `core1_entry()`: task do core1, desenha no OLED.
- `main()`: loop de inferência (core0), atualiza métricas.

## Como compilar e gravar no Pico
1) Configure o Pico SDK (como já está no projeto).  
2) Gere o build:
```
mkdir build
cd build
cmake ..
ninja
```
3) Grave o `.uf2` na placa (copie para o drive do Pico).

## Como rodar o `img.py`
1) Instale dependências no PC:
```
python -m pip install pyserial tensorflow
```
2) Ajuste `PORT` e `DIGITO` no `img.py`.  
3) Execute:
```
python img.py
```
O script manda a imagem + label e imprime a predição do MCU.

## Observações
- Para duas aplicações não disputarem a COM, feche qualquer Serial Monitor ao rodar o `img.py`.
