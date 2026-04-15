#!/usr/bin/env python3
"""Download and create the ragrag validation corpus.

Run: python scripts/fetch_validation_data.py

Idempotent: skips files that already exist with correct size.
"""
from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "validation" / "fixtures"
PDFS_DIR = FIXTURES / "pdfs"
TEXT_DIR = FIXTURES / "text"
IMAGES_DIR = FIXTURES / "images"
UNSUPPORTED_DIR = FIXTURES / "unsupported"


PDF_DOWNLOADS = [
    {
        "url": "https://files.zubax.com/products/com.zubax.fluxgrip/FluxGrip_FG40_datasheet.pdf",
        "filename": "FluxGrip_FG40_datasheet.pdf",
        "description": "FluxGrip FG40 motor controller datasheet (specified in DESIGN.md §17)",
        "min_size_bytes": 100_000,
    },
    {
        "url": "https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf",
        "filename": "esp32_datasheet_en.pdf",
        "description": "ESP32 datasheet with block diagrams, clock tree, pin diagrams",
        "min_size_bytes": 1_000_000,
    },
    {
        "url": "https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf",
        "filename": "rp2040-datasheet.pdf",
        "description": "RP2040 datasheet with PIO state machine diagrams, DMA ring buffers",
        "min_size_bytes": 1_000_000,
    },
    {
        "url": "https://www.st.com/resource/en/datasheet/stm32h743vi.pdf",
        "filename": "stm32h743vi.pdf",
        "description": "STM32H743VI datasheet — used by scripts/benchmark_stm32h743.py as the quality-regression corpus",
        "min_size_bytes": 5_000_000,
    },
]


def download_file(url: str, dest: Path, min_size: int = 0) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size >= max(min_size, 1):
        print(f"  SKIP (exists): {dest.name}")
        return False

    print(f"  Downloading: {dest.name}")
    print(f"    from: {url}")
    try:
        headers = {"User-Agent": "ragrag-validation/1.0 (research use)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(resp.read())
        size = dest.stat().st_size
        print(f"    OK: {size:,} bytes")
        if min_size and size < min_size:
            print(f"    WARNING: file smaller than expected ({size} < {min_size})")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def create_synthetic_text_files() -> None:
    """Create synthetic embedded development text fixtures."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # C source file — GPIO driver
    c_file = TEXT_DIR / "stm32_gpio_driver.c"
    if not c_file.exists():
        c_file.write_text("""\
/**
 * @file stm32_gpio_driver.c
 * @brief STM32 GPIO driver implementation using HAL-style register access.
 *
 * Supports GPIO initialization, mode configuration, and interrupt setup.
 * Designed for STM32F4 family with GPIOA-GPIOK peripheral support.
 */
#include "stm32_gpio_driver.h"
#include <stdint.h>

/* GPIO mode register bit positions */
#define GPIO_MODE_POS(pin)   ((pin) * 2U)
#define GPIO_PUPD_POS(pin)   ((pin) * 2U)
#define GPIO_OSPEED_POS(pin) ((pin) * 2U)

/**
 * @brief Initialize a GPIO pin with the specified configuration.
 *
 * Configures the pin mode (input/output/alternate function/analog),
 * output type (push-pull/open-drain), output speed, and pull-up/pull-down.
 *
 * @param gpio   Pointer to GPIO peripheral register block (e.g. GPIOA)
 * @param config Pointer to GPIO configuration structure
 */
void GPIO_Init(GPIO_TypeDef *gpio, const GPIO_Config_t *config) {
    uint32_t pin = config->pin;

    /* Configure pin mode */
    gpio->MODER &= ~(0x3U << GPIO_MODE_POS(pin));
    gpio->MODER |= (config->mode << GPIO_MODE_POS(pin));

    /* Configure output type (only for output/alternate function modes) */
    if (config->mode == GPIO_MODE_OUTPUT || config->mode == GPIO_MODE_AF) {
        gpio->OTYPER &= ~(1U << pin);
        gpio->OTYPER |= (config->otype << pin);
    }

    /* Configure output speed */
    gpio->OSPEEDR &= ~(0x3U << GPIO_OSPEED_POS(pin));
    gpio->OSPEEDR |= (config->speed << GPIO_OSPEED_POS(pin));

    /* Configure pull-up/pull-down resistors */
    gpio->PUPDR &= ~(0x3U << GPIO_PUPD_POS(pin));
    gpio->PUPDR |= (config->pull << GPIO_PUPD_POS(pin));

    /* Configure alternate function if needed */
    if (config->mode == GPIO_MODE_AF) {
        uint32_t af_reg = (pin < 8U) ? 0U : 1U;
        uint32_t af_pos = (pin % 8U) * 4U;
        gpio->AFR[af_reg] &= ~(0xFU << af_pos);
        gpio->AFR[af_reg] |= (config->alternate << af_pos);
    }
}

/**
 * @brief Read the input data register for all pins on a GPIO port.
 *
 * @param gpio  Pointer to GPIO peripheral register block
 * @return Current IDR value (16-bit pin states, bit N = pin N state)
 */
uint16_t GPIO_ReadPort(const GPIO_TypeDef *gpio) {
    return (uint16_t)(gpio->IDR & 0xFFFFU);
}

/**
 * @brief Set or clear specific GPIO output pins using BSRR atomic write.
 *
 * Uses the Bit Set/Reset Register (BSRR) for atomic operation — no
 * read-modify-write, safe in interrupt context without disabling IRQs.
 *
 * @param gpio GPIO peripheral pointer
 * @param set_mask  Bitmask of pins to set HIGH
 * @param reset_mask Bitmask of pins to set LOW
 */
void GPIO_Write(GPIO_TypeDef *gpio, uint16_t set_mask, uint16_t reset_mask) {
    /* Upper 16 bits of BSRR = reset pins; lower 16 bits = set pins */
    gpio->BSRR = ((uint32_t)reset_mask << 16U) | (uint32_t)set_mask;
}

/**
 * @brief Configure EXTI interrupt for a GPIO pin.
 *
 * Maps the specified GPIO pin to EXTI line and configures edge trigger.
 * Caller must enable NVIC interrupt for the corresponding EXTIx_IRQn.
 *
 * @param port_index  GPIO port index: 0=A, 1=B, ..., 10=K
 * @param pin         Pin number 0-15
 * @param trigger     EXTI_TRIGGER_RISING, EXTI_TRIGGER_FALLING, or EXTI_TRIGGER_BOTH
 */
void GPIO_ConfigureEXTI(uint8_t port_index, uint8_t pin, uint8_t trigger) {
    /* Connect EXTI line to GPIO port via SYSCFG_EXTICRx */
    uint32_t cr_idx = pin / 4U;
    uint32_t cr_pos = (pin % 4U) * 4U;
    SYSCFG->EXTICR[cr_idx] &= ~(0xFU << cr_pos);
    SYSCFG->EXTICR[cr_idx] |= ((uint32_t)port_index << cr_pos);

    /* Configure edge trigger */
    if (trigger & EXTI_TRIGGER_RISING)  EXTI->RTSR |=  (1U << pin);
    else                                EXTI->RTSR &= ~(1U << pin);
    if (trigger & EXTI_TRIGGER_FALLING) EXTI->FTSR |=  (1U << pin);
    else                                EXTI->FTSR &= ~(1U << pin);

    /* Unmask EXTI line */
    EXTI->IMR |= (1U << pin);
}
""")
        print(f"  Created: {c_file.name}")

    # Header file
    h_file = TEXT_DIR / "stm32_gpio_driver.h"
    if not h_file.exists():
        h_file.write_text("""\
/**
 * @file stm32_gpio_driver.h
 * @brief STM32 GPIO driver public API.
 */
#ifndef STM32_GPIO_DRIVER_H
#define STM32_GPIO_DRIVER_H

#include <stdint.h>

/* GPIO mode values for MODER register */
#define GPIO_MODE_INPUT   0x0U  /**< Input floating */
#define GPIO_MODE_OUTPUT  0x1U  /**< General purpose output */
#define GPIO_MODE_AF      0x2U  /**< Alternate function */
#define GPIO_MODE_ANALOG  0x3U  /**< Analog mode */

/* Output type */
#define GPIO_OTYPE_PUSH_PULL   0x0U
#define GPIO_OTYPE_OPEN_DRAIN  0x1U

/* Output speed */
#define GPIO_SPEED_LOW       0x0U
#define GPIO_SPEED_MEDIUM    0x1U
#define GPIO_SPEED_HIGH      0x2U
#define GPIO_SPEED_VERY_HIGH 0x3U

/* Pull-up/down */
#define GPIO_PUPD_NONE     0x0U
#define GPIO_PUPD_PULLUP   0x1U
#define GPIO_PUPD_PULLDOWN 0x2U

/* EXTI trigger */
#define EXTI_TRIGGER_RISING   0x01U
#define EXTI_TRIGGER_FALLING  0x02U
#define EXTI_TRIGGER_BOTH     0x03U

/** GPIO pin configuration structure */
typedef struct {
    uint8_t  pin;        /**< Pin number 0-15 */
    uint8_t  mode;       /**< GPIO_MODE_* constant */
    uint8_t  otype;      /**< GPIO_OTYPE_* constant */
    uint8_t  speed;      /**< GPIO_SPEED_* constant */
    uint8_t  pull;       /**< GPIO_PUPD_* constant */
    uint8_t  alternate;  /**< Alternate function 0-15 (AF mode only) */
} GPIO_Config_t;

/* Minimal GPIO register map */
typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

/* Public API */
void     GPIO_Init(GPIO_TypeDef *gpio, const GPIO_Config_t *config);
uint16_t GPIO_ReadPort(const GPIO_TypeDef *gpio);
void     GPIO_Write(GPIO_TypeDef *gpio, uint16_t set_mask, uint16_t reset_mask);
void     GPIO_ConfigureEXTI(uint8_t port_index, uint8_t pin, uint8_t trigger);

#endif /* STM32_GPIO_DRIVER_H */
""")
        print(f"  Created: {h_file.name}")

    # Markdown documentation
    md_file = TEXT_DIR / "embedded_notes.md"
    if not md_file.exists():
        md_file.write_text("""\
# Embedded Systems Development Notes

## Clock Configuration

The system clock (SYSCLK) must be configured before initializing peripherals.
For STM32F4, the typical flow is:

1. Configure HSE (High-Speed External) crystal oscillator
2. Enable PLL and set PLL multipliers for target frequency
3. Switch SYSCLK to PLL output
4. Update SystemCoreClock variable

### APB Bus Prescalers

```c
/* Configure AHB, APB1, APB2 prescalers */
RCC->CFGR |= RCC_CFGR_HPRE_DIV1;   /* AHB = SYSCLK (168 MHz) */
RCC->CFGR |= RCC_CFGR_PPRE1_DIV4;  /* APB1 = 42 MHz (max 45 MHz) */
RCC->CFGR |= RCC_CFGR_PPRE2_DIV2;  /* APB2 = 84 MHz (max 90 MHz) */
```

The clock tree determines the maximum frequency for each peripheral domain.
APB1 clocks: TIM2-TIM7, TIM12-TIM14, USART2-3, UART4-5, SPI2-3, I2C1-3.
APB2 clocks: TIM1, TIM8-TIM11, USART1/6, SPI1/4, ADC1-3, SDIO, EXTI.

### PLL Configuration for 168 MHz

| Parameter | Value |
|-----------|-------|
| HSE freq  | 8 MHz |
| PLL_M     | 8     |
| PLL_N     | 336   |
| PLL_P     | 2     |
| SYSCLK    | 168 MHz |

## DMA Configuration

Direct Memory Access (DMA) allows peripherals to transfer data without CPU intervention.

### DMA Stream Setup

```c
/* Example: USART1 RX via DMA2 Stream 5 Channel 4 */
DMA2_Stream5->CR = 0;                          /* Disable stream first */
while (DMA2_Stream5->CR & DMA_SxCR_EN);        /* Wait for disable */

DMA2_Stream5->PAR  = (uint32_t)&USART1->DR;   /* Peripheral address */
DMA2_Stream5->M0AR = (uint32_t)rx_buffer;     /* Memory address */
DMA2_Stream5->NDTR = RX_BUFFER_SIZE;           /* Transfer count */

DMA2_Stream5->CR =
    (4U << DMA_SxCR_CHSEL_Pos) |  /* Channel 4 */
    DMA_SxCR_MINC               |  /* Memory increment */
    DMA_SxCR_CIRC               |  /* Circular mode */
    DMA_SxCR_TCIE               |  /* Transfer complete interrupt */
    DMA_SxCR_EN;                   /* Enable */
```

### DMA Interrupt Handling

The DMA controller generates interrupts for:
- Transfer complete (TC)
- Half-transfer complete (HT) — useful for double-buffering
- Transfer error (TE)
- FIFO error (FE)
- Direct mode error (DME)

## SPI Configuration

SPI supports full-duplex synchronous serial communication with up to 45 Mbit/s on APB2.

### SPI Initialization

```c
/* SPI1 @ APB2 = 84 MHz, SPI clock = 84/8 = 10.5 MHz */
SPI1->CR1 = SPI_CR1_MSTR      |  /* Master mode */
            SPI_CR1_SSM        |  /* Software slave management */
            SPI_CR1_SSI        |  /* Internal slave select */
            (2U << SPI_CR1_BR_Pos); /* Baud rate = fPCLK/8 */
SPI1->CR2 = SPI_CR2_SSOE;        /* SS output enable */
SPI1->CR1 |= SPI_CR1_SPE;        /* Enable SPI */
```

## Interrupt Handling

### NVIC Priority Groups

STM32 uses a nested vectored interrupt controller (NVIC) with configurable priority groups.
The PRIGROUP field in AIRCR divides the 4-bit priority into preemption priority and sub-priority.

| PRIGROUP | Preemption bits | Sub-priority bits |
|----------|----------------|-------------------|
| 0        | 4              | 0                 |
| 4        | 0              | 4                 |
| 7        | 4              | 0 (no groups)     |

### Critical Sections

For shared data between ISR and main context:

```c
/* Enter critical section — disable all interrupts */
uint32_t primask = __get_PRIMASK();
__disable_irq();

/* Critical section — access shared resource */
shared_buffer[head++] = data;

/* Exit critical section — restore interrupt state */
__set_PRIMASK(primask);
```

## Memory Layout

Typical STM32F407 memory regions:

| Region    | Start      | Size   | Description             |
|-----------|-----------|--------|-------------------------|
| Flash     | 0x08000000 | 1 MB   | Code + read-only data   |
| SRAM1     | 0x20000000 | 112 KB | Main data RAM           |
| SRAM2     | 0x2001C000 | 16 KB  | Secondary RAM           |
| CCMRAM    | 0x10000000 | 64 KB  | Core-coupled memory     |
| Peripherals | 0x40000000 | — | APB1/2, AHB1/2/3 buses  |

The core-coupled memory (CCMRAM) is connected directly to the Cortex-M4 bus matrix
and accessible only by the CPU (not DMA). Ideal for time-critical ISR stacks.
""")
        print(f"  Created: {md_file.name}")

    # YAML config
    yaml_file = TEXT_DIR / "project_config.yaml"
    if not yaml_file.exists():
        yaml_file.write_text("""\
# Embedded project configuration for ragrag validation

project:
  name: stm32f407-demo
  target: STM32F407VGTx
  cpu: cortex-m4
  fpu: true
  toolchain: arm-none-eabi-gcc
  version: "1.2.3"

build:
  optimization: -O2
  debug_level: -g3
  warnings:
    - -Wall
    - -Wextra
    - -Werror
    - -Wno-unused-parameter
  defines:
    - STM32F407xx
    - USE_HAL_DRIVER
    - HSE_VALUE=8000000
    - VECT_TAB_OFFSET=0x0

memory:
  flash:
    start: 0x08000000
    size_kb: 1024
  sram1:
    start: 0x20000000
    size_kb: 112
  sram2:
    start: 0x2001C000
    size_kb: 16
  ccmram:
    start: 0x10000000
    size_kb: 64

clock:
  hse_frequency_hz: 8000000
  sysclk_hz: 168000000
  ahb_prescaler: 1
  apb1_prescaler: 4
  apb2_prescaler: 2
  pll:
    m: 8
    n: 336
    p: 2
    q: 7

peripherals:
  uart1:
    baudrate: 115200
    tx_pin: PA9
    rx_pin: PA10
    dma: DMA2_Stream7_CH4
  spi1:
    mode: master
    clock_hz: 10500000
    cpol: 0
    cpha: 0
    nss: software
  i2c1:
    speed: fast   # 400 kHz
    sda_pin: PB7
    scl_pin: PB6
""")
        print(f"  Created: {yaml_file.name}")


def create_image_fixture() -> None:
    """Create a synthetic SPI timing diagram image."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / "timing_diagram.png"
    if img_path.exists():
        print(f"  SKIP (exists): {img_path.name}")
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  WARNING: Pillow not available, creating minimal PNG")
        # Write a minimal 1x1 white PNG
        img_path.write_bytes(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
            b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        return

    W, H = 900, 300
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((W // 2 - 120, 10), "SPI Timing Diagram", fill="black")

    # Signal labels
    signals = ["CLK", "CS#", "MOSI", "MISO"]
    y_positions = [70, 110, 150, 190]
    label_x = 30

    for label, y in zip(signals, y_positions):
        draw.text((label_x, y - 10), label, fill="black")

    # Draw waveforms
    timeline_start = 100
    bit_width = 70

    def draw_signal(draw, y_base, transitions, high=True):
        """Draw a digital signal waveform."""
        HIGH = y_base - 20
        LOW = y_base + 10
        x = timeline_start
        prev_level = LOW
        for t, level in transitions:
            x_next = timeline_start + t * bit_width
            draw.line([(x, prev_level), (x_next, prev_level)], fill="blue", width=2)
            draw.line([(x_next, prev_level), (x_next, level)], fill="blue", width=2)
            x = x_next
            prev_level = level
        draw.line([(x, prev_level), (W - 50, prev_level)], fill="blue", width=2)

    # CLK: alternating
    clk_y = y_positions[0]
    clk_h = clk_y - 20
    clk_l = clk_y + 10
    clk_trans = [(i, clk_l if i % 2 == 0 else clk_h) for i in range(10)]
    draw_signal(draw, clk_y, clk_trans)

    # CS#: low for duration
    cs_y = y_positions[1]
    cs_h = cs_y - 20
    cs_l = cs_y + 10
    draw_signal(draw, cs_y, [(0, cs_h), (1, cs_l), (9, cs_h)])

    # MOSI: data pattern
    mosi_y = y_positions[2]
    mosi_h = mosi_y - 20
    mosi_l = mosi_y + 10
    pattern = [1, 0, 1, 1, 0, 1, 0, 0]
    mosi_trans = [(0, mosi_h)]
    for i, bit in enumerate(pattern):
        level = mosi_h if bit else mosi_l
        mosi_trans.append((i + 1, level))
    draw_signal(draw, mosi_y, mosi_trans)

    # MISO: different data
    miso_y = y_positions[3]
    miso_h = miso_y - 20
    miso_l = miso_y + 10
    pattern2 = [0, 1, 0, 1, 1, 0, 1, 1]
    miso_trans = [(0, miso_l)]
    for i, bit in enumerate(pattern2):
        level = miso_h if bit else miso_l
        miso_trans.append((i + 1, level))
    draw_signal(draw, miso_y, miso_trans)

    # Time axis
    draw.line([(timeline_start, 230), (W - 50, 230)], fill="gray")
    draw.text((W // 2 - 30, 240), "Time →", fill="gray")

    img.save(str(img_path), "PNG")
    print(f"  Created: {img_path.name} ({img_path.stat().st_size:,} bytes)")


def create_unsupported_fixture() -> None:
    """Create a fake ELF binary to test unsupported file handling."""
    UNSUPPORTED_DIR.mkdir(parents=True, exist_ok=True)
    elf_path = UNSUPPORTED_DIR / "firmware.elf"
    if not elf_path.exists():
        # Minimal fake ELF header (first 4 bytes are ELF magic)
        elf_data = b'\x7fELF\x02\x01\x01\x00' + b'\x00' * 120
        elf_path.write_bytes(elf_data)
        print(f"  Created: {elf_path.name} ({len(elf_data)} bytes)")
    else:
        print(f"  SKIP (exists): {elf_path.name}")


def main() -> int:
    print("=== ragrag validation corpus setup ===\n")
    
    errors = []

    # Create synthetic fixtures first (no network needed)
    print("[1/3] Creating synthetic text fixtures...")
    create_synthetic_text_files()

    print("\n[2/3] Creating image fixture...")
    create_image_fixture()

    print("\n[2.5/3] Creating unsupported file fixture...")
    create_unsupported_fixture()

    # Download PDFs
    print("\n[3/3] Downloading PDF datasheets...")
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    
    for info in PDF_DOWNLOADS:
        print(f"\n  [{info['description']}]")
        success = download_file(
            info["url"],
            PDFS_DIR / info["filename"],
            info.get("min_size_bytes", 0),
        )
        if not success and not (PDFS_DIR / info["filename"]).exists():
            errors.append(info["filename"])

    # Summary
    print("\n=== Summary ===")
    all_files = list(FIXTURES.rglob("*"))
    real_files = [f for f in all_files if f.is_file()]
    print(f"Total fixture files: {len(real_files)}")
    for f in sorted(real_files):
        rel = f.relative_to(FIXTURES)
        print(f"  {rel} ({f.stat().st_size:,} bytes)")

    if errors:
        print(f"\nERRORS: Failed to download: {errors}")
        print("These PDFs are required for full validation. Check network connectivity.")
        return 1
    
    print("\nAll fixtures ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
