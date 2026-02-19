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
