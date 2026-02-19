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
