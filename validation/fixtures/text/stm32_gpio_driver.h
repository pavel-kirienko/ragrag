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
