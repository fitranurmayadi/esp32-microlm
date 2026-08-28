idf_component_register(
    SRCS 
        "main.cpp"
        "kibo_inference.cpp"
        "kibo_model_data.S"
    INCLUDE_DIRS "."
    REQUIRES 
        esp_timer
        esp_partition
        freertos
        driver
        esp_system
        spi_flash
        esp_psram
)
