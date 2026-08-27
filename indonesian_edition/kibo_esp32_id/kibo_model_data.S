.section .rodata.kibo_model, "a"
.global kibo_embedded_model_start
.global kibo_embedded_model_end
.align 4

kibo_embedded_model_start:
.incbin "indonesian_edition/kibo_esp32_id/kibo_model_int8.bin"
kibo_embedded_model_end:
