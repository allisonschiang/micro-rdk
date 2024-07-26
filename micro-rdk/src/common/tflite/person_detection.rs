use crate::esp32::esp_idf_svc::sys::tflite::{
    g_person_detect_model_data
};

#[derive(Error, Debug)]
pub enum TfliteError {
    // #[error("pin {0} error: {1}")]
    // GpioPinError(u32, &'static str),
    // #[error("pin {0} error: {1}")]
    // GpioPinOtherError(u32, Box<dyn std::error::Error + Send + Sync>),
    // #[error("analog reader {0} not found")]
    // AnalogReaderNotFound(String),
    // #[error("board unsupported argument {0} ")]
    // BoardUnsupportedArgument(&'static str),
    // #[error("i2c bus {0} not found")]
    // I2CBusNotFound(String),
    // #[error(transparent)]
    // OtherBoardError(#[from] Box<dyn std::error::Error + Send + Sync>),
    // #[error("method: {0} not supported")]
    // BoardMethodNotSupported(&'static str),
    // #[error(transparent)]
    // BoardI2CError(#[from] I2CErrors),
    // #[error(transparent)]
    // #[cfg(feature = "esp32")]
    // EspError(#[from] EspError),
    #[error(transparent)]
    ModelVersionError(&'static str),
}

[DoCommand]
pub fn person_detection()->Result<Str, ModelError>{
    let model = Model
    let interpreter = MicroInterpreter
    let input = TfLiteTensor


    if let e =setup(model, ){
        return err()
    }
    else{
        while true{
            run_model()
        }
    }

}

pub fn setup() -> Option<SetupError>{

    model = tflite::GetModel(g_person_detect_model_data);

    if (model.version != TFLITE_SCHEMA_VERSION) {
        TfliteError::ModelVersionError("Model provided is schema version %d not equal to supported version")
    }

    if (tensor_arena == NULL) {
        tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return; 
    }  

    
    // Get information about the memory area to use for the model's input.
    input = interpreter->input(0);

}

pub fn run_model() -> Option<ModelError>{

}