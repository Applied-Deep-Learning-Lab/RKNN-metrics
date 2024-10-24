from rknnlite.api import RKNNLite


class RKNN_model_container():
    def __init__(self, model_path, core_mask: int = RKNNLite.NPU_CORE_0) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        ret = rknn.load_rknn(model_path)
        if ret:
            raise OSError(f"{model_path}: Loading RKNN model failed!")

        print('--> Init runtime environment')
        ret = rknn.init_runtime(core_mask=core_mask)
        if ret:
            raise OSError(f"{model_path}: Init runtime enviroment failed!")
        print('done')
        
        self.rknn = rknn 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result
    