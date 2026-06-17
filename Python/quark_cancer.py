from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp

dataframe = pd.read_csv("data/breast-cancer/data.csv")
x = dataframe.drop(["diagnosis", "id"], axis=1)
scaler = pp.MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x).astype(np.float32)


class WDBCDataReader(CalibrationDataReader):
    def __init__(self, data, n=200):
        self.data = data[:n]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        sample = {"input": self.data[self.idx:self.idx+1]}
        self.idx += 1
        return sample


onnx = "data/breast-cancer/cancer_mlp.onnx"
quantize_static(
    model_input=onnx,
    model_output=onnx.replace(".onnx", "_int8_quark.onnx"),
    calibration_data_reader=WDBCDataReader(x_scaled),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)
