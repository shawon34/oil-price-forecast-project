import joblib
import numpy as np

print("NumPy version:", np.__version__)
print("NumPy path:", np.__file__)

try:
    model = joblib.load('trained_oil_price_model.sav')
    print("Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print("Model loading failed:", e)
    import traceback
    traceback.print_exc()