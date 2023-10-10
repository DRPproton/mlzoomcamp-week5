import pickle


with open('model1.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

print(model, dv)

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform(client)

y_pred = model.predict_proba(X)[0,1]

print(y_pred)

