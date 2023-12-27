import numpy as np
from flask import Flask, request, render_template

import pyedflib as edf
import numpy as np



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


4



app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    result=""
    image_file = request.files['image']
    image_source = "data/"+image_file.filename
    f = edf.EdfReader(image_source)

    n = f.signals_in_file
    features = []
    #for i in range(n):
    #   signal = f.readSignal(i)
    #  features.append(signal)
    signal_labels2 = f.getSignalLabels()
    fz_cz_index = signal_labels2.index('FZ-CZ')
    cz_pz_index = signal_labels2.index('CZ-PZ')
    buffers = np.zeros((2, f.getNSamples()[0]))
    buffers[0] = f.readSignal(fz_cz_index)
    buffers[1] = f.readSignal(cz_pz_index)
    array_buffer = np.array(buffers)
    features.append(array_buffer)

    from tensorflow.keras.models import load_model
    loaded_model = load_model("model")
    X = np.array(features).reshape(-1,1280,2)

    # Predict the probabilities for each class
    probs = loaded_model.predict(X)

    # Apply a threshold of 0.5 to obtain binary predictions
    binary_preds = (probs[:, 0] >= 0.75).astype(int)

    # Print the binary predictions

    if(sum(binary_preds)/len(binary_preds)<=0.87):
        result="There is a possibility of SEIZURE in your EEG. Please consult a Doctor on urgent basis "
    else:
        result="There seems NO possibility of seizure in your EEG. Still consult a Doctor for further steps"

    return render_template('index.html', prediction=result)




if __name__ == "__main__":
    app.run(debug=True)
