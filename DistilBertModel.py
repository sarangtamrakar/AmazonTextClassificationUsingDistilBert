import spacy
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datetime import datetime


class DistilBertModelClass:
    def __init__(self):
        self.ModelDir = "ModelDir/"
        self.label_dict = {0:1,1:2,2:3,3:4,4:5}

    def LoadModel(self):
        # distilModel = DistilBertForSequenceClassification.from_pretrained(self.ModelDir)
        # model_int8 = torch.quantization.quantize_dynamic(
        # distilModel,  # the original model
        # {torch.nn.Linear},  # a set of layers to dynamically quantize
        # dtype=torch.qint8)

        quantizeModel = torch.load("ModelDir/quantized_model.pt")
        print("load the model")
        return quantizeModel

    def cleanText(self, text):
        nlp = spacy.load("en_core_web_sm")
        text = text.lower()
        text = " ".join(re.sub("[^a-z]", " ", text).split())
        doc = nlp(text)
        sent = " ".join([token.lemma_ for token in doc])
        return sent

    def TokenizeText(self, text):
        sent = self.cleanText(text)
        tokenizer = DistilBertTokenizer.from_pretrained(self.ModelDir)
        encodings = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_id = encodings["input_ids"]
        mask = encodings["attention_mask"]

        return input_id, mask

    def Predict(self, text, Model):
        input_id, mask = self.TokenizeText(text)
        output = Model(input_id, mask)[0]
        # prediction_value = tf.argmax(output, axis=1).numpy()[0] to do in tf
        prediction_value = torch.argmax(output, dim=1).item()  # correct way in torch
        # prediction_value = np.argmax(output.detach().numpy()) way to do in numpy
        final_class = self.label_dict[prediction_value]
        return final_class
