import json
from flask import Flask, request, jsonify
from os.path import isfile
import sys
import numpy as np
import pickle
import time
import re
import tensorflow as tf
from tensorflow import keras
import cv2
from . import imageOCRScripts as ocr
from TranslationModules import Encode,Decoder,Transformer,CustomSchedule



app = Flask(__name__)
letter_count1= {0: 'CHECK', 1: 'क', 2: 'ख', 3: 'ग', 4: 'घ', 5: 'ङ', 6: 'च',
                    7: 'छ', 8: 'ज', 9: 'झ', 10: 'ञ',
                    11: 'ट',
                    12: 'ठ', 13: 'ड', 14: 'ढ', 15: 'ण', 16: 'त', 17: 'थ',
                    18: 'द',

                    19: 'ध', 20: 'न', 21: 'प', 22: 'फ',
                    23: 'ब',
                    24: 'भ', 25: 'म', 26: 'य', 27: 'र', 28: 'ल', 29: 'व', 30: 'श',
                    31: 'ष',32: 'स', 33: 'ह',
                    34: 'क्ष', 35: 'त्र', 36: 'ज्ञ',
                    37: 'CHECK'}# 38: '1',39: '2',40: '3',41:'4',42:'5',43:'6',44:'7',45:'8',46:'9',
                    #47: 'अ',48: 'आ ',49: 'इ ',50: 'ई',51: 'उ',52: 'ऊ',53: 'ए',
                    #54: 'ऐ',55: 'ओ',56: 'औ',57:'अं',58:'अः',
                    #59: 'CHECK'}
letter_count={0:'check',1: 'ka', 2: 'kha', 3: 'ga', 4: 'gha',
              5: 'kna', 6: 'cha', 7: 'chha', 8: 'ja', 9: 'jha',
              10: 'yna', 11: 'taamatar', 12: 'thaa', 13: 'daa', 14: 'dhaa',
              15: 'adna', 16: 'tabala', 17: 'tha', 18: 'da',
              19: 'dha', 20: 'na', 21: 'pa', 22: 'pha', 23: 'ba',
              24: 'bha', 25: 'ma', 26: 'yaw', 27: 'ra', 28: 'la',
              29: 'waw', 30: 'motosaw', 31: 'petchiryakha',
              32: 'patalosaw', 33: 'ha', 34: 'chhya', 35: 'tra',
              36: 'gya', 37: 0}

# Set the model Hyperparams
num_layers = 6
d_model = 64
dff = 2048
num_heads = 32
dropout_rate = 0.1
tokenizer_one = pickle.load(open("tokenizer_hi","rb"))
tokenizer_two = pickle.load(open("tokenizer_en","rb"))

# creating the embedding Matrices
# Define the vocab sizes

input_vocab_size = len(tokenizer_one.word_index) + 2
target_vocab_size = len(tokenizer_two.word_index) + 2





# create the optimizer object with the custom learning rate

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
learning_rate,
beta_1=0.9,
beta_2=0.98,
    epsilon=1e-9)


# Define the loss for the model

# Create the required type of loss

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none')

    # Define how loss is calculated

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Create the loss and accuracy objects

train_loss = tf.keras.metrics.Mean(
    name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')



transformer = Transformer(num_layers, d_model, num_heads, dff,
                        input_vocab_size, target_vocab_size,
                        pe_input=input_vocab_size,
                        pe_target=target_vocab_size,
                        rate=dropout_rate)

# Load  pre-trained weights
transformer.load_weights('transformer_htoe.weights')


# Utility functions for prediction

MAX_LENGTH = 64

def preprocess_string(s):
    ''' String preprocessing function

    Args:
        s: The string to be preprocessed

    Returns:
        s: The preprocessed String
    '''
    s = re.sub(r'[a-zA-Z]', '', s) # Removes english chars from hindi text
    s = re.sub(r"[\(\[].*?[\)\]]", "", s) # Removes text between braces
    s = re.sub(r'([!.?।])', r' \1', s) #Includes space between some characters
    s = re.sub(r'\s+', r' ', s) #Reduces multispace string to a single space

    return s

def evaluate(inp_sentence):
    start_token = [len(tokenizer_one.word_index)]
    end_token = [len(tokenizer_one.word_index) + 1]

    inp_sentence = start_token + tokenizer_one.texts_to_sequences(
        [inp_sentence])[0] + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [len(tokenizer_two.word_index)]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(64):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == len(tokenizer_two.word_index)+1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def translate(sentence):
    result, attention_weights = evaluate(sentence)
    predicted_sentence = tokenizer_two.sequences_to_texts(
        [[i.numpy()] for i in result if i < len(tokenizer_two.word_index)])

    return ' '.join(predicted_sentence)


##############------------------Translation Module Ends--------------------#################

##############---------------------OCR Module Ends--------------------------#################
def preprocess(src_img):
    copy = src_img.copy()
    height = src_img.shape[0]
    width = src_img.shape[1]
    print("\n Resizing Image........")
    src_img = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)

    height = src_img.shape[0]
    width = src_img.shape[1]

    print("\tHeight =",height,"\n\tWidth =",width)
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# Applying Adaptive Threshold with kernel
    bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
    coords = np.column_stack(np.where(bin_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    center = (w//2,h//2)

    angle = 0
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    bin_img = cv2.warpAffine(bin_img,M,(w,h),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    bin_img1 = bin_img.copy()

    bin_img2 = bin_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)

#Noise Removal From Image
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contr_retrival = final_thr.copy()
    count_x = np.zeros(shape= (height))
    for y in range(height):
        for x in range(width):
            if bin_img[y][x] == 255 :
                count_x[y] = count_x[y]+1

    local_minima = []
    for y in range(len(count_x)):
        if y >= 10 and y <= len(count_x)-11:
            arr1 = count_x[y-10:y+10]
        elif y < 10:
            arr1 = count_x[0:y+10]
        else:
            arr1 = count_x[y-10:len(count_x)-1]
        if min(arr1) == count_x[y]:
            local_minima.append(y)
    final_local = []
    init = []
    end = []
    for z in range(len(local_minima)):

        if z != 0 and z!= len(local_minima)-1:
            if local_minima[z] != (local_minima[z-1] +1) and local_minima[z] != (local_minima[z+1] -1):
                final_local.append(local_minima[z])
            elif local_minima[z] != (local_minima[z-1] + 1) and local_minima[z] == (local_minima[z+1] -1):
                init.append(local_minima[z])
            elif local_minima[z] == (local_minima[z-1] + 1) and local_minima[z] != (local_minima[z+1] -1):
                end.append(local_minima[z])
        elif z == 0:
            if local_minima[z] != (local_minima[z+1]-1):
                final_local.append(local_minima[z])
            elif local_minima[z] == (local_minima[z+1]-1):
                init.append(local_minima[z])
        elif z == len(local_minima)-1:
            if local_minima[z] != (local_minima[z-1]+1):
                final_local.append(local_minima[z])
            elif local_minima[z] == (local_minima[z-1]+1):
                end.append(local_minima[z])
    for j in range(len(init)):
        mid = (init[j] + end[j])/2
        if (mid % 1) != 0:
            mid = mid+0.5
        final_local.append(int(mid))

    final_local = sorted(final_local)

    no_of_lines = len(final_local) - 1

    lines_img = []

    for i in range(no_of_lines):
        lines_img.append(bin_img2[final_local[i]:final_local[i+1], :])

    contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    mean_lttr_width = letter_width(contours)
    print("\nAverage Width of Each Letter:- ", mean_lttr_width)
    x_lines = []

    for i in range(len(lines_img)):
        x_lines.append(end_wrd_dtct(final_local, i, bin_img, mean_lttr_width))

    for i in range(len(x_lines)):
        x_lines[i].append(width)



    #-------------/Word Detection-----------------#

    for i in range(no_of_lines):
        letter_seg(lines_img, x_lines, i)


    chr_img = bin_img1.copy()

    contours, hierarchy = cv2.findContours(chr_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    for cnt in contours:
        cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,0,255),4)

    plt.imshow(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = np.copy(thresh)
    cols = horizontal.shape[1]
    horizontal_size = cols //10
    print(horizontal_size)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray, [c], -1, (255,0,0), 2)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x,y,w,h = cv2.boundingRect(cnt)
        if y-5>0:
            cv2.rectangle(img,(x,y-5),(x+w,y+h),(0,0,255),1)
            cropped = img[y-5:y + h, x:x + w]
        else:
            cv2.rectangle(img,(x,0),(x+w,h),(0,0,255),1)
            cropped = img[0:h, x:x + w]
        gray_image = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        capp=cv2.bitwise_not(gray_image)

        prob,classs,letter = keras_predict(model1, capp)
        if str(int(prob)*100)!='0' and classs!=31 :
            cv2.putText(img,letter_count[classs], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
            print(letter_count1[classs],str(int(prob)*100)+"%")


def keras_predict(model, image):
        processed = preprocess(image)
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        pred_class="".join(pred_class)
        return pred_class

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




@app.route('/OCRImage',methods=['POST'])
def OCRRequestHandler():
  if 'file' not in request.files:
    print('File not found')
    return jsonify({
      'result':'Error file not found'
      })
  image=request.files['files'].read()
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  model = keras.models.load_model('devnagri1.h5')
  text=keras_predict(model,image)
  return jsonify({
        'text': 'text'
    })





@app.route('/',methods=['GET'])
def requestHandler():
    return jsonify({
        'orignalText': 'text',
        'translatedText':'translatedText'
    })
@app.route('/translater',methods=['GET'])
def GetRequestHandler():
    text = request.args.get('text')
    translatedText=translate(text)
    return jsonify({
        'orignalText': text,
        'translatedText':translatedText
    })

@app.route('/translate',methods=['POST'])
def postRequestHandler():
    text = json.loads(request.text)
    translatedText=translate(text)
    return jsonify({
        'orignalText': text,
        'translatedText':translatedText
    })