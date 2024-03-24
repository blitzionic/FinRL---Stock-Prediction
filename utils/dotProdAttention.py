from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer
from keras.backend import softmax

'''
size of model for now, actual sizes will be depending on data fetched
d_k = 64
d_v = 64
batch_size = 64
input_seq_length = 5 ? (maybe dates can be ommited?)
input sequence length, queries,keys, values determined by word tokenization and embedding
'''




class dotProductAttention(Layer):
    def __init__(self,**kwargs):
        super(dotProductAttention,self).__init__(**kwargs)


    def call(self, queries,keys,values, d_k,mask = None):
        #score queries against keys from transposing and scaling
        scores = matmul(queries, keys,tranpose_b =True ) / math.sqrt(cast(d_k, float32))
        if mask != None:
            scores += -1e9 * mask

        weights = softmax(scores)

        return matmul(weights, values) #computed attention
        


#layersss!!








attention = dotProductAttention()